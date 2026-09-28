"""Check clips: file faults, words (speech-to-text) and tones.

    Checker().check({ id, path, jyutping, text?, voice? }) → {
      id, secs, problems: [...], notes: [...], heard, heard_jyutping,
      calibrated, tones: [{ syll, want, heard, p, probs }] }

problems are likely wrong (CHECK); notes are worth a listen (LISTEN).
words is False when the word check was skipped: no text, or speech-to-text
unavailable (quota used up: the rest of the run uses cached results only,
and says so once on stderr).

Alignment and pitch are cached per clip (by content hash), so a rerun
only analyses new or changed clips.
"""
import hashlib
import sys

import numpy as np

from . import audio, config, pitch, stt
from .align import Aligner
from .lang import cantonese as lang
from .tones import Model, features, measure


class Analyser:
    """Decoding, alignment and pitch for a clip, cached."""

    def __init__(self):
        self._aligner = None

    @property
    def aligner(self):
        if self._aligner is None:
            self._aligner = Aligner()
        return self._aligner

    def analyse(self, path, sylls):
        # Alignment ignores tone, so a clip is analysed once whatever tones
        # it is asked about.
        data = open(path, 'rb').read()
        tracker = {'crepe-tiny': 'tiny', 'crepe-full': 'full'}.get(pitch.BACKEND, pitch.BACKEND)  # older cache keys
        key = hashlib.sha1(data + ' '.join(lang.base(s) for s in sylls).encode() + tracker.encode()).hexdigest()
        cached = config.cache('analysis') / f'{key}.npz'
        if cached.exists():
            z = np.load(cached, allow_pickle=True)
            return {k: z[k] for k in z.files} | {'spans': list(z['spans'])}
        x = audio.decode(path)
        secs, problems = audio.basics(x, len(sylls))
        spans = self.aligner.align(x, [lang.base(s) for s in sylls], [lang.rhyme_at(s) for s in sylls]) if secs > 0 else []
        t, f0, per = pitch.track(x) if secs > 0 else (np.zeros(0),) * 3
        out = {'secs': np.float64(secs), 'problems': np.array(problems, dtype=object), 'spans': spans,
               't': t, 'f0': f0, 'per': per}
        np.savez(cached, **{**out, 'spans': np.array(spans, dtype=object)})
        return out


def syllable_rows(a, sylls, ref_hz):
    """Feature rows for an analysed clip (None if it has no speech)."""
    if not len(a['spans']):
        return None
    measured, med = measure(a['t'], a['f0'], a['per'], a['spans'], ref_hz)
    return features(sylls, [lang.checked(s) for s in sylls], measured, med, ref_hz)


class Checker:
    def __init__(self, model=None, words=True):
        self.model = model or Model.load()
        self.analyser = Analyser()
        self.words = words
        self.stt_offline = False

    def check(self, item):
        sylls = lang.syllables(item['jyutping'])
        voice = item.get('voice')
        r = {'id': item.get('id'), 'problems': [], 'notes': [], 'tones': []}
        try:
            a = self.analyser.analyse(item['path'], sylls)
        except Exception as err:  # ffmpeg can't decode it
            r['problems'].append(f"can't decode: {str(err).splitlines()[0][:120]}")
            return r
        r['secs'] = round(float(a['secs']), 2)
        r['problems'] += list(a['problems'])

        heard = None
        if self.words and item.get('text') and r['secs'] > 0:
            try:
                heard = stt.heard(item['path'], offline=self.stt_offline)
            except stt.Unavailable as err:
                self.stt_offline = True
                sys.stderr.write(f'\nspeech-to-text unavailable ({err}); word checks use cached results only from here on\n')
        r['words'] = heard is not None
        if heard is not None:
            readings = lang.heard_readings(heard)
            r['heard'], r['heard_jyutping'] = heard, ' '.join(readings[0]) if readings else ''
            want_han = ''.join(c for c in item['text'] if '㐀' <= c <= '鿿')
            # The same characters, or the same syllables ignoring tone
            # (digits read every way they can be).
            if heard != want_han and [lang.base(s) for s in sylls] not in readings:
                r['notes'].append(f"speech-to-text heard {heard or 'nothing'} {r['heard_jyutping']}")

        r['calibrated'] = self.model.knows(voice)
        rows = syllable_rows(a, sylls, self.model.refs.get(voice) if r['calibrated'] else None)
        if rows is None:
            return r
        probs = self.model.probs(rows, voice)
        chars = [c for c in item.get('text', '') if '㐀' <= c <= '鿿']
        chars = chars if len(chars) == len(sylls) else [lang.base(s) for s in sylls]
        th = self.model.thresholds[voice if r['calibrated'] else 'shape']
        doubts, sure = [], []
        for i, s in enumerate(sylls):
            want = lang.tone(s)
            p = probs[i]
            heard = int(np.argmax(p)) + 1
            r['tones'].append({'syll': s, 'char': chars[i], 'want': want, 'heard': heard,
                               'p': round(float(p[want - 1]), 3), 'probs': [round(float(v), 3) for v in p]})
            if p[want - 1] < th['check']:
                sure.append(i)
            elif p[want - 1] < th['listen']:
                doubts.append(i)
        say = lambda i: (f"{chars[i]} tone {r['tones'][i]['want']} sounds like {r['tones'][i]['heard']}"
                         f" (p {r['tones'][i]['p']:.2f})")
        if sure:
            # A voice without calibration can't be judged on height, so
            # even a clear doubt is only worth a listen.
            (r['problems'] if r['calibrated'] else r['notes']).append('tone: ' + '; '.join(map(say, sure)))
        if doubts:
            r['notes'].append('tone doubtful: ' + '; '.join(map(say, doubts)))
        return r
