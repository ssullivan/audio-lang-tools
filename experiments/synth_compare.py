"""Experiment: analysis by synthesis. For a clip read from characters,
compare each syllable's pitch with Azure's sapi readings of the expected
jyutping and of each other tone at that syllable, in the same context.
Good clips: the word's characters. Wrong clips: one character swapped for
another with the same syllable in a different tone (今日 → 琴日).

    uv run python experiments/synth_compare.py [n]

Result (2026-09-28, 94 words, CREPE tiny model): AUC for telling wrong
from right clips 0.935 for this margin vs 0.939 for the tone classifier;
at 5% false flags it caught 57% vs 60%; combined, 64%. Not adopted: too
little gain for ten extra TTS calls a clip. It also showed that
character-read clips are much harder than sapi-read ones, which is why the
benchmark's hanzi kinds exist.
"""
import json
import random
import sys

import numpy as np
from pycantonese.data.rime_cantonese import CHARS_TO_JYUTPING

from audio_lang_tools import bench, config, tts
from audio_lang_tools.check import Analyser, Checker
from audio_lang_tools.lang import cantonese as y
from audio_lang_tools.tones import _resample, measure

VOICE = bench.VOICE
A = Analyser()
checker = Checker(words=False)
ref_hz = checker.model.refs[VOICE]
clips = config.cache('experiments')


def clip_for(body, name):
    p = clips / f'{name}.mp3'
    p.write_bytes(tts.say(body, VOICE))
    return str(p)


def contours(path, sylls):
    a = A.analyse(path, sylls)
    m, _ = measure(a['t'], a['f0'], a['per'], a['spans'], ref_hz)
    return [_resample(s['abs'], 10) if s['abs'] is not None else None for s in m]


def swap_char(text, sylls, rng):
    """(i, char, syllable): a character reading base(sylls[i]) in another tone."""
    opts = []
    for i, s in enumerate(sylls):
        for t in y.tones_for(s):
            alt = y.base(s) + str(t)
            if t == y.tone(s) or not y.sayable(alt, VOICE):
                continue
            freq = y.syllable_frequency()
            for c in y.chars_by_syllable().get(alt, [])[:6]:
                if CHARS_TO_JYUTPING.get(c) == alt and '一' <= c <= '鿿' and freq[alt] >= 3:
                    opts.append((i, c, alt))
                    break
    return rng.choice(opts) if opts else None


def main(n):
    rng = random.Random(1)
    items = [i for i in bench.load_manifest() if i['split'] == 'test' and i['kind'] == 'good']
    rng.shuffle(items)
    rows = []
    for it in items[:n]:
        sylls = y.syllables(it['jyutping'])
        text = it['text']
        sw = swap_char(text, sylls, rng)
        if not sw:
            continue
        i, c, alt = sw
        wrong_text = text[:i] + c + text[i + 1:]
        try:
            good = clip_for(text, f"good-{it['id']}")
            wrong = clip_for(wrong_text, f"wrong-{it['id']}")
            # Variants: the expected reading, and each other tone at each syllable.
            variants = {}
            for k, s in enumerate(sylls):
                for t in y.tones_for(s):
                    alt_s = y.base(s) + str(t)
                    if not y.sayable(alt_s, VOICE):
                        continue
                    v = sylls[:k] + [alt_s] + sylls[k + 1:]
                    variants[(k, t)] = contours(clip_for(tts.phonemes(v, text), f"var-{'_'.join(v)}"), v)
        except tts.Rejected:
            continue
        for kind, path in (('good', good), ('wrong', wrong)):
            cs = contours(path, sylls)
            res = checker.check({'path': path, 'jyutping': it['jyutping'], 'voice': VOICE})
            worst_margin, worst_p = 99, 1.0
            for k, s in enumerate(sylls):
                d = {t: float(np.mean(np.abs(cs[k] - variants[(k, t)][k])))
                     for (kk, t) in variants if kk == k and cs[k] is not None and variants[(k, t)][k] is not None}
                want = y.tone(s)
                if want in d and len(d) > 1:
                    margin = min(v for t, v in d.items() if t != want) - d[want]
                    worst_margin = min(worst_margin, margin)
                worst_p = min(worst_p, res['tones'][k]['p'] if res['tones'] else 1)
            rows.append({'id': it['id'], 'text': text if kind == 'good' else wrong_text, 'kind': kind,
                         'margin': worst_margin, 'p': worst_p, 'n': len(sylls), 'swap': f'{alt}' if kind == 'wrong' else ''})
        print(len(rows) // 2, end=' ', file=sys.stderr, flush=True)
    json.dump(rows, open(clips / 'synth_compare.json', 'w'), ensure_ascii=False)
    from sklearn.metrics import roc_auc_score
    lab = np.array([r['kind'] == 'wrong' for r in rows])
    m = np.array([r['margin'] for r in rows]); p = np.array([r['p'] for r in rows])
    print(f'\n{len(rows)} clips. AUC (wrong vs good): synthesis margin {roc_auc_score(lab, -m):.3f}, classifier {roc_auc_score(lab, -p):.3f}')
    for fp in (0.03, 0.05, 0.1):
        tm = np.quantile(m[~lab], fp); tp = np.quantile(p[~lab], fp)
        print(f'  at {fp:.0%} false flags: synthesis catches {np.mean(m[lab] < tm):.1%}, classifier {np.mean(p[lab] < tp):.1%}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
