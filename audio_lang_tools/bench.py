"""The benchmark: labelled clips that measure the checker, and the tone
model's training data.

Azure honours tones in sapi phonemes (except each voice's GAPS), so right
and wrong readings can be made on demand:

  good        a word read from its exact jyutping
  wrong-tone  the same with one syllable's tone changed to another tone
              that syllable really has (an attested syllable+tone)
  hanzi-good  a word read from its characters, as a site makes most clips
              (only words marked plain)
  hanzi-wrong the same with one character swapped for a common one whose
              usual reading is that syllable in another tone (今日 → 琴日)

The hanzi kinds are the realistic measure: sapi readings are cleaner than
the voice's natural ones, so the checker scores higher on them. Their
labels are a little noisy (the voice may misread a character, or say a
question-final 呀 high), so a few "false" flags there are real findings.
  single      (train only) an attested syllable alone, in each of its tones
  string      (train only) 2–5 random attested syllables in a row
  fault       (test only) a good clip made silent, cut short or tripled
  fixture     real cases with a known answer (bench/fixtures.json)

Words are split into train and test by a hash of their text, so a word is
always on the same side and none is on both.

    make(words, voice)  words: [{ id, text, jyutping }] (e.g. a site's vocab)
    train()             tone model from the train split → Model.save()
    score(results?)     the checker on the test split (or `results` from
                        another checker: [{ id, problems, notes }]), against
                        bench/thresholds.json
"""
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from . import config, tts
from .lang import cantonese as lang

VOICE = 'zh-HK-HiuMaanNeural'
TEST_SHARE = 0.35
FIXTURES = config.ROOT / 'bench/fixtures.json'
THRESHOLDS = config.ROOT / 'bench/thresholds.json'


def bench_dir() -> Path:
    return config.cache('bench')


def manifest_path() -> Path:
    return bench_dir() / 'manifest.json'


def load_manifest() -> list[dict]:
    p = manifest_path()
    if not p.exists():
        raise SystemExit('No benchmark yet: run `altools bench make --words <file>` first.')
    return json.loads(p.read_text())


def split_of(text: str) -> str:
    h = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xffffffff
    return 'test' if h < TEST_SHARE else 'train'


def han(text: str) -> str:
    return ''.join(c for c in text if '㐀' <= c <= '鿿')


def progress(i, n, what):
    sys.stderr.write(f'\r{what} {i}/{n}')
    if i == n:
        sys.stderr.write('\n')


# ---------------------------------------------------------------- make

def _wrong_tone(sylls, voice, rng):
    """(index, new syllable) changing one syllable to another tone it
    really has, or None."""
    options = [(i, lang.base(s) + str(t)) for i, s in enumerate(sylls)
               for t in lang.tones_for(s) if t != lang.tone(s) and lang.sayable(lang.base(s) + str(t), voice)]
    return rng.choice(options) if options else None


def _chars_for(sylls):
    return ''.join(lang.chars_by_syllable()[s][0] for s in sylls)


def make(words: list[dict], voices: list[str] = (VOICE,), seed: int = 0, singles: int = 1200, strings: int = 900):
    """Every voice gets the sapi kinds (the model learns each voice, and the
    score reports each); the first voice also gets the character-read kinds,
    faults and fixtures."""
    items = []
    for k, voice in enumerate(voices):
        items += _voice_items(words, voice, seed + k, singles, strings, primary=k == 0,
                              tag='' if k == 0 else voice.split('-')[-1].removesuffix('Neural') + '-')
    clips = config.cache('bench', 'clips')

    def synth(item):
        body = item.get('body') or tts.phonemes(lang.syllables(item['spoken']), item['text'])
        path = clips / f"{item['id']}.mp3"
        try:
            path.write_bytes(tts.say(body, item['voice']))
            item['path'] = str(path)
        except tts.Rejected as err:  # a syllable the voice has no phones for
            sys.stderr.write(f"\nskipped {item['id']} {item['spoken']}: {err}\n")

    with ThreadPoolExecutor(6) as pool_:
        for k, _ in enumerate(pool_.map(synth, items), 1):
            if k % 20 == 0 or k == len(items):
                progress(k, len(items), 'synthesizing')

    rng = random.Random(seed)
    items = [i for i in items if 'path' in i]
    items += _faults([i for i in items if i['kind'] == 'good' and i['split'] == 'test' and i['voice'] == voices[0]], clips, rng)
    items += _fixtures(clips)
    manifest_path().write_text(json.dumps(items, ensure_ascii=False, indent=1))
    counts = defaultdict(int)
    for i in items:
        counts[(i['voice'], i['split'], i['kind'])] += 1
    for (voice, split, kind), n in sorted(counts.items()):
        print(f'{voice:28} {split:5} {kind:11} {n}')


def _voice_items(words, voice, seed, singles, strings, primary, tag):
    rng = random.Random(seed)
    items, seen = [], set()
    for w in words:
        sylls = lang.syllables(w['jyutping'])
        if w['jyutping'] in seen or not all(lang.sayable(s, voice) for s in sylls):
            continue
        seen.add(w['jyutping'])
        text = han(w['text'])
        text = text if len(text) == len(sylls) else _chars_for(sylls)
        split = split_of(text)
        base = {'split': split, 'text': text, 'jyutping': w['jyutping'], 'voice': voice, 'source': w.get('id')}
        items.append({**base, 'id': f'{tag}good-{len(items)}', 'kind': 'good', 'spoken': w['jyutping']})
        wrong = _wrong_tone(sylls, voice, rng)
        if wrong:
            i, s = wrong
            spoken = sylls[:i] + [s] + sylls[i + 1:]
            items.append({**base, 'id': f'{tag}wrong-{len(items)}', 'kind': 'wrong-tone', 'spoken': ' '.join(spoken),
                          'changed': i})
        if primary and w.get('plain', True) and len(han(w['text'])) == len(sylls):
            items.append({**base, 'id': f'hanzi-good-{len(items)}', 'kind': 'hanzi-good', 'spoken': w['jyutping'],
                          'body': escape(text)})
            swaps = [(i, lang.base(s) + str(t)) for i, s in enumerate(sylls) for t in lang.tones_for(s)
                     if t != lang.tone(s) and lang.sayable(lang.base(s) + str(t), voice)
                     and lang.char_read_as(lang.base(s) + str(t))]
            if swaps:
                i, s = rng.choice(swaps)
                spoken = sylls[:i] + [s] + sylls[i + 1:]
                items.append({**base, 'id': f'hanzi-wrong-{len(items)}', 'kind': 'hanzi-wrong', 'spoken': ' '.join(spoken),
                              'changed': i, 'body': escape(text[:i] + lang.char_read_as(s) + text[i + 1:])})

    # Training material beyond the words: attested syllables alone, in
    # their tones, and random strings, weighted by how common they are.
    freq = lang.syllable_frequency()
    attested = [s for s in lang.chars_by_syllable() if lang.sayable(s, voice) and freq[s] >= 3]
    by_base = defaultdict(list)
    for s in attested:
        by_base[lang.base(s)].append(s)
    bases = sorted(by_base, key=lambda b: -sum(freq[s] for s in by_base[b]))
    pool = [s for b in bases for s in by_base[b]][:singles]
    for s in pool:
        items.append({'id': f'{tag}single-{len(items)}', 'kind': 'single', 'split': 'train', 'text': _chars_for([s]),
                      'jyutping': s, 'spoken': s, 'voice': voice})
    weights = np.array([freq[s] for s in attested], dtype=float) ** 0.5
    weights /= weights.sum()
    nrng = np.random.default_rng(seed)
    for _ in range(strings):
        sylls = list(nrng.choice(attested, size=int(nrng.integers(2, 6)), p=weights))
        items.append({'id': f'{tag}string-{len(items)}', 'kind': 'string', 'split': 'train', 'text': _chars_for(sylls),
                      'jyutping': ' '.join(sylls), 'spoken': ' '.join(sylls), 'voice': voice})
    return items


def _faults(goods, clips, rng):
    """Silent, truncated and tripled copies of some good test clips."""
    out = []
    ff = lambda *a: subprocess.run(['ffmpeg', '-v', 'error', '-y', *a], check=True)
    for g in rng.sample(goods, min(40, len(goods))):
        for kind, args in [('silent', ['-af', 'volume=0.01']), ('truncated', ['-t', '0.12']),
                           ('tripled', ['-filter_complex', '[0][0][0]concat=n=3:v=0:a=1'])]:
            path = clips / f"{g['id']}-{kind}.mp3"
            ff('-i', g['path'], *args, str(path))
            out.append({**g, 'id': f"{g['id']}-{kind}", 'kind': 'fault', 'fault': kind, 'path': str(path)})
    return out


def _fixtures(clips):
    """bench/fixtures.json: real cases. Each is { id, jyutping, text, expect:
    'pass' | 'flag', and either file (in bench/fixtures/) or ssml (spoken
    by `voice`) }."""
    if not FIXTURES.exists():
        return []
    out = []
    for f in json.loads(FIXTURES.read_text()):
        if 'file' in f:
            path = FIXTURES.parent / 'fixtures' / f['file']
        else:
            path = clips / f"fixture-{f['id']}.mp3"
            path.write_bytes(tts.say(f['ssml'], f['voice']))
        out.append({**f, 'kind': 'fixture', 'split': 'test', 'path': str(path)})
    return out


# ---------------------------------------------------------------- train

def _rows(items, analyser, refs):
    from .check import syllable_rows
    X, y, groups, info = [], [], [], []
    for k, it in enumerate(items, 1):
        sylls = lang.syllables(it['spoken'])
        a = analyser.analyse(it['path'], sylls)
        rows = syllable_rows(a, sylls, refs.get(it['voice']))
        if rows is not None:
            X.append(rows)
            y += [lang.tone(s) for s in sylls]
            groups += [k] * len(sylls)
            info += [(it, i) for i in range(len(sylls))]
        if k % 50 == 0 or k == len(items):
            progress(k, len(items), 'analysing')
    return np.vstack(X), np.array(y), np.array(groups), info


def _voice_refs(items, analyser):
    """Each voice's reference pitch (Hz): the median of its clips' medians."""
    meds = defaultdict(list)
    for it in items:
        a = analyser.analyse(it['path'], lang.syllables(it['spoken']))
        voiced = a['per'] > 0.4
        if voiced.sum() > 4:
            meds[it['voice']].append(float(np.median(a['f0'][voiced])))
    return {v: float(np.median(m)) for v, m in meds.items()}


SAPI_KINDS = ('good', 'wrong-tone', 'single', 'string')   # exact labels: the model learns from these
HANZI_KINDS = ('hanzi-good', 'hanzi-wrong')             # realistic: thresholds are set on these
CHECK_FP, LISTEN_FP = 0.05, 0.15                        # false-flag budgets on right clips


def _clip_scores(p_want, info):
    """Lowest probability of the expected tone in each clip, from
    per-syllable probabilities."""
    low = {}
    for p, (it, _) in zip(p_want, info):
        low[it['id']] = min(low.get(it['id'], 1.0), p)
    return low


def _pick_thresholds(low, items_by_id, good_kind, check_fp=CHECK_FP, listen_fp=LISTEN_FP):
    """The highest thresholds whose false flags on right clips stay within
    check_fp (CHECK) and listen_fp (LISTEN)."""
    good = np.sort([v for k, v in low.items() if items_by_id[k]['kind'] == good_kind])
    q = lambda share: float(good[int(share * len(good))]) if len(good) else 0.0
    return {'check': q(check_fp), 'listen': max(q(listen_fp), q(check_fp))}


def train(folds: int = 5):
    """Fit tone models on the train split's sapi clips (exact labels): an abs
    model per voice, and one shape model on every voice. Thresholds are set
    on the train split's character-read clips (the main voice's), so false
    flags stay within budget on clips made the way a site makes them. Other
    voices have no character-read clips, so theirs are set on their sapi
    clips (cross-validated), which are cleaner: expect more false flags on
    their real clips than the budget."""
    from sklearn.model_selection import GroupKFold

    from .check import Analyser
    from .tones import ABS_COLS, Model, make_classifier, probs
    split = [i for i in load_manifest() if i['split'] == 'train']
    items = [i for i in split if i['kind'] in SAPI_KINDS]
    hanzi = [i for i in split if i['kind'] in HANZI_KINDS]
    analyser = Analyser()
    refs = _voice_refs(items, analyser)
    X, y, groups, info = _rows(items, analyser, refs)
    Xh, _, _, info_h = _rows(hanzi, analyser, refs)
    print(f'{len(items)} sapi clips ({len(y)} syllables) to learn from, {len(hanzi)} character-read clips to set thresholds; voices: {refs}')
    voice = np.array([it['voice'] for it, _ in info])
    # The tone each syllable should have (for a wrong clip, the one it was
    # asked for: that's what the checker is asked about).
    want = lambda inf: np.array([lang.tone(lang.syllables(it['jyutping'])[i]) for it, i in inf])
    want_x, want_h = want(info), want(info_h)
    by_id = {i['id']: i for i in split}

    def oof_probs(Xm, ym, gm):
        out = np.zeros((len(ym), 6))
        for tr, te in GroupKFold(folds).split(Xm, ym, gm):
            out[te] = probs(make_classifier().fit(Xm[tr], ym[tr]), Xm[te])
        return out

    def report(name, acc, th, low, wrong_kind):
        wrong = [v for k, v in low.items() if by_id[k]['kind'] == wrong_kind]
        print(f'{name}: cross-validated syllable accuracy {acc:.3f}; thresholds {th}; train {wrong_kind} caught: '
              f"CHECK {np.mean([v < th['check'] for v in wrong]):.1%}, any {np.mean([v < th['listen'] for v in wrong]):.1%}")

    abs_clfs, thresholds = {}, {}
    for v in refs:
        m = voice == v
        oof = oof_probs(X[m], y[m], groups[m])
        abs_clfs[v] = make_classifier().fit(X[m], y[m])
        if v == VOICE:
            low = _clip_scores(probs(abs_clfs[v], Xh)[np.arange(len(want_h)), want_h - 1], info_h)
            thresholds[v] = _pick_thresholds(low, by_id, 'hanzi-good')
            report(v, float((oof.argmax(1) + 1 == y[m]).mean()), thresholds[v], low, 'hanzi-wrong')
        else:
            inf = [x for x, keep in zip(info, m) if keep]
            low = _clip_scores(oof[np.arange(m.sum()), want_x[m] - 1], inf)
            low = {k: p for k, p in low.items() if by_id[k]['kind'] in ('good', 'wrong-tone')}
            thresholds[v] = _pick_thresholds(low, by_id, 'good')
            report(v, float((oof.argmax(1) + 1 == y[m]).mean()), thresholds[v], low, 'wrong-tone')

    Xs = np.delete(X, ABS_COLS, axis=1)
    oof = oof_probs(Xs, y, groups)
    shape_clf = make_classifier().fit(Xs, y)
    low = _clip_scores(probs(shape_clf, np.delete(Xh, ABS_COLS, axis=1))[np.arange(len(want_h)), want_h - 1], info_h)
    thresholds['shape'] = _pick_thresholds(low, by_id, 'hanzi-good')
    report('shape (any voice)', float((oof.argmax(1) + 1 == y).mean()), thresholds['shape'], low, 'hanzi-wrong')
    Model(abs_clfs, shape_clf, refs, thresholds).save()


# ---------------------------------------------------------------- score

def _is_tone(msg: str) -> bool:
    return msg.startswith('tone')


def score(results: list[dict] | None = None, name: str = 'new', fail: bool = True) -> int:
    items = [i for i in load_manifest() if i['split'] == 'test']
    if results is None:
        from .check import Checker
        checker = Checker()
        results = []
        for k, it in enumerate(items, 1):
            results.append(checker.check(it))
            if k % 25 == 0 or k == len(items):
                progress(k, len(items), 'checking')
    (bench_dir() / f'results-{name}.json').write_text(json.dumps(results, ensure_ascii=False))
    res = {r['id']: r for r in results}

    def flags(it, tone_only=False):
        r = res.get(it['id'])
        if r is None:
            return None
        pick = (lambda m: _is_tone(m)) if tone_only else (lambda m: True)
        return (any(map(pick, r['problems'])), any(map(pick, r['problems'] + r['notes'])))

    def rate(group, which, tone_only=False):
        vals = [flags(i, tone_only) for i in group]
        vals = [v[which] for v in vals if v is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (float('nan'), 0)

    size = lambda it: len(lang.syllables(it['jyutping']))
    bucket = lambda it: '1' if size(it) == 1 else '2-3' if size(it) <= 3 else '4+'
    kinds, by_voice = defaultdict(list), defaultdict(list)
    for it in items:
        kinds[it['kind']].append(it)
        by_voice[(it['kind'], it['voice'])].append(it)
    # The character-read set is in the main voice; the sapi set in every
    # voice the model learns ("sapi" is the main voice, "sapi-WanLung"...).
    voices = sorted({i['voice'] for i in kinds['good']}, key=lambda v: v != VOICE)
    short = lambda v: v.split('-')[-1].removesuffix('Neural')
    sets = [('hanzi', 'hanzi-good', 'hanzi-wrong', VOICE)] + [
        ('sapi' if v == VOICE else f'sapi-{short(v)}', 'good', 'wrong-tone', v) for v in voices]

    out = {'name': name}
    print(f'\n== {name}: {len(res)} of {len(items)} test clips checked')
    print(f"{'':28}{'CHECK':>8}{'any flag':>10}{'n':>6}")
    line = lambda label, a, b: print(f'{label:28}{a[0]:8.1%}{b[0]:10.1%}{a[1]:6}')
    # Keys: <set>_<tone|all>_<good|wrong_<size>>_<check|flag>, e.g.
    # hanzi_tone_wrong_2-3_check. "tone" counts only tone messages.
    for tone_only in (True, False):
        tag = 'tone' if tone_only else 'all'
        for label, good_kind, wrong_kind, voice in sets:
            print(f'-- {label} ({"character-read, realistic" if label == "hanzi" else "sapi-read, exact labels"}), {tag} messages')
            g = by_voice[(good_kind, voice)]
            key = f'{label}_{tag}'
            out[f'{key}_good_check'], out[f'{key}_good_flag'] = rate(g, 0, tone_only)[0], rate(g, 1, tone_only)[0]
            line('  right clips (false flags)', rate(g, 0, tone_only), rate(g, 1, tone_only))
            for b in ('1', '2-3', '4+'):
                w = [i for i in by_voice[(wrong_kind, voice)] if bucket(i) == b]
                c, a = rate(w, 0, tone_only), rate(w, 1, tone_only)
                out[f'{key}_wrong_{b}_check'], out[f'{key}_wrong_{b}_flag'] = c[0], a[0]
                line(f'  wrong tone, {b} syll', c, a)
    f = kinds['fault']
    out['fault_check'] = rate(f, 0)[0]
    line('faults (file)', rate(f, 0), rate(f, 1))

    # Fixtures are about tones: speech-to-text notes don't count.
    fx_ok = []
    for it in kinds['fixture']:
        fl = flags(it, tone_only=True)
        if fl is None:
            continue
        got = 'flag' if fl[1] else 'pass'
        fx_ok.append(got == it['expect'])
        r = res[it['id']]
        print(f"  fixture {it['id']:22} expect {it['expect']:4} got {got:4} {'ok' if got == it['expect'] else 'WRONG'}  "
              f"{'; '.join(r['problems'] + r['notes'])[:110]}")
    out['fixtures_ok'] = sum(fx_ok) / len(fx_ok) if fx_ok else float('nan')

    # Tone confusion on good clips (only for results with tone details).
    conf = np.zeros((6, 6), dtype=int)
    for it in by_voice[('good', VOICE)]:
        for t in res.get(it['id'], {}).get('tones', []) or []:
            if isinstance(t, dict) and 'heard' in t and t['heard'] in range(1, 7):
                conf[t['want'] - 1, t['heard'] - 1] += 1
    if conf.sum():
        out['syllable_accuracy'] = float(np.trace(conf) / conf.sum())
        print(f"\nsyllable tone accuracy on sapi-read right clips: {out['syllable_accuracy']:.1%}  (rows: said, columns: heard)")
        print('     ' + ''.join(f'{t:>6}' for t in range(1, 7)))
        for t in range(6):
            print(f'  {t + 1}  ' + ''.join(f'{v:6}' for v in conf[t]))

    (bench_dir() / f'score-{name}.json').write_text(json.dumps(out, indent=1))
    if not fail or not THRESHOLDS.exists():
        return 0
    limits = json.loads(THRESHOLDS.read_text())
    bad = [f'{k} = {out.get(k, float("nan")):.3f}, needs {"≤" if k.endswith(("good_check", "good_flag")) else "≥"} {v}'
           for k, v in limits.items()
           if not (out.get(k, float('nan')) <= v if k.endswith(('good_check', 'good_flag')) else out.get(k, float('nan')) >= v)]
    for b in bad:
        print('BELOW THRESHOLD:', b)
    return 1 if bad else 0
