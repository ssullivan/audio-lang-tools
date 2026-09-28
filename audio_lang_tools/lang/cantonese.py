"""Cantonese (yue): jyutping syllables, the six tones, which syllable+tone
pairs exist, and which ones a TTS voice can't say.

Syllables are jyutping with a tone number: "neoi2". Tones 1–6 only (no
7/8/9): a syllable ending in p, t or k ("checked") takes tone 1, 3 or 6.
"""
from collections import Counter
from functools import lru_cache

import pycantonese

CODE = 'yue'
TONES = (1, 2, 3, 4, 5, 6)
CHECKED_TONES = (1, 3, 6)

# Syllable+tones a voice can't say with sapi phonemes: Azure reads them as
# another tone whatever the SSML says. Found by ear and by pitch
# (learn-cantonese: 今年 nin2 came out nin4; 女 neoi2 alone came out like
# neoi5). Benchmark clips never ask a voice for these.
GAPS = {'zh-HK-HiuMaanNeural': {'nin2', 'neoi2'}}


def syllables(jyutping: str) -> list[str]:
    return jyutping.split()


def tone(syll: str) -> int:
    return int(syll[-1])


def base(syll: str) -> str:
    return syll[:-1]


def parts(syll: str):
    """(onset, nucleus, coda) of one syllable."""
    p = pycantonese.parse_jyutping(syll)[0]
    return p.onset, p.nucleus, p.coda


def rhyme_at(syll: str) -> int:
    """Index of the first letter of the rhyme (nucleus + coda), which
    carries the tone. A syllabic m or ng (唔 m4, 五 ng5) is all rhyme."""
    return len(parts(syll)[0])


def checked(syll: str) -> bool:
    return parts(syll)[2] in ('p', 't', 'k')


def tones_for(syll: str) -> tuple[int, ...]:
    return CHECKED_TONES if checked(syll) else TONES


def to_jyutping(text: str) -> list[str | None]:
    """Jyutping per character (None where unknown), for what speech-to-text heard."""
    out = []
    for word, jp in pycantonese.characters_to_jyutping(text):
        sylls = jp.split() if jp else []
        chars = [c for c in word]
        out += sylls if len(sylls) == len(chars) else [None] * len(chars)
    return out


@lru_cache(maxsize=1)
def chars_by_syllable() -> dict[str, list[str]]:
    """Every syllable+tone that a character is read as, in rime-cantonese's
    words (女仔 neoi5 zai2, 仔女 zai2 neoi2) or in HKCanCor, with its
    characters, most frequent first. These are the attested syllables the
    benchmark may ask for."""
    from pycantonese.data.rime_cantonese import CHARS_TO_JYUTPING
    freq = Counter()
    for tok in pycantonese.hkcancor().tokens():
        if tok.jyutping and len(tok.word) == len(tok.jyutping.split()):
            freq.update(zip(tok.word, tok.jyutping.split()))
    pairs = set(freq)
    for word, jp in CHARS_TO_JYUTPING.items():
        sylls = jp.split()
        if len(word) == len(sylls):
            pairs.update(zip(word, sylls))
    out: dict[str, list[str]] = {}
    for char, syll in pairs:
        if syll[-1:] in set('123456') and syll[:-1].isalpha() and '\u3400' <= char <= '\u9fff':
            out.setdefault(syll, []).append(char)
    # Common characters (the main CJK block) before rare ones (Extension A).
    for syll, chars in out.items():
        chars.sort(key=lambda c: (-freq[(c, syll)], c < '\u4e00', c))
    return out


@lru_cache(maxsize=1)
def syllable_frequency() -> Counter:
    """How often each syllable+tone occurs in HKCanCor."""
    freq = Counter()
    for tok in pycantonese.hkcancor().tokens():
        if tok.jyutping:
            freq.update(tok.jyutping.split())
    return freq


# Speech-to-text writes numbers as digits ("4蚊半", "1.1個字" for 一點一個字),
# so a heard text is compared with every way its digits can be read.
DIGIT = ['ling', 'jat', 'ji', 'saam', 'sei', 'ng', 'luk', 'cat', 'baat', 'gau']


def _below_10000(n: int) -> list[list[str]]:
    """Toneless readings of 1–9999: 二 or 兩 before a place, 廿 / 卅 for
    20s / 30s, 十 alone for 10–19, and 零 filling gaps."""
    if n < 10:
        return [[DIGIT[n]]] + ([['loeng']] if n == 2 else [])
    out = []
    for place, name in ((1000, 'cin'), (100, 'baak'), (10, 'sap')):
        if n >= place:
            head, rest = divmod(n, place)
            heads = [[DIGIT[head]]] + ([['loeng']] if head == 2 and place > 10 else [])
            if place == 10 and head == 1:
                heads = [[], ['jat']]
            fronts = [h + [name] for h in heads]
            if place == 10 and head == 2 and rest:
                fronts.append(['jaa'])    # 廿一
            if place == 10 and head == 3 and rest:
                fronts.append(['saa'])    # 卅一
            if not rest:
                return fronts
            gap = rest < place // 10
            tails = _below_10000(rest)
            for f in fronts:
                for t in tails:
                    out.append(f + (['ling'] if gap else []) + t)
                    if not gap and place == 100 and rest % 10 == 0:
                        out.append(f + t[:1])   # 百五 for 150
            return out
    return out


def number_readings(token: str) -> list[list[str]]:
    """Toneless syllables for a run of digits as speech-to-text writes it:
    an integer, or two numbers joined by . or : (點: 1.1 → 一點一)."""
    for sep in '.:':
        if sep in token:
            a, b = token.split(sep, 1)
            return [x + ['dim'] + y for x in number_readings(a) for y in number_readings(b)] if a and b else []
    if not token.isdigit():
        return []
    n = int(token)
    if n == 0:
        return [['ling']]
    if n < 10000:
        return _below_10000(n)
    high, low = divmod(n, 10000)
    if high >= 10000:
        return [[DIGIT[int(d)] for d in token]]
    lows = [[]] if not low else [(['ling'] if low < 1000 else []) + r for r in _below_10000(low)]
    return [h + ['maan'] + l for h in _below_10000(high) for l in lows]


def heard_readings(text: str, limit: int = 64) -> list[list[str]]:
    """Every toneless syllable sequence a heard text can stand for (digits
    read every way they can be), at most `limit` of them."""
    import itertools
    import re
    parts = []
    for tok in re.findall(r'\d+(?:[.:]\d+)?|.', text):
        if tok[0].isdigit():
            parts.append(number_readings(tok) or [[tok]])
        else:
            jp = to_jyutping(tok)[0] if '㐀' <= tok <= '鿿' else None
            parts.append([[base(jp)]] if jp else [[tok.lower()]])
    return [sum(combo, []) for combo in itertools.islice(itertools.product(*parts), limit)]


def char_read_as(syll: str) -> str | None:
    """A common character whose usual reading is `syll`, so a voice reading
    it from text most likely says `syll` (琴 for kam4), or None."""
    from pycantonese.data.rime_cantonese import CHARS_TO_JYUTPING
    for c in chars_by_syllable().get(syll, [])[:8]:
        if CHARS_TO_JYUTPING.get(c) == syll and '一' <= c <= '鿿':
            return c
    return None


def sayable(syll: str, voice: str) -> bool:
    return syll in chars_by_syllable() and syll not in GAPS.get(voice, set())
