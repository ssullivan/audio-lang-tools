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
