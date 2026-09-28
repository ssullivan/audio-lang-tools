"""Azure text-to-speech, for making benchmark and training clips.

say(ssml_body, voice) → mp3 bytes, cached by request. phonemes(sylls, text)
reads syllables exactly with sapi phones, which Azure honours, tones
included, except where the voice lacks a syllable (see lang.cantonese.GAPS).
"""
import hashlib
import time
from xml.sax.saxutils import escape

import requests

from . import config

FORMAT = 'audio-24khz-48kbitrate-mono-mp3'  # what learn-cantonese's clips use


class Rejected(RuntimeError):
    """Azure refused the request (400), e.g. a sapi syllable it has no phones for."""


def phonemes(sylls: list[str], text: str) -> str:
    """SSML reading `sylls` (jyutping with tone numbers) exactly; `text` is
    the characters shown inside the element."""
    ph = ' '.join(f'{s[:-1]} {s[-1]}' for s in sylls)
    return f'<phoneme alphabet="sapi" ph="{ph}">{escape(text)}</phoneme>'


def ssml(body: str, voice: str) -> str:
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-HK">'
            f'<voice name="{voice}">{body}</voice></speak>')


def say(body: str, voice: str) -> bytes:
    request = ssml(body, voice)
    path = config.cache('tts') / (hashlib.sha1(request.encode()).hexdigest() + '.mp3')
    if path.exists():
        return path.read_bytes()
    key, region = config.azure()
    for attempt in range(6):
        res = requests.post(f'https://{region}.tts.speech.microsoft.com/cognitiveservices/v1', data=request.encode(), headers={
            'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': FORMAT, 'User-Agent': 'audio-lang-tools'}, timeout=60)
        if res.ok and res.content:
            path.write_bytes(res.content)
            return res.content
        if res.status_code in (401, 429) or res.status_code >= 500:
            time.sleep(2 * 2 ** attempt)
            continue
        # 400, or 200 with no audio (a character it has no reading for).
        if res.status_code in (200, 400):
            raise Rejected(f'Azure TTS {res.status_code} {res.text[:200] or "(no audio)"}')
        raise RuntimeError(f'Azure TTS {res.status_code}: {res.text[:200]}')
    raise RuntimeError('Azure TTS: too many retries')
