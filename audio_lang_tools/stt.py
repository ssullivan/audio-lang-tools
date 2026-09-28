"""Word check with Azure speech-to-text (zh-HK short audio), given no hint
of the expected text. What it heard is compared with the expected
syllables ignoring tone, so a homophone (分 for 墳) isn't a mismatch;
tones are tones.py's job. Speech-to-text is weak on bare syllables and
biased towards common words, so a mismatch means "listen", not "wrong".
Results are cached by clip hash.
"""
import hashlib
import json
import time

import requests

from . import config
from .audio import wav


def recognize(path, offline: bool = False) -> dict | None:
    """Azure's detailed result; with offline, only a cached one (else None)."""
    data = open(path, 'rb').read()
    cached = config.cache('stt') / (hashlib.sha1(data).hexdigest() + '.json')
    if cached.exists():
        return json.loads(cached.read_text())
    if offline:
        return None
    key, region = config.azure()
    body = wav(path)
    try:
        config.spend('stt', (len(body) - 44) / 32000)  # 16 kHz 16-bit mono after a 44-byte header
    except config.BudgetExceeded as err:
        raise Unavailable(str(err)) from err
    url = (f'https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1'
           '?language=zh-HK&format=detailed')
    for attempt in range(6):
        try:
            res = requests.post(url, data=body, timeout=60, headers={
                'Ocp-Apim-Subscription-Key': key, 'Accept': 'application/json',
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000'})
        except requests.ConnectionError:
            time.sleep(2 * 2 ** attempt)
            continue
        if res.ok:
            cached.write_text(res.text)
            return res.json()
        if 'quota' in res.text.lower() or res.status_code == 403:
            raise Unavailable(f'Azure STT {res.status_code}: {res.text[:120]}')
        # Azure now and then answers 401 to a burst of requests; retry that too.
        if res.status_code in (401, 429) or res.status_code >= 500:
            time.sleep(2 * 2 ** attempt)
            continue
        raise RuntimeError(f'Azure STT {res.status_code}: {res.text[:200]}')
    raise Unavailable('Azure STT: too many retries')


class Unavailable(RuntimeError):
    """Speech-to-text can't be used now (quota used up, no access)."""


def heard(path, offline: bool = False) -> str | None:
    """What speech-to-text heard (no spaces), or None if offline and not cached."""
    r = recognize(path, offline)
    if r is None:
        return None
    best = (r.get('NBest') or [{}])[0]
    return (best.get('Lexical') or '').replace(' ', '')
