"""Word check with Azure speech-to-text (zh-HK), given no hint of the
expected text. What it heard is compared with the expected syllables
ignoring tone, so a homophone (分 for 墳) isn't a mismatch; tones are
tones.py's job. Speech-to-text is weak on bare syllables and biased
towards common words, so a mismatch means "listen", not "wrong".

Uses Azure's fast transcription API (api-version 2024-11-15). The older
short-audio endpoint kept answering "Quota exceeded" after the resource
moved from the free to the standard tier (2026-09-28), while fast
transcription worked with the same key. Results are cached by clip hash;
results cached from the short-audio endpoint (NBest/Lexical) still count.
"""
import hashlib
import json
import re
import time

import requests

from . import config
from .audio import wav

API = 'speechtotext/transcriptions:transcribe?api-version=2024-11-15'


def recognize(path, offline: bool = False) -> dict | None:
    """Azure's result; with offline, only a cached one (else None)."""
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
    for attempt in range(6):
        try:
            res = requests.post(f'https://{region}.api.cognitive.microsoft.com/{API}', timeout=60,
                                headers={'Ocp-Apim-Subscription-Key': key},
                                files={'audio': ('clip.wav', body, 'audio/wav'),
                                       'definition': (None, json.dumps({'locales': ['zh-HK']}))})
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


def text_of(result: dict) -> str:
    """What a result heard: characters and letters only (no spaces or
    punctuation), from either API's format."""
    if 'NBest' in result or 'RecognitionStatus' in result:
        text = ((result.get('NBest') or [{}])[0].get('Lexical') or '')
    else:
        text = ''.join(p.get('text', '') for p in result.get('combinedPhrases', []))
    return re.sub(r'[^\w]', '', text)


def heard(path, offline: bool = False) -> str | None:
    """What speech-to-text heard, or None if offline and not cached."""
    r = recognize(path, offline)
    return None if r is None else text_of(r)
