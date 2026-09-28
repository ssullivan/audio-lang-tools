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


def recognize(path) -> dict:
    data = open(path, 'rb').read()
    cached = config.cache('stt') / (hashlib.sha1(data).hexdigest() + '.json')
    if cached.exists():
        return json.loads(cached.read_text())
    key, region = config.azure()
    url = (f'https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1'
           '?language=zh-HK&format=detailed')
    for attempt in range(6):
        try:
            res = requests.post(url, data=wav(path), timeout=60, headers={
                'Ocp-Apim-Subscription-Key': key, 'Accept': 'application/json',
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000'})
        except requests.ConnectionError:
            time.sleep(2 * 2 ** attempt)
            continue
        if res.ok:
            cached.write_text(res.text)
            return res.json()
        # Azure now and then answers 401 to a burst of requests; retry that too.
        if res.status_code in (401, 429) or res.status_code >= 500:
            time.sleep(2 * 2 ** attempt)
            continue
        raise RuntimeError(f'Azure STT {res.status_code}: {res.text[:200]}')
    raise RuntimeError('Azure STT: too many retries')


def heard(path) -> str:
    r = recognize(path)
    best = (r.get('NBest') or [{}])[0]
    return (best.get('Lexical') or '').replace(' ', '')
