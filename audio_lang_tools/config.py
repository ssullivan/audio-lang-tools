"""Paths and credentials.

Everything generated (models, Azure results, benchmark clips) lives in the
cache dir, never in the repo. Keys come from the environment, or from a
config file of KEY=value lines: $ALTOOLS_CONFIG, else
~/.config/audio-lang-tools/config.env. Never commit or print a key.

    AZURE_SPEECH_KEY, AZURE_SPEECH_REGION   Azure Speech (STT, and TTS for the benchmark)

Azure costs money (free credits run out), so each run may make at most
ALTOOLS_AZURE_BUDGET new, uncached requests (default 1000: enough to
check a whole site; rebuilding the benchmark needs about 12,000 and must
raise it on purpose). spend() counts them; usage() reports them.
"""
import atexit
import os
import re
import sys
from collections import Counter
from pathlib import Path

CACHE = Path(os.environ.get('ALTOOLS_CACHE', Path.home() / '.cache/audio-lang-tools'))
CONFIG = Path(os.environ.get('ALTOOLS_CONFIG', Path.home() / '.config/audio-lang-tools/config.env'))
ROOT = Path(__file__).resolve().parent.parent


def cache(*parts: str) -> Path:
    """A directory inside the cache, created on first use."""
    d = CACHE.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_config():
    if not CONFIG.exists():
        return
    for line in CONFIG.read_text().splitlines():
        m = re.match(r'\s*(?:export\s+)?([A-Z_]+)\s*=\s*"?([^"\n]*)"?\s*$', line)
        if m and m[1] not in os.environ:
            os.environ[m[1]] = m[2]


def secret(name: str, required: bool = True) -> str | None:
    _load_config()
    value = os.environ.get(name)
    if required and not value:
        raise SystemExit(f'Set {name} (environment or {CONFIG}).')
    return value


def azure() -> tuple[str, str]:
    return secret('AZURE_SPEECH_KEY'), secret('AZURE_SPEECH_REGION')


class BudgetExceeded(RuntimeError):
    """This run has made its ALTOOLS_AZURE_BUDGET of new Azure requests."""


_spent = Counter()
_audio_secs = [0.0]


def spend(kind: str, audio_secs: float = 0.0):
    """Count one new (uncached) Azure request of `kind` ('tts' or 'stt'),
    or raise BudgetExceeded if the run's budget is used up."""
    budget = int(os.environ.get('ALTOOLS_AZURE_BUDGET', '1000'))
    if sum(_spent.values()) >= budget:
        raise BudgetExceeded(f'{budget} new Azure requests in this run (ALTOOLS_AZURE_BUDGET); '
                             'raise it on purpose to go on')
    _spent[kind] += 1
    _audio_secs[0] += audio_secs


@atexit.register
def usage():
    if _spent:
        sys.stderr.write(f"Azure: {_spent['tts']} new TTS and {_spent['stt']} new speech-to-text requests "
                         f"({_audio_secs[0] / 60:.1f} min of audio) in this run; cached results cost nothing.\n")
