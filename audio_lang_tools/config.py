"""Paths and credentials.

Everything generated (models, Azure results, benchmark clips) lives in the
cache dir, never in the repo. Keys come from the environment, or from a
config file of KEY=value lines: $ALTOOLS_CONFIG, else
~/.config/audio-lang-tools/config.env. Never commit or print a key.

    AZURE_SPEECH_KEY, AZURE_SPEECH_REGION   Azure Speech (STT, and TTS for the benchmark)
"""
import os
import re
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
