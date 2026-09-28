"""No file in the repo may hold a key (when keys are set on this machine)."""
import subprocess

from audio_lang_tools import config

NAMES = ['AZURE_SPEECH_KEY', 'MINIMAX_KEY', 'MINIMAX_API_KEY']


def test_no_key_in_repo():
    keys = [k for k in (config.secret(n, required=False) for n in NAMES) if k and len(k) > 8]
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=config.ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    for f in files:
        data = (config.ROOT / f).read_bytes()
        for k in keys:
            assert k.encode() not in data, f'{f} holds a key'
