"""Decoding and file checks.

decode(path) → mono 16 kHz float32 samples (ffmpeg). basics(x, syllables)
→ (seconds of speech, [problems]): broken, silent, clipped, or too short
or long for its syllable count.
"""
import subprocess

import numpy as np

RATE = 16000


def decode(path) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(RATE), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def wav(path) -> bytes:
    return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(RATE), '-f', 'wav', '-'],
                          capture_output=True, check=True).stdout


def speech_span(x: np.ndarray) -> tuple[float, float]:
    """Start and end (s) of speech: the first and last 10 ms window above 5%
    of the peak (Azure pads clips with silence)."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    win = RATE // 100
    n = len(x) // win
    if not n or not peak:
        return 0.0, 0.0
    loud = np.flatnonzero(np.abs(x[:n * win]).reshape(n, win).max(axis=1) > peak * 0.05)
    return (loud[0] / 100, (loud[-1] + 1) / 100) if len(loud) else (0.0, 0.0)


def basics(x: np.ndarray, syllables: int) -> tuple[float, list[str]]:
    problems = []
    peak = float(np.abs(x).max()) if len(x) else 0.0
    clipped = int((np.abs(x) > 0.999).sum())
    a, b = speech_span(x)
    secs = b - a
    if peak < 0.05:
        problems.append('nearly silent')
    if clipped > 20:
        problems.append(f'{clipped} clipped samples')
    per = secs / syllables
    if per < 0.12:
        problems.append(f'too short: {secs:.2f} s for {syllables} syllables')
    if per > 0.6:
        problems.append(f'too long: {secs:.2f} s for {syllables} syllables')
    return secs, problems
