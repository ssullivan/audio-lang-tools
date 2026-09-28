"""Pitch tracking, with a choice of tracker (ALTOOLS_PITCH):

  crepe-tiny  CREPE tiny (torchcrepe): 0.1 s a clip on CPU
  crepe-full  CREPE full: more accurate, 2.5 s a clip on CPU
  rmvpe       RMVPE (rmvpe.py): robust on breathy and creaky voice. The
              default: on the benchmark it beat CREPE tiny everywhere (tone
              accuracy 94% vs 91%; character-read wrong tones caught as
              CHECK 97/84/84% vs 92/75/65% by length, with fewer false
              flags) at 0.1–0.2 s a clip

Each gives a pitch and a voicing score per 10 ms frame; VOICED is the
score above which a frame counts as voiced, and LOOSE the looser one
tones.py falls back to for short creaky syllables. The analysis cache is
per tracker, so switching is cheap after the first run. The benchmark
(`altools bench score`) decides which is best.

    track(x) → (times s, f0 Hz, voicing 0–1), 10 ms frames
    semitones(hz, ref_hz) → semitones above ref_hz
"""
import os

import numpy as np
import torch
import torchcrepe

from .audio import RATE

HOP = RATE // 100
BACKEND = os.environ.get('ALTOOLS_PITCH', 'rmvpe')
# (VOICED, LOOSE) per tracker. CREPE's periodicity on noise reaches ~0.35
# on a few frames.
THRESHOLDS = {'crepe-tiny': (0.4, 0.2), 'crepe-full': (0.4, 0.2), 'rmvpe': (0.3, 0.1)}
VOICED, LOOSE = THRESHOLDS[BACKEND]

_rmvpe = None


def track(x: np.ndarray, model: str = BACKEND):
    if model == 'rmvpe':
        global _rmvpe
        if _rmvpe is None:
            from .rmvpe import RMVPE
            _rmvpe = RMVPE()
        f0, voicing = _rmvpe.track(x)
    else:
        size = model.removeprefix('crepe-')
        with torch.inference_mode():
            f0, per = torchcrepe.predict(torch.from_numpy(x).unsqueeze(0), RATE, HOP, 50, 600, size,
                                         return_periodicity=True, batch_size=1024, device='cpu')
        f0, voicing = torchcrepe.filter.median(f0, 3)[0].numpy(), torchcrepe.filter.median(per, 3)[0].numpy()
    return np.arange(len(f0)) * HOP / RATE, f0, voicing


def semitones(hz, ref_hz: float):
    return 12 * np.log2(np.asarray(hz) / ref_hz)
