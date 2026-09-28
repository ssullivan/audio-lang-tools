"""Tones: pitch features per syllable, and a classifier that gives the
probability of each tone.

Features for syllable i (see features()):
  - its pitch over its rhyme, from where the aligner puts its rhyme to the
    next syllable, on the longest voiced stretch, trimmed 10% at each end,
    resampled to N points:
      abs  semitones above the voice's reference pitch (a calibrated voice:
           one the model was trained on); tells level tones 1, 3, 6 apart
      rel  semitones above the clip's median pitch; shape only
  - slope, range, voiced and syllable duration, checked (p/t/k), position
  - the neighbours' rel contours (coarticulation) and the clip's own level

Models (bench.py trains them): an "abs" model per voice it has training
clips for, and a "shape" model, without the abs features, for any other
voice (which then can't tell level tones apart by height, so its doubts
are only worth a listen).

    Model.load() / model.save(); model.probs(feature rows, voice) → [n, 6]
"""
import pickle

import numpy as np

from . import config
from . import pitch
from .pitch import LOOSE, VOICED, semitones

N = 10            # points per contour
NB = 5            # points per neighbour contour
MIN_FRAMES = 4    # voiced 10 ms frames needed to measure a syllable
# A short syllable ending in p, t or k often creaks, low ones most (十
# sap6): CREPE's periodicity drops and the rhyme has too few voiced frames.
# Left unmeasured, the model learned "unmeasured checked syllable → tone
# 6". So a syllable too short to measure on its rhyme is measured on the
# whole syllable, with a looser voicing threshold, from 2 frames.
LOOSE_MIN = 2


def _stretch(t, f0, per, a, b, voiced=VOICED, min_frames=MIN_FRAMES):
    """Frames of the longest voiced stretch in [a, b) (gaps up to 2 frames
    bridged), or None."""
    idx = np.flatnonzero((t >= a) & (t < b) & (per > voiced))
    if len(idx) < min_frames:
        return None
    runs, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= 3:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < min_frames:
        return None
    cut = int(len(run) * 0.1)
    run = run[cut:len(run) - cut] if len(run) - 2 * cut >= min_frames else run
    return np.arange(run[0], run[-1] + 1)


def _resample(v, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


def measure(t, f0, per, spans, ref_hz=None):
    """Per syllable: {st (semitones over ref_hz or the clip median), dur,
    voiced}; st is None where there is too little voicing. Also the clip's
    median pitch (Hz)."""
    a, b = spans[0]['start'], spans[-1]['end']
    sel = (t >= a) & (t < b) & (per > VOICED)
    med = float(np.median(f0[sel])) if sel.sum() >= MIN_FRAMES else float(np.median(f0[per > VOICED])) if (per > VOICED).any() else 200.0
    out = []
    for sp in spans:
        frames = _stretch(t, f0, per, sp['rhyme'], sp['end'])
        # The loose fallback needs one clearly voiced frame: noise alone
        # reaches CREPE periodicity ~0.35 on a few frames.
        in_syll = (t >= sp['start']) & (t < sp['end'])
        if frames is None and (per[in_syll] > VOICED).any():
            frames = _stretch(t, f0, per, sp['start'], sp['end'], LOOSE, LOOSE_MIN)
        if frames is None:
            out.append({'rel': None, 'abs': None, 'dur': sp['end'] - sp['start'], 'voiced': 0.0})
            continue
        hz = f0[frames]
        out.append({
            'rel': semitones(hz, med),
            'abs': semitones(hz, ref_hz) if ref_hz else None,
            'dur': sp['end'] - sp['start'],
            'voiced': len(frames) / 100,
        })
    return out, med


def features(sylls, checked, measured, clip_med_hz, ref_hz=None):
    """One feature row per syllable (NaN where unmeasured)."""
    nan = lambda n: np.full(n, np.nan)
    level = semitones(clip_med_hz, ref_hz) if ref_hz else np.nan
    rows = []
    n = len(sylls)
    for i, m in enumerate(measured):
        rel = _resample(m['rel'], N) if m['rel'] is not None else nan(N)
        ab = _resample(m['abs'], N) if m['abs'] is not None else nan(N)
        if m['rel'] is not None and len(m['rel']) > 1:
            x = np.arange(len(m['rel'])) / 100
            slope = np.polyfit(x, m['rel'], 1)[0]
            rng = float(np.ptp(m['rel']))
        else:
            slope = rng = np.nan
        nb = []
        for j in (i - 1, i + 1):
            if 0 <= j < n and measured[j]['rel'] is not None:
                nb += list(_resample(measured[j]['rel'], NB))
            else:
                nb += list(nan(NB))
        rows.append(np.concatenate([ab, rel, [slope, rng, m['voiced'], m['dur'], float(checked[i]),
                                               n, i / max(1, n - 1), float(i == n - 1), level], nb]))
    return np.array(rows)


# Columns that depend on the voice's calibrated pitch: dropped by the
# shape model.
ABS_COLS = list(range(N)) + [2 * N + 8]


class Model:
    """An abs classifier per trained voice (one model for all voices judged
    HiuMaan's clips worse than its own model did), one shape classifier for
    any voice, each voice's reference pitch, and decision thresholds per
    model (probability of the expected tone below `check` → CHECK, below
    `listen` → LISTEN), keyed by voice or 'shape'."""

    def __init__(self, abs_clfs, shape_clf, refs, thresholds):
        self.abs_clfs, self.shape_clf, self.refs, self.thresholds = abs_clfs, shape_clf, refs, thresholds

    def knows(self, voice):
        return voice in self.abs_clfs

    @staticmethod
    def path():
        """One model per pitch tracker: its features depend on the tracker."""
        return config.cache('models') / f'tones-{pitch.BACKEND}.pkl'

    @classmethod
    def load(cls, path=None):
        path = path or cls.path()
        if not path.exists():
            raise SystemExit(f'No tone model at {path}: run `altools bench train` first.')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path=None):
        path = path or self.path()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def probs(self, rows, voice):
        """[n, 6]: probability of tones 1–6 for each row, with the voice's own
        model if it has one, else the shape model."""
        if self.knows(voice):
            return probs(self.abs_clfs[voice], rows)
        return probs(self.shape_clf, np.delete(rows, ABS_COLS, axis=1))


def probs(clf, x):
    """[n, 6] tone probabilities from a classifier (classes it never saw: 0)."""
    p = clf.predict_proba(x)
    full = np.zeros((len(x), 6))
    for k, c in enumerate(clf.classes_):
        full[:, int(c) - 1] = p[:, k]
    return full


def make_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True, random_state=0)
