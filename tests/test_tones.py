import numpy as np
import pytest

from audio_lang_tools import config, pitch, tones

RATE = 16000


def voice(f_start, f_end, dur=0.4):
    """A voice-like tone gliding from f_start to f_end Hz."""
    n = int(dur * RATE)
    f = np.linspace(f_start, f_end, n)
    ph = 2 * np.pi * np.cumsum(f) / RATE
    x = sum(np.sin(h * ph) / h for h in range(1, 8))
    env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / 400)
    return (0.3 * env * x).astype(np.float32)


def contour_of(x, ref_hz=200.0):
    t, f0, per = pitch.track(x, model=pitch.BACKEND)
    spans = [{'start': 0.0, 'rhyme': 0.0, 'end': len(x) / RATE}]
    measured, med = tones.measure(t, f0, per, spans, ref_hz)
    return measured[0], med


@pytest.mark.skipif(not (config.CACHE / 'models/rmvpe.pt').exists(), reason='RMVPE weights not downloaded')
def test_rmvpe_follows_a_glide_and_ignores_noise():
    t, f0, v = pitch.track(voice(180, 260), model='rmvpe')
    voiced = v > pitch.THRESHOLDS['rmvpe'][0]
    assert voiced.mean() > 0.8
    hz = f0[voiced]
    assert abs(hz[2] / 180 - 1) < 0.05 and abs(hz[-3] / 260 - 1) < 0.05
    noise = (1e-4 * np.random.default_rng(3).standard_normal(RATE // 2)).astype(np.float32)
    assert not (pitch.track(noise, model='rmvpe')[2] > pitch.THRESHOLDS['rmvpe'][1]).any()


def test_rising_and_level_contours():
    rise, _ = contour_of(voice(180, 260))
    level, _ = contour_of(voice(220, 220))
    assert rise['rel'][-1] - rise['rel'][0] > 3          # about 6 semitones
    assert abs(level['rel'][-1] - level['rel'][0]) < 1
    assert abs(np.median(level['abs']) - 12 * np.log2(220 / 200)) < 0.5


def test_features_shape_and_neighbours():
    a, _ = contour_of(voice(180, 260))
    b, med = contour_of(voice(240, 170))
    rows = tones.features(['si2', 'si4'], [False, False], [a, b], med, ref_hz=200.0)
    assert rows.shape == (2, 2 * tones.N + 9 + 2 * tones.NB)
    slope = 2 * tones.N
    assert rows[0, slope] > 0 and rows[1, slope] < 0
    # The first syllable has no left neighbour; its right one is the fall.
    left = rows[0, 2 * tones.N + 9:2 * tones.N + 9 + tones.NB]
    right = rows[0, 2 * tones.N + 9 + tones.NB:]
    assert np.isnan(left).all() and right[0] > right[-1]


def test_unvoiced_syllable_is_unmeasured():
    # Seeds whose noise CREPE scores 0.2–0.35 on a few frames, which its
    # loose fallback alone would take for voicing. Runs with the default
    # tracker (ALTOOLS_PITCH): run the tests with each.
    for seed in (2, 3, 4):
        noise = (1e-4 * np.random.default_rng(seed).standard_normal(RATE // 2)).astype(np.float32)
        t, f0, per = pitch.track(noise, model=pitch.BACKEND)
        measured, _ = tones.measure(t, f0, per, [{'start': 0, 'rhyme': 0, 'end': 0.5}])
        assert measured[0]['rel'] is None


def test_short_creaky_syllable_is_measured_on_the_whole_syllable():
    # 十 sap6: a clear syllable, then one whose rhyme has a single clearly
    # voiced frame but three weakly voiced ones across the syllable.
    t = np.arange(40) / 100
    f0 = np.full(40, 300.0)
    f0[20:] = 180.0
    per = np.zeros(40)
    per[2:18] = 0.9
    per[[22, 23, 24]] = [0.3, 0.3, 0.6]
    spans = [{'start': 0.0, 'rhyme': 0.02, 'end': 0.2}, {'start': 0.2, 'rhyme': 0.24, 'end': 0.4}]
    measured, _ = tones.measure(t, f0, per, spans, ref_hz=200.0)
    assert measured[1]['abs'] is not None
    assert np.allclose(measured[1]['abs'], 12 * np.log2(180 / 200))
