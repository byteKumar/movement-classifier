import numpy as np
import pytest

from movement.data import SAMPLE_RATE_HZ, WINDOW
from movement.features import ALL_CHANNELS, STATS, extract_features, feature_names
from movement.metrics import mean_std, score

T = np.arange(WINDOW) / SAMPLE_RATE_HZ


def _feat(F, channel, stat):
    return F[:, feature_names().index(f"{channel}_{stat}")]


def test_shape_and_names_agree():
    F = extract_features(np.random.default_rng(0).normal(size=(5, WINDOW, 6)).astype(np.float32))
    assert F.shape == (5, len(feature_names()))
    assert len(feature_names()) == len(ALL_CHANNELS) * len(STATS) + 3
    assert F.dtype == np.float32 and np.isfinite(F).all()


def test_sine_lands_in_the_right_band_and_frequency():
    X = np.zeros((1, WINDOW, 6))
    f = 10 * SAMPLE_RATE_HZ / WINDOW  # FFT bin 10 = 3.906 Hz; bin-aligned, so no spectral leakage
    X[0, :, 0] = 1.5 + np.sin(2 * np.pi * f * T)
    F = extract_features(X)
    assert _feat(F, "acc_x", "mean")[0] == pytest.approx(1.5, abs=1e-6)
    assert _feat(F, "acc_x", "dom_freq")[0] == pytest.approx(f, abs=1e-6)
    # all of the sine's power (variance 0.5) is in the 3-5 Hz band
    assert _feat(F, "acc_x", "band_3_5hz")[0] == pytest.approx(0.5, rel=1e-5)
    assert _feat(F, "acc_x", "band_1_3hz")[0] == pytest.approx(0.0, abs=1e-6)


def test_band_powers_add_up_to_variance_above_0_3_hz():
    x = np.random.default_rng(1).normal(size=(3, WINDOW, 6))
    F = extract_features(x)
    bands = sum(_feat(F, "gyro_y", s) for s in STATS if s.startswith("band_"))
    # Parseval: the lowest non-DC bin is 0.39 Hz (>= 0.3), so the bands cover every bin but DC
    assert np.allclose(bands, x[:, :, 4].var(axis=1), rtol=1e-4)


def test_constant_signal_gives_finite_features_and_zero_corr():
    X = np.ones((2, WINDOW, 6))
    F = extract_features(X)
    assert np.isfinite(F).all()
    assert (F[:, -3:] == 0).all()


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        extract_features(np.zeros((2, 64, 6)))


def test_score_and_mean_std():
    s = score(np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2, 3, 4, 4]))
    assert s["accuracy"] == pytest.approx(5 / 6)
    assert s["per_class_f1"]["LAYING"] == 0.0
    assert s["confusion"][5][4] == 1
    m, sd = mean_std([1.0, 2.0, 3.0])
    assert m == 2.0 and sd == pytest.approx(1.0)
