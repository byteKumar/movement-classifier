import numpy as np
import pytest

from movement.drift import FEATURES, KSDriftDetector, add_noise, drop_axis, gain, window_summary


def _windows(n, rng, scale=1.0):
    # each window: a random level per channel plus noise, like a mix of postures
    level = rng.normal([0, 0.8, 0.3, 0, 0, 0], 0.3, size=(n, 1, 6))
    return (level + rng.normal(0, 0.2, size=(n, 128, 6))).astype(np.float32) * scale


def test_summary_shape_and_values():
    X = np.zeros((3, 128, 6), np.float32)
    X[:, :, 2] = 5
    S = window_summary(X)
    assert S.shape == (3, len(FEATURES))
    assert (S[:, FEATURES.index("acc_z_mean")] == 5).all() and (S[:, 6:] == 0).all()


def test_rarely_fires_on_the_same_distribution():
    rng = np.random.default_rng(0)
    det = KSDriftDetector.from_windows(_windows(4000, rng))
    fired = sum(det.check(_windows(256, rng))["drift"] for _ in range(100))
    # expected false-alarm rate is at most alpha = 1%
    assert fired <= 4


def test_fires_on_a_large_gain_change():
    rng = np.random.default_rng(1)
    det = KSDriftDetector.from_windows(_windows(4000, rng))
    r = det.check(gain(_windows(256, rng), 1.5, [0, 1, 2]))
    assert r["drift"] and r["min_p"] < det.threshold and r["drifted_features"]


def test_shift_functions():
    rng = np.random.default_rng(2)
    X = _windows(50, rng)
    assert (drop_axis(X, 4)[:, :, 4] == 0).all() and np.array_equal(drop_axis(X, 4)[:, :, :4], X[:, :, :4])
    g = gain(X, 2.0, [0])
    assert np.allclose(g[:, :, 0], 2 * X[:, :, 0]) and np.array_equal(g[:, :, 1:], X[:, :, 1:])
    std = np.ones(6)
    noisy = add_noise(np.zeros((200, 128, 6), np.float32), 0.5, std, rng)
    assert noisy.std() == pytest.approx(0.5, rel=0.02)
    assert np.array_equal(X, _windows(50, np.random.default_rng(2)))  # inputs not modified


def test_rejects_bad_reference():
    with pytest.raises(ValueError):
        KSDriftDetector(np.zeros((10, 5)))
