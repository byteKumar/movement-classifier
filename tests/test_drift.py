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


# ---------- calibrated detector ----------

def _people(n_subjects, per_subject, rng):
    """Windows from several people; each person has their own phone tilt (channel offsets)."""
    X, subj = [], []
    for s in range(n_subjects):
        tilt = rng.normal(0, 0.15, size=(1, 1, 6))
        X.append(_windows(per_subject, rng) + tilt)
        subj += [s] * per_subject
    return np.concatenate(X).astype(np.float32), np.array(subj)


def test_ks_distances_match_scipy():
    from scipy.stats import ks_2samp

    from movement.drift import ks_distances

    rng = np.random.default_rng(3)
    ref, S = rng.normal(size=(500, 12)), rng.normal(0.2, 1, size=(80, 12))
    S[:5, 0] = ref[:5, 0]  # ties
    d = ks_distances(np.sort(ref, axis=0), S)
    assert np.allclose(d, [ks_2samp(ref[:, j], S[:, j]).statistic for j in range(12)])


def test_calibrated_ignores_new_people_but_catches_sensor_faults():
    from movement.drift import CalibratedDriftDetector, null_distances

    rng = np.random.default_rng(4)
    X, subj = _people(17, 300, rng)
    ref = window_summary(X)
    null_d = null_distances(ref, subj, n_partitions=100, batches_per=20, seed=0)
    assert null_d.shape == (2000, len(FEATURES))
    cal = CalibratedDriftDetector(ref, null_d)
    assert 0 < cal.null_fire_rate <= 0.01
    new, _ = _people(4, 300, np.random.default_rng(5))  # 4 people the reference has never seen
    new_batches = [new[rng.choice(len(new), 256, replace=False)] for _ in range(40)]
    old = KSDriftDetector(ref)
    assert sum(old.check(b)["drift"] for b in new_batches) > 20  # the original problem
    assert sum(cal.check(b)["drift"] for b in new_batches) <= 8
    assert all(cal.check(drop_axis(b, 2))["drift"] for b in new_batches[:10])
    assert all(cal.check(gain(b, 1.5, [0, 1, 2]))["drift"] for b in new_batches[:10])


def test_load_detector_reads_both_formats(tmp_path):
    from movement.drift import CalibratedDriftDetector, load_detector

    ref = np.random.default_rng(6).normal(size=(300, 12))
    np.savez(tmp_path / "plain.npz", summary=ref)
    np.savez(tmp_path / "cal.npz", summary=ref, null_d=np.random.default_rng(7).random((2000, 12)), alpha=0.01)
    assert isinstance(load_detector(tmp_path / "plain.npz"), KSDriftDetector)
    assert isinstance(load_detector(tmp_path / "cal.npz"), CalibratedDriftDetector)


def test_calibrated_rejects_a_null_too_small_for_alpha():
    from movement.drift import CalibratedDriftDetector

    with pytest.raises(ValueError):
        CalibratedDriftDetector(np.zeros((10, 12)), np.zeros((50, 12)), alpha=0.01)
    with pytest.raises(ValueError):  # 12 features tie at the smallest p: 12/400 = 3% > 1%
        CalibratedDriftDetector(np.zeros((10, 12)), np.random.default_rng(0).random((400, 12)), alpha=0.01)
