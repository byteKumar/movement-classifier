import json
import logging

import numpy as np

from movement.drift import BATCH, KSDriftDetector, window_summary
from movement.monitoring import Monitor


def _batches(rng, sizes):
    return [rng.normal([0, 0.8, 0.3, 0, 0, 0], 0.5, size=(n, 128, 6)).astype(np.float32) for n in sizes]


def test_running_input_stats_match_numpy_over_all_requests():
    rng = np.random.default_rng(0)
    m = Monitor()
    bs = _batches(rng, [1, 32, 5, 17])
    for b in bs:
        m.record(b, np.zeros(len(b), dtype=np.int64))
    allx = np.concatenate(bs).reshape(-1, 6)
    snap = m.snapshot()
    assert snap["requests"] == 4 and snap["windows"] == 55
    for j, c in enumerate(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]):
        st = snap["input_stats"][c]
        assert np.isclose(st["mean"], allx[:, j].mean(), atol=1e-9)
        assert np.isclose(st["std"], allx[:, j].std(), atol=1e-9)
        assert st["min"] == allx[:, j].min() and st["max"] == allx[:, j].max()


def test_class_distribution():
    m = Monitor()
    m.record(np.zeros((4, 128, 6), np.float32), np.array([0, 3, 3, 5]))
    snap = m.snapshot()
    assert snap["predicted_class_counts"]["SITTING"] == 2
    assert snap["predicted_class_share"]["LAYING"] == 0.25


def test_drift_needs_a_full_batch_then_runs():
    rng = np.random.default_rng(1)
    ref = _batches(rng, [2000])[0]
    m = Monitor(KSDriftDetector(window_summary(ref)))
    m.record(_batches(rng, [100])[0], np.zeros(100, dtype=np.int64))
    assert m.drift()["drift"] is None
    b = _batches(rng, [BATCH])[0]
    b[:, :, 2] = 0  # dead acc_z axis
    m.record(b, np.zeros(BATCH, dtype=np.int64))
    r = m.drift()
    assert r["drift"] is True and "acc_z_std" in r["drifted_features"]


def test_logs_a_summary_line_every_n_windows(caplog):
    m = Monitor(log_every=10)
    with caplog.at_level(logging.INFO, logger="movement.monitor"):
        for _ in range(5):
            m.record(np.zeros((4, 128, 6), np.float32), np.zeros(4, dtype=np.int64))
    lines = [r.message for r in caplog.records if r.name == "movement.monitor"]
    assert len(lines) == 2  # crossed 10 and 20 windows
    assert json.loads(lines[0])["windows"] == 12
