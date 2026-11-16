import numpy as np

from movement.evaluation import hardest_pair, plot_confusion, row_normalize


def test_row_normalize_handles_empty_rows():
    r = row_normalize([[2, 2], [0, 0]])
    assert r.tolist() == [[0.5, 0.5], [0.0, 0.0]]


def test_hardest_pair_uses_rates_not_counts():
    conf = np.diag([1000, 1000, 10, 10, 1000, 1000])
    conf[0, 1] = 30   # 30 of 1030 walking -> upstairs: 2.9%
    conf[2, 3] = 5    # 5 of 15 downstairs -> sitting: 33%
    hp = hardest_pair(conf)
    assert hp["classes"] == ["WALKING_DOWNSTAIRS", "SITTING"]
    assert hp["a_as_b"] == 5 and hp["b_as_a"] == 0


def test_hardest_pair_counts_both_directions():
    conf = np.diag([100] * 6)
    conf[3, 4], conf[4, 3] = 10, 10  # sitting <-> standing, 9% each way
    conf[0, 1] = 15                  # walking -> upstairs, 13% one way
    assert hardest_pair(conf)["classes"] == ["SITTING", "STANDING"]


def test_plot_writes_a_png(tmp_path):
    p = tmp_path / "c.png"
    plot_confusion(np.eye(6, dtype=int) * 10, "t", p)
    assert p.stat().st_size > 1000 and p.read_bytes()[:4] == b"\x89PNG"
