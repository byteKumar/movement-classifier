from pathlib import Path

import numpy as np
import pytest

from movement.data import (
    TEST_SUBJECTS,
    VAL_SUBJECTS,
    WINDOW,
    Windows,
    load_windows,
    make_windows,
    split_by_subject,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "hapt" / "RawData"
needs_data = pytest.mark.skipif(not RAW.exists(), reason="run scripts/download_data.py first")

# ---------- windowing on synthetic signals ----------


def test_windows_are_full_length_and_half_overlapping():
    sig = np.zeros((1000, 6))
    starts = make_windows(sig, 0, 1000)
    assert starts[:3] == [0, 64, 128]
    assert all(s + WINDOW <= 1000 for s in starts)
    # floor((1000 - 128) / 64) + 1 = 14 windows
    assert len(starts) == 14


def test_windows_stay_inside_segment():
    sig = np.zeros((1000, 6))
    starts = make_windows(sig, 100, 400)
    assert starts[0] == 100
    assert all(100 <= s and s + WINDOW <= 400 for s in starts)


def test_segment_shorter_than_window_gives_nothing():
    assert make_windows(np.zeros((500, 6)), 10, 10 + WINDOW - 1) == []
    assert make_windows(np.zeros((500, 6)), 10, 10 + WINDOW) == [10]


def test_segment_past_end_raises():
    with pytest.raises(ValueError):
        make_windows(np.zeros((100, 6)), 0, 200)


def test_load_windows_on_tiny_fake_dataset(tmp_path):
    # exp 1, user 3: 400 samples, label WALKING on 1-based [1, 200], SITTING on [201, 400]
    acc = np.arange(400 * 3, dtype=np.float32).reshape(400, 3)
    gyro = -acc
    np.savetxt(tmp_path / "acc_exp01_user03.txt", acc)
    np.savetxt(tmp_path / "gyro_exp01_user03.txt", gyro)
    (tmp_path / "labels.txt").write_text("1 3 1 1 200\n1 3 4 201 400\n1 3 7 1 50\n")
    w = load_windows(tmp_path)
    # each 200-sample segment fits floor((200-128)/64)+1 = 2 windows; label 7 is dropped
    assert len(w) == 4
    assert w.y.tolist() == [0, 0, 3, 3]
    assert w.start.tolist() == [0, 64, 200, 264]
    assert w.X.shape == (4, WINDOW, 6)
    # window content is the exact slice of the raw signal, acc then gyro
    np.testing.assert_array_equal(w.X[2, :, :3], acc[200:328])
    np.testing.assert_array_equal(w.X[2, :, 3:], gyro[200:328])

# ---------- subject split ----------


def _fake(subjects):
    n = len(subjects)
    return Windows(np.zeros((n, WINDOW, 6), np.float32), np.zeros(n, np.int64),
                   np.asarray(subjects), np.zeros(n, np.int64), np.zeros(n, np.int64))


def test_split_puts_each_subject_in_one_part():
    w = _fake(list(range(1, 31)) * 3)
    train, val, test = split_by_subject(w)
    s_tr, s_va, s_te = set(train.subject), set(val.subject), set(test.subject)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    assert s_te == set(TEST_SUBJECTS) and s_va == set(VAL_SUBJECTS)
    assert len(train) + len(val) + len(test) == len(w)


def test_split_rejects_overlapping_val_and_test():
    with pytest.raises(ValueError):
        split_by_subject(_fake([1, 2, 3]), test_subjects=(2,), val_subjects=(2, 3))

# ---------- leakage checks on the real dataset ----------


@pytest.fixture(scope="module")
def real():
    return load_windows(RAW)


@needs_data
def test_real_dataset_shape_and_labels(real):
    assert real.X.shape[1:] == (WINDOW, 6)
    assert set(np.unique(real.y)) == set(range(6))
    assert set(np.unique(real.subject)) == set(range(1, 31))
    assert np.isfinite(real.X).all()


@needs_data
def test_real_split_has_no_subject_leakage(real):
    train, val, test = split_by_subject(real)
    assert set(train.subject).isdisjoint(test.subject)
    assert set(train.subject).isdisjoint(val.subject)
    assert set(val.subject).isdisjoint(test.subject)
    # recordings (experiments) never straddle splits either
    assert set(train.exp).isdisjoint(test.exp) and set(val.exp).isdisjoint(test.exp)


def count_sample_sharing(a: Windows, b: Windows) -> int:
    """How many windows in `b` share at least one raw sample with a window in `a`.

    With 50% overlap, a random per-window split leaks this way: the test
    window is never identical to a train window, but half of it is.
    """
    by_exp: dict[int, np.ndarray] = {}
    for e in np.unique(a.exp):
        by_exp[e] = np.sort(a.start[a.exp == e])
    n = 0
    for e, s in zip(b.exp, b.start):
        starts = by_exp.get(e)
        if starts is None:
            continue
        i = np.searchsorted(starts, s)
        near = starts[max(i - 1, 0): i + 1]
        n += bool(np.any(np.abs(near - s) < WINDOW))
    return n


@needs_data
def test_no_test_window_shares_samples_with_train_or_val(real):
    train, val, test = split_by_subject(real)
    assert count_sample_sharing(train, test) == 0
    assert count_sample_sharing(val, test) == 0
    assert count_sample_sharing(train, val) == 0


@needs_data
def test_sharing_check_catches_a_random_window_split(real):
    # Guard against the check above passing vacuously.
    part = np.random.default_rng(0).random(len(real)) < 0.7
    assert count_sample_sharing(real.subset(part), real.subset(~part)) > 0


@needs_data
def test_each_recording_belongs_to_one_subject(real):
    # Overlapping windows share samples only within one recording, so if every
    # recording maps to one subject, overlap can never cross the split.
    for e in np.unique(real.exp):
        assert len(np.unique(real.subject[real.exp == e])) == 1


def test_val_subjects_match_their_documented_derivation():
    train_pool = sorted(set(range(1, 31)) - set(TEST_SUBJECTS))
    drawn = np.random.default_rng(0).choice(train_pool, 4, replace=False)
    assert tuple(sorted(drawn.tolist())) == VAL_SUBJECTS
