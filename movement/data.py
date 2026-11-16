"""Load raw HAPT recordings, cut fixed windows, and split by subject.

Raw data layout (UCI HAPT, `RawData/`):
  acc_expXX_userYY.txt   one row per sample, 3 columns (x, y, z) in g, 50 Hz
  gyro_expXX_userYY.txt  same, angular velocity in rad/s
  labels.txt             exp_id user_id activity_id start end
                         (start/end are 1-based, inclusive sample indices)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
WINDOW = 128  # 2.56 s at 50 Hz, as in the UCI HAR dataset
STRIDE = 64  # 50% overlap

# The six basic activities. Postural transitions (ids 7-12) are dropped.
CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# The dataset authors' own test subjects (Test/subject_id_test.txt). Keeping
# them as our test set makes numbers comparable to published UCI HAR results.
TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)
# Held out from the 21 training subjects for early stopping / model selection.
# Picked once with np.random.default_rng(0) and then frozen here.
VAL_SUBJECTS = (8, 17, 21, 25)

_FILE_RE = re.compile(r"acc_exp(\d+)_user(\d+)\.txt$")


@dataclass
class Windows:
    X: np.ndarray  # (n, WINDOW, 6) float32
    y: np.ndarray  # (n,) int64, 0..5 indexes CLASS_NAMES
    subject: np.ndarray  # (n,) int64
    exp: np.ndarray  # (n,) int64, experiment (recording) id
    start: np.ndarray  # (n,) int64, 0-based start sample within the recording

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Windows":
        return Windows(self.X[mask], self.y[mask], self.subject[mask], self.exp[mask], self.start[mask])


def make_windows(signal: np.ndarray, start: int, end: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """Start offsets of every full window inside signal[start:end].

    `end` is exclusive. A window is kept only if it lies entirely inside the
    segment, so no window mixes two activities. The trailing partial window is dropped.
    """
    if end > len(signal):
        raise ValueError(f"segment end {end} past signal length {len(signal)}")
    return list(range(start, end - window + 1, stride))


def _read_xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32).to_numpy()


def load_windows(raw_dir: str | Path) -> Windows:
    """Read every recording in `raw_dir` and return all labeled windows."""
    raw_dir = Path(raw_dir)
    labels = pd.read_csv(
        raw_dir / "labels.txt", sep=r"\s+", header=None,
        names=["exp", "user", "activity", "start", "end"],
    )
    labels = labels[labels.activity.between(1, len(CLASS_NAMES))]

    X, y, subj, exps, starts = [], [], [], [], []
    for acc_path in sorted(raw_dir.glob("acc_exp*_user*.txt")):
        exp_id, user_id = (int(g) for g in _FILE_RE.search(acc_path.name).groups())
        gyro_path = raw_dir / acc_path.name.replace("acc_", "gyro_")
        acc, gyro = _read_xyz(acc_path), _read_xyz(gyro_path)
        if len(acc) != len(gyro):
            raise ValueError(f"{acc_path.name}: acc has {len(acc)} rows, gyro {len(gyro)}")
        signal = np.concatenate([acc, gyro], axis=1)  # (T, 6)

        for row in labels[labels.exp == exp_id].itertuples():
            if row.user != user_id:
                raise ValueError(f"labels.txt says exp {exp_id} is user {row.user}, file says {user_id}")
            # 1-based inclusive [start, end]  ->  0-based half-open [start-1, end)
            for s in make_windows(signal, row.start - 1, row.end):
                X.append(signal[s : s + WINDOW])
                y.append(row.activity - 1)
                subj.append(user_id)
                exps.append(exp_id)
                starts.append(s)

    return Windows(
        X=np.stack(X).astype(np.float32),
        y=np.asarray(y, dtype=np.int64),
        subject=np.asarray(subj, dtype=np.int64),
        exp=np.asarray(exps, dtype=np.int64),
        start=np.asarray(starts, dtype=np.int64),
    )


def split_by_subject(
    w: Windows,
    test_subjects=TEST_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
) -> tuple[Windows, Windows, Windows]:
    """Train / val / test split where each subject lands in exactly one part."""
    test_subjects, val_subjects = set(test_subjects), set(val_subjects)
    if test_subjects & val_subjects:
        raise ValueError("a subject is in both val and test")
    is_test = np.isin(w.subject, list(test_subjects))
    is_val = np.isin(w.subject, list(val_subjects))
    is_train = ~(is_test | is_val)
    return w.subset(is_train), w.subset(is_val), w.subset(is_test)


def save_windows(w: Windows, path: str | Path) -> None:
    np.savez_compressed(path, X=w.X, y=w.y, subject=w.subject, exp=w.exp, start=w.start)


def load_cached(path: str | Path) -> Windows:
    d = np.load(path)
    return Windows(d["X"], d["y"], d["subject"], d["exp"], d["start"])
