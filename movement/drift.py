"""Input drift check: two-sample Kolmogorov-Smirnov test per summary feature.

Each window is summarized by the mean and the std of each of its 6 channels
(12 numbers). Testing summaries rather than raw samples matters: the 128
samples inside a window are strongly correlated, and KS assumes independent
draws, so a test on raw samples would fire on almost anything.

A batch is flagged as drifted if any of the 12 features has a KS p-value
below alpha / 12 (Bonferroni correction for testing 12 features at once).
The defaults (alpha 0.01, batches of 256 windows) were fixed before the
evaluation in scripts/drift_eval.py was run and were not tuned afterwards.

That first check (KSDriftDetector) failed on clean data: it flagged 47 of 200
batches from the test subjects and 200 of 200 from the validation subjects.
A p-value from a reference of ~6,000 windows is small for any real
difference, and different people (phone tilt, gait) are a real difference.
It answers "is this the same 17 people?", not "is the sensor broken?".

CalibratedDriftDetector keeps the KS distance D per feature but judges it
against what new *people* look like: a null distribution of D built by
holding out groups of training subjects (null_distances). A batch's
per-feature empirical p-value is the share of null batches with a D at least
as large; the batch alarms when the smallest of the 12 is at or below the
alpha-quantile of the same quantity over the null batches (min-p, which
handles the 12 correlated features without Bonferroni). See
scripts/drift_calibrated.py for how it was chosen without the test subjects.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from movement.data import CHANNELS

FEATURES = [f"{c}_mean" for c in CHANNELS] + [f"{c}_std" for c in CHANNELS]
ALPHA = 0.01
BATCH = 256


def window_summary(X: np.ndarray) -> np.ndarray:
    """(n, 128, 6) -> (n, 12): per-channel mean, then per-channel std."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


class KSDriftDetector:
    def __init__(self, reference_summary: np.ndarray, alpha: float = ALPHA):
        if reference_summary.ndim != 2 or reference_summary.shape[1] != len(FEATURES):
            raise ValueError(f"reference must be (n, {len(FEATURES)}) window summaries")
        self.ref = np.asarray(reference_summary, dtype=np.float64)
        self.alpha = alpha
        self.threshold = alpha / len(FEATURES)

    @classmethod
    def from_windows(cls, X_ref: np.ndarray, alpha: float = ALPHA) -> "KSDriftDetector":
        return cls(window_summary(X_ref), alpha)

    def check_summary(self, S: np.ndarray) -> dict:
        p = np.array([ks_2samp(self.ref[:, j], S[:, j]).pvalue for j in range(len(FEATURES))])
        worst = int(p.argmin())
        return {
            "drift": bool((p < self.threshold).any()),
            "n_windows": int(len(S)),
            "threshold": self.threshold,
            "min_p": float(p[worst]),
            "worst_feature": FEATURES[worst],
            "drifted_features": [FEATURES[j] for j in np.where(p < self.threshold)[0]],
        }

    def check(self, X: np.ndarray) -> dict:
        return self.check_summary(window_summary(X))


def ks_distances(ref_sorted: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic D for each column. ref_sorted: (n, k) sorted along axis 0."""
    S = np.sort(np.asarray(S, dtype=np.float64), axis=0)
    n, m = len(ref_sorted), len(S)
    out = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        z = np.concatenate([ref_sorted[:, j], S[:, j]])
        f_ref = np.searchsorted(ref_sorted[:, j], z, side="right") / n
        f_s = np.searchsorted(S[:, j], z, side="right") / m
        out[j] = np.abs(f_ref - f_s).max()
    return out


def null_distances(summary: np.ndarray, subjects: np.ndarray, n_partitions: int = 200, holdout: int = 4,
                   batches_per: int = 25, batch: int = BATCH, seed: int = 0) -> np.ndarray:
    """KS distances of clean batches from people the reference has not seen.

    Each partition holds out `holdout` subjects, uses the rest as the reference,
    and draws `batches_per` batches from the held-out subjects' windows.
    Returns (n_partitions * batches_per, 12).
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(subjects)
    out = []
    for _ in range(n_partitions):
        held = np.isin(subjects, rng.choice(ids, holdout, replace=False))
        ref_sorted = np.sort(summary[~held].astype(np.float64), axis=0)
        pool = summary[held]
        for _ in range(batches_per):
            out.append(ks_distances(ref_sorted, pool[rng.choice(len(pool), batch, replace=False)]))
    return np.array(out)


class CalibratedDriftDetector:
    def __init__(self, reference_summary: np.ndarray, null_d: np.ndarray, alpha: float = ALPHA):
        if reference_summary.ndim != 2 or reference_summary.shape[1] != len(FEATURES):
            raise ValueError(f"reference must be (n, {len(FEATURES)}) window summaries")
        if null_d.ndim != 2 or null_d.shape[1] != len(FEATURES):
            raise ValueError(f"null distances must be (n, {len(FEATURES)})")
        self.ref_sorted = np.sort(np.asarray(reference_summary, dtype=np.float64), axis=0)
        self.null_sorted = np.sort(np.asarray(null_d, dtype=np.float64), axis=0)
        self.alpha = alpha
        n = len(null_d)
        # each null batch against the null itself (it counts itself, so no +1)
        null_minp = (self._at_least(null_d) / (n + 1)).min(axis=1)
        # min-p values tie (each feature's most extreme null batch gets the same p),
        # so take the largest value whose share of null batches at or below it is <= alpha
        values, counts = np.unique(null_minp, return_counts=True)
        ok = np.cumsum(counts) / n <= alpha
        if not ok[0]:
            raise ValueError(f"{n} null batches are too few for alpha={alpha}")
        self.threshold = float(values[ok][-1])
        self.null_fire_rate = float((null_minp <= self.threshold).mean())

    def _at_least(self, D: np.ndarray) -> np.ndarray:
        """How many null batches have a distance >= D, per feature."""
        n = len(self.null_sorted)
        return np.stack([n - np.searchsorted(self.null_sorted[:, j], D[:, j], side="left")
                         for j in range(D.shape[1])], axis=1)

    def check_summary(self, S: np.ndarray) -> dict:
        d = ks_distances(self.ref_sorted, S)
        p = (1 + self._at_least(d[None, :])[0]) / (len(self.null_sorted) + 1)
        worst = int(p.argmin())
        return {
            "drift": bool(p.min() <= self.threshold),
            "n_windows": int(len(S)),
            "threshold": self.threshold,
            "min_p": float(p[worst]),
            "worst_feature": FEATURES[worst],
            "worst_ks_d": float(d[worst]),
            "drifted_features": [FEATURES[j] for j in np.where(p <= self.threshold)[0]],
        }

    def check(self, X: np.ndarray) -> dict:
        return self.check_summary(window_summary(X))


def load_detector(path) -> KSDriftDetector | CalibratedDriftDetector:
    """The service's detector: calibrated if the reference file carries a null."""
    ref = np.load(path)
    if "null_d" in ref.files:
        return CalibratedDriftDetector(ref["summary"], ref["null_d"], float(ref["alpha"]))
    return KSDriftDetector(ref["summary"])


# ---------- synthetic shifts used to evaluate the check ----------

def add_noise(X: np.ndarray, frac: float, channel_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with std = frac x each channel's training std."""
    return (X + rng.normal(size=X.shape) * (frac * channel_std)).astype(np.float32)


def gain(X: np.ndarray, factor: float, channels) -> np.ndarray:
    """Multiply the given channels by `factor` (a mis-calibrated sensor)."""
    out = X.copy()
    out[:, :, list(channels)] *= factor
    return out


def drop_axis(X: np.ndarray, channel: int) -> np.ndarray:
    """One axis reads 0 (a dead or disconnected axis)."""
    out = X.copy()
    out[:, :, channel] = 0.0
    return out
