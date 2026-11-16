"""Input drift check: two-sample Kolmogorov-Smirnov test per summary feature.

Each window is summarized by the mean and the std of each of its 6 channels
(12 numbers). Testing summaries rather than raw samples matters: the 128
samples inside a window are strongly correlated, and KS assumes independent
draws, so a test on raw samples would fire on almost anything.

A batch is flagged as drifted if any of the 12 features has a KS p-value
below alpha / 12 (Bonferroni correction for testing 12 features at once).
The defaults (alpha 0.01, batches of 256 windows) were fixed before the
evaluation in scripts/drift_eval.py was run and were not tuned afterwards.
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
