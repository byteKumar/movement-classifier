"""Metrics shared by every model, so baseline and CNN are scored identically."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from movement.data import CLASS_NAMES


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {name: float(f) for name, f in zip(CLASS_NAMES, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mean_std(values) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1), the usual choice over seeds."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
