"""In-process monitoring for the service: what comes in and what goes out.

Keeps, since startup:
  - how many windows were predicted as each class (output distribution)
  - per-channel count / mean / std / min / max of every input sample (Welford's
    method, so no samples are stored)
and a ring buffer of the most recent window summaries for the drift check.
A one-line JSON summary is logged every `log_every` windows.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque

import numpy as np

from movement.data import CHANNELS, CLASS_NAMES
from movement.drift import BATCH, KSDriftDetector, window_summary

log = logging.getLogger("movement.monitor")


class Monitor:
    def __init__(self, detector: KSDriftDetector | None = None, recent: int = 2048, log_every: int = 10_000):
        self.detector = detector
        self.log_every = log_every
        self._lock = threading.Lock()
        self._recent = deque(maxlen=recent)
        self.requests = 0
        self.windows = 0
        self.class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        self._n = 0
        self._mean = np.zeros(len(CHANNELS))
        self._m2 = np.zeros(len(CHANNELS))
        self._min = np.full(len(CHANNELS), np.inf)
        self._max = np.full(len(CHANNELS), -np.inf)
        self._next_log = log_every

    def record(self, X: np.ndarray, class_ids: np.ndarray) -> None:
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        n_b = len(flat)
        mean_b = flat.mean(axis=0)
        m2_b = ((flat - mean_b) ** 2).sum(axis=0)
        summaries = window_summary(X)
        with self._lock:
            self.requests += 1
            self.windows += len(X)
            self.class_counts += np.bincount(class_ids, minlength=len(CLASS_NAMES))
            # Chan et al. parallel update of running mean / sum of squared deviations
            n = self._n + n_b
            delta = mean_b - self._mean
            self._mean = self._mean + delta * n_b / n
            self._m2 = self._m2 + m2_b + delta**2 * self._n * n_b / n
            self._n = n
            self._min = np.minimum(self._min, flat.min(axis=0))
            self._max = np.maximum(self._max, flat.max(axis=0))
            self._recent.extend(summaries)
            should_log = self.windows >= self._next_log
            if should_log:
                self._next_log += self.log_every
        if should_log:
            log.info(json.dumps(self.snapshot()))

    def snapshot(self) -> dict:
        with self._lock:
            std = np.sqrt(self._m2 / self._n) if self._n else np.zeros(len(CHANNELS))
            total = max(int(self.class_counts.sum()), 1)
            return {
                "requests": self.requests,
                "windows": self.windows,
                "predicted_class_counts": dict(zip(CLASS_NAMES, self.class_counts.tolist())),
                "predicted_class_share": {c: round(int(k) / total, 4) for c, k in zip(CLASS_NAMES, self.class_counts)},
                "input_stats": {c: {"mean": float(m), "std": float(s),
                                    "min": float(lo) if self._n else None, "max": float(hi) if self._n else None}
                                for c, m, s, lo, hi in zip(CHANNELS, self._mean, std, self._min, self._max)},
            }

    def drift(self) -> dict:
        if self.detector is None:
            return {"drift": None, "reason": "no drift reference loaded"}
        with self._lock:
            recent = np.array(self._recent)
        if len(recent) < BATCH:
            return {"drift": None, "reason": f"need {BATCH} recent windows, have {len(recent)}"}
        return self.detector.check_summary(recent[-BATCH:])
