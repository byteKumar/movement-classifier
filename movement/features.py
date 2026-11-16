"""Hand-crafted features for the gradient boosting baseline.

For each window (128 samples x 6 channels) we add two magnitude channels
(|acc| and |gyro|, which do not depend on how the phone is rotated) and compute
per channel: mean, std, min, max, energy, dominant frequency, and FFT band
power. We also add the correlation between each pair of accelerometer axes.
"""

from __future__ import annotations

import numpy as np

from movement.data import CHANNELS, SAMPLE_RATE_HZ, WINDOW

# Frequency bands in Hz, [low, high). 0 Hz (the mean) is excluded on purpose:
# it is already the `mean` feature. Walking cadence is ~1.5-2.5 Hz.
BANDS = [(0.3, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 25.01)]
FREQS = np.fft.rfftfreq(WINDOW, d=1.0 / SAMPLE_RATE_HZ)  # 65 bins, 0 .. 25 Hz
ALL_CHANNELS = CHANNELS + ["acc_mag", "gyro_mag"]
STATS = ["mean", "std", "min", "max", "energy", "dom_freq"] + [f"band_{lo:g}_{min(hi, 25):g}hz" for lo, hi in BANDS]
AXIS_PAIRS = [(0, 1), (0, 2), (1, 2)]


def feature_names() -> list[str]:
    names = [f"{c}_{s}" for c in ALL_CHANNELS for s in STATS]
    names += [f"corr_acc_{'xyz'[a]}{'xyz'[b]}" for a, b in AXIS_PAIRS]
    return names


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (n, WINDOW, 6) -> (n, n_features) float32."""
    if X.ndim != 3 or X.shape[1:] != (WINDOW, len(CHANNELS)):
        raise ValueError(f"expected (n, {WINDOW}, {len(CHANNELS)}), got {X.shape}")
    X = X.astype(np.float64)
    acc_mag = np.linalg.norm(X[:, :, 0:3], axis=2, keepdims=True)
    gyro_mag = np.linalg.norm(X[:, :, 3:6], axis=2, keepdims=True)
    S = np.concatenate([X, acc_mag, gyro_mag], axis=2)  # (n, T, 8)

    mean = S.mean(axis=1)
    centered = S - mean[:, None, :]
    # One-sided power spectrum of the de-meaned signal, scaled so that the sum
    # over all bins equals the signal variance (Parseval).
    spec = np.abs(np.fft.rfft(centered, axis=1)) ** 2 / WINDOW**2
    spec[:, 1:-1, :] *= 2
    bands = [spec[:, (FREQS >= lo) & (FREQS < hi), :].sum(axis=1) for lo, hi in BANDS]
    dom_freq = FREQS[1:][spec[:, 1:, :].argmax(axis=1)]  # skip the DC bin

    per_channel = np.stack(
        [mean, S.std(axis=1), S.min(axis=1), S.max(axis=1), (S**2).mean(axis=1), dom_freq, *bands],
        axis=2,
    )  # (n, 8 channels, n_stats)

    std = centered.std(axis=1)
    corr = []
    for a, b in AXIS_PAIRS:
        cov = (centered[:, :, a] * centered[:, :, b]).mean(axis=1)
        denom = std[:, a] * std[:, b]
        corr.append(np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12))

    feats = np.concatenate([per_channel.reshape(len(X), -1), np.stack(corr, axis=1)], axis=1)
    return feats.astype(np.float32)
