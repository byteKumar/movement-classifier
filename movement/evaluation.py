"""Confusion-matrix helpers used by scripts/evaluate.py."""

from __future__ import annotations

import numpy as np

from movement.data import CLASS_NAMES


def row_normalize(conf) -> np.ndarray:
    """Each row (true class) sums to 1: entry [i, j] = share of class i predicted as j."""
    c = np.asarray(conf, dtype=np.float64)
    totals = c.sum(axis=1, keepdims=True)
    return np.divide(c, totals, out=np.zeros_like(c), where=totals > 0)


def hardest_pair(conf) -> dict:
    """The two classes most often mistaken for each other (in either direction).

    Uses row-normalized rates so a big class doesn't win just by being big.
    """
    r = row_normalize(conf)
    best, pair = -1.0, (0, 0)
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if r[i, j] + r[j, i] > best:
                best, pair = r[i, j] + r[j, i], (i, j)
    i, j = pair
    c = np.asarray(conf)
    return {
        "classes": [CLASS_NAMES[i], CLASS_NAMES[j]],
        "a_as_b": int(c[i, j]), "b_as_a": int(c[j, i]),
        "a_as_b_rate": float(r[i, j]), "b_as_a_rate": float(r[j, i]),
    }


def plot_confusion(conf, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = row_normalize(conf)
    c = np.asarray(conf)
    labels = [n.replace("WALKING_", "WALK_").replace("_", " ").lower() for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=120)
    im = ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(r)):
        for j in range(len(r)):
            ax.text(j, i, f"{c[i, j]}\n{r[i, j]:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if r[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
