"""Window the raw recordings once and cache them as data/windows.npz.

Usage: python scripts/prepare_data.py
"""

from pathlib import Path

import numpy as np

from movement.data import CLASS_NAMES, load_windows, save_windows, split_by_subject

ROOT = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    w = load_windows(ROOT / "hapt" / "RawData")
    save_windows(w, ROOT / "windows.npz")
    train, val, test = split_by_subject(w)
    print(f"windows: {len(w)}  shape {w.X.shape}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        counts = np.bincount(part.y, minlength=len(CLASS_NAMES))
        print(f"{name:5s} subjects={len(np.unique(part.subject)):2d} windows={len(part):5d} per-class={counts.tolist()}")


if __name__ == "__main__":
    main()
