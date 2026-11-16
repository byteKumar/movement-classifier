"""Evaluate the KS drift check on clean data and on synthetic sensor shifts.

Plan (fixed before the first run, not tuned afterwards):
  reference   : window summaries of the 17 training subjects
  clean       : 200 batches of 256 windows drawn at random (without replacement
                inside a batch) from the 9 test subjects; also 200 batches from
                the 4 validation subjects, as a second unseen population
  shifts      : applied to the same 200 test-subject batches
                  noise 0.1 x channel std, noise 0.5 x channel std,
                  accelerometer gain x1.1, x1.5,
                  acc_z axis dropped (reads 0), gyro_y axis dropped
  check       : KS per feature, alpha 0.01 with Bonferroni over 12 features
  "detected"  : a shift counts as detected if the check fires on >= 50% of its batches
  false alarm : a clean batch on which the check fires

Result: 6 of 6 shifts detected, but 47 of 200 clean test batches and 200 of
200 clean val batches flagged. scripts/drift_calibrated.py is the follow-up
and now writes the reference file the service uses for GET /drift.

Usage: python scripts/drift_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from movement.data import CHANNELS, load_cached, split_by_subject
from movement.drift import ALPHA, BATCH, KSDriftDetector, add_noise, drop_axis, gain

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
N_BATCHES = 200
DETECTED_IF = 0.5


def batches(X: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    return [X[rng.choice(len(X), BATCH, replace=False)] for _ in range(N_BATCHES)]


def main() -> None:
    train, val, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))
    det = KSDriftDetector.from_windows(train.X)
    ART.mkdir(exist_ok=True)
    ch_std = train.X.reshape(-1, 6).std(axis=0)

    rng = np.random.default_rng(0)
    test_batches = batches(test.X, rng)
    val_batches = batches(val.X, rng)
    shifts = {
        "noise 0.1 std": lambda X: add_noise(X, 0.1, ch_std, rng),
        "noise 0.5 std": lambda X: add_noise(X, 0.5, ch_std, rng),
        "acc gain x1.1": lambda X: gain(X, 1.1, [0, 1, 2]),
        "acc gain x1.5": lambda X: gain(X, 1.5, [0, 1, 2]),
        "acc_z dropped": lambda X: drop_axis(X, CHANNELS.index("acc_z")),
        "gyro_y dropped": lambda X: drop_axis(X, CHANNELS.index("gyro_y")),
    }

    def run(name, bs):
        res = [det.check(b) for b in bs]
        fired = sum(r["drift"] for r in res)
        worst = {}
        for r in res:
            worst[r["worst_feature"]] = worst.get(r["worst_feature"], 0) + 1
        row = {"name": name, "fired": fired, "batches": len(bs), "rate": fired / len(bs),
               "median_min_p": float(np.median([r["min_p"] for r in res])),
               "most_common_worst_feature": max(worst, key=worst.get)}
        print(f"{name:22s} fired {fired:3d}/{len(bs)} ({row['rate']:.1%})  median min p={row['median_min_p']:.2e}  "
              f"worst feature: {row['most_common_worst_feature']}")
        return row

    print(f"reference: {len(train)} windows from 17 training subjects; batch={BATCH}; alpha={ALPHA} "
          f"(per-feature threshold {det.threshold:.2e})")
    clean = [run("clean: test subjects", test_batches), run("clean: val subjects", val_batches)]
    shifted = [run(name, [f(b) for b in test_batches]) for name, f in shifts.items()]
    detected = sum(r["rate"] >= DETECTED_IF for r in shifted)
    print(f"shifts detected: {detected} of {len(shifted)}; false alarms on clean test batches: "
          f"{clean[0]['fired']} of {clean[0]['batches']}")
    (ART / "drift_eval.json").write_text(json.dumps({
        "alpha": ALPHA, "batch": BATCH, "n_batches": N_BATCHES, "detected_if_rate": DETECTED_IF,
        "clean": clean, "shifts": shifted, "shifts_detected": detected, "shifts_total": len(shifted),
    }, indent=2))


if __name__ == "__main__":
    main()
