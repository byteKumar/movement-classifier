"""Calibrate the drift check on new people, decide on val, then report on test once.

Why: the first check (scripts/drift_eval.py, KS p-values, alpha 0.01 with
Bonferroni) flagged 47 of 200 clean test-subject batches and 200 of 200 clean
val-subject batches. It treats "people the reference has not seen" as drift.

Plan (fixed and committed before either stage was run):
  null       : 200 random partitions of the 17 training subjects into 13 reference
               and 4 held-out subjects (seed 0); 25 batches of 256 windows from the
               held-out subjects per partition = 5,000 clean "new people" batches.
               Per batch, the KS distance D of each of the 12 summary features
               against the 13-subject reference. Only training subjects are used.
               Groups of 4 match the val group; a batch mixed from fewer people
               differs more from the reference, so this errs towards fewer alarms.
  detector   : CalibratedDriftDetector. Reference = all 17 training subjects (what
               the service uses). Per-feature empirical p-value of D against the
               null; alarm when the smallest of the 12 is at or below the 1%
               quantile of the same min-p over the null batches (alpha 0.01).
  stage val  : decides, without using the test subjects. 200 clean batches of 256
               windows from the 4 val subjects (seed 1) and the same 6 shifts as
               drift_eval.py applied to them. A shift is detected if the check fires
               on >= 50% of its batches.
               Adopt the calibrated detector if val false alarms <= 10 of 200 (5%)
               AND >= 4 of 6 shifts are detected. Otherwise keep the KS p-value
               detector and report both.
               The decision goes to artifacts/drift_selection.json.
  stage test : runs once, only after the decision exists. Exactly the batches of
               drift_eval.py (seed 0, the same draw order), so the original
               detector's 47 of 200 must reproduce. Reports both detectors.

Usage: python scripts/drift_calibrated.py --stage val
       python scripts/drift_calibrated.py --stage test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from movement.data import CHANNELS, load_cached, split_by_subject
from movement.drift import (ALPHA, BATCH, CalibratedDriftDetector, KSDriftDetector, add_noise, drop_axis, gain,
                            null_distances, window_summary)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
N_BATCHES = 200
DETECTED_IF = 0.5
MAX_VAL_FALSE_ALARMS = 10
MIN_SHIFTS_DETECTED = 4
NULL = dict(n_partitions=200, holdout=4, batches_per=25, batch=BATCH, seed=0)


def batches(X: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    return [X[rng.choice(len(X), BATCH, replace=False)] for _ in range(N_BATCHES)]


def shift_fns(ch_std: np.ndarray, rng: np.random.Generator) -> dict:
    return {
        "noise 0.1 std": lambda X: add_noise(X, 0.1, ch_std, rng),
        "noise 0.5 std": lambda X: add_noise(X, 0.5, ch_std, rng),
        "acc gain x1.1": lambda X: gain(X, 1.1, [0, 1, 2]),
        "acc gain x1.5": lambda X: gain(X, 1.5, [0, 1, 2]),
        "acc_z dropped": lambda X: drop_axis(X, CHANNELS.index("acc_z")),
        "gyro_y dropped": lambda X: drop_axis(X, CHANNELS.index("gyro_y")),
    }


def run(dets: dict, name: str, bs: list[np.ndarray]) -> dict:
    S = [window_summary(b) for b in bs]
    row = {"name": name, "batches": len(bs)}
    for key, det in dets.items():
        res = [det.check_summary(s) for s in S]
        row[key] = sum(r["drift"] for r in res)
    print(f"{name:22s} " + "  ".join(f"{k}: {row[k]:3d}/{len(bs)}" for k in dets))
    return row


def summarize(rows_clean: dict, rows_shift: list[dict], keys) -> dict:
    return {k: {"false_alarms": rows_clean[k], "shifts_detected": sum(r[k] / r["batches"] >= DETECTED_IF
                                                                     for r in rows_shift)} for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    stage = ap.parse_args().stage
    w = load_cached(ROOT / "data" / "windows.npz")
    train, val, test = split_by_subject(w)
    del w
    ref = window_summary(train.X)
    ch_std = train.X.reshape(-1, 6).std(axis=0)
    null_path = ART / "drift_null.npy"
    if stage == "val":
        null_d = null_distances(ref, train.subject, **NULL)
        np.save(null_path, null_d)
    else:
        null_d = np.load(null_path)
    dets = {"ks_pvalue": KSDriftDetector(ref, ALPHA), "calibrated": CalibratedDriftDetector(ref, null_d, ALPHA)}
    cal = dets["calibrated"]
    print(f"null: {len(null_d)} batches; min-p threshold {cal.threshold:.2e}; "
          f"fires on {cal.null_fire_rate:.2%} of null batches")

    if stage == "val":
        del test
        rng = np.random.default_rng(1)
        val_batches = batches(val.X, rng)
        clean = run(dets, "clean: val subjects", val_batches)
        shifted = [run(dets, name, [f(b) for b in val_batches]) for name, f in shift_fns(ch_std, rng).items()]
        s = summarize(clean, shifted, dets)
        adopt = (s["calibrated"]["false_alarms"] <= MAX_VAL_FALSE_ALARMS
                 and s["calibrated"]["shifts_detected"] >= MIN_SHIFTS_DETECTED)
        print(f"val summary: {s}\nadopt calibrated detector: {adopt}")
        (ART / "drift_selection.json").write_text(json.dumps({
            "rule": f"adopt if val false alarms <= {MAX_VAL_FALSE_ALARMS}/{N_BATCHES} and "
                    f">= {MIN_SHIFTS_DETECTED}/6 shifts detected",
            "null": NULL, "threshold": cal.threshold, "null_fire_rate": cal.null_fire_rate,
            "val": {"clean": clean, "shifts": shifted, "summary": s}, "adopt_calibrated": adopt,
        }, indent=2))
        payload = {"summary": ref}
        if adopt:
            payload.update(null_d=null_d, alpha=np.float64(ALPHA))
        np.savez_compressed(ART / "drift_reference.npz", **payload)
        return

    sel_path = ART / "drift_selection.json"
    if not sel_path.exists():
        raise SystemExit("run --stage val first: the decision must exist before the test subjects are used")
    sel = json.loads(sel_path.read_text())
    if "test" in sel:
        print("note: the test stage already ran once; these numbers repeat it, they are not a new look")
    rng = np.random.default_rng(0)  # same draws as drift_eval.py: test, then val, then the shifts
    test_batches = batches(test.X, rng)
    batches(val.X, rng)
    clean = run(dets, "clean: test subjects", test_batches)
    shifted = [run(dets, name, [f(b) for b in test_batches]) for name, f in shift_fns(ch_std, rng).items()]
    s = summarize(clean, shifted, dets)
    print(f"test summary: {s}; adopted on val: {'calibrated' if sel['adopt_calibrated'] else 'ks_pvalue'}")
    sel["test"] = {"clean": clean, "shifts": shifted, "summary": s}
    sel_path.write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
