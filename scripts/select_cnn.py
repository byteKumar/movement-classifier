"""Decide between CNN v1 (no augmentation) and v2 (rotation + gain augmentation)
using cross-validation over the 21 NON-test subjects only.

Why this exists: CNN v1 scored below the baseline on the test subjects, and the
test errors pointed at phone orientation (test subject 10). Changing the model
because of a test result is test-set tuning, so the change is judged here, on
subjects the final test never uses. The idea for v2 did come from looking at
test errors; RESULTS.md says so.

Decision rule (fixed before this script was first run):
    Adopt v2 if BOTH
      (a) its mean held-out macro-F1 over all (fold, seed) runs is higher than v1's, and
      (b) its per-fold mean (over seeds) beats v1's on at least 3 of the 5 folds.
    Otherwise keep v1.

Setup: 21 subjects dealt into 5 folds (seed 0). For each fold, 4 early-stopping
subjects are drawn from the other 16-17 (seed = fold index); the model trains on
the remaining 12-13. Seeds 0, 1, 2 for each config. The HGB baseline is run on
the same folds for reference only; it plays no part in the decision.

Usage: python scripts/select_cnn.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np

from movement import baseline, tracking
from movement.cnn import TrainConfig, pick_device, predict, train_cnn
from movement.data import TEST_SUBJECTS, fold_split, load_cached, subject_folds
from movement.features import extract_features
from movement.metrics import mean_std, score

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "v1": TrainConfig(),
    "v2": TrainConfig(rotate_deg=30.0, gain_jitter=0.1),
}
SEEDS = [0, 1, 2]
K = 5
N_VAL = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    w = load_cached(ROOT / "data" / "windows.npz")
    pool = sorted(set(np.unique(w.subject).tolist()) - set(TEST_SUBJECTS))
    w = w.subset(np.isin(w.subject, pool))  # the test subjects are not even loaded past here
    folds = subject_folds(pool, K, seed=0)
    print("folds:", folds, "device:", device)

    tracking.setup("cnn-selection")
    machine = tracking.machine_info()
    runs = []
    for fi, held in enumerate(folds):
        tr, va, ho = fold_split(w, held, N_VAL, seed=fi)
        F_tr, F_va, F_ho = (extract_features(p.X) for p in (tr, va, ho))
        ho_pred = baseline.fit_and_predict(F_tr, tr.y, F_va, va.y, [F_ho], seed=0)["preds"][0]
        runs.append({"config": "hgb", "fold": fi, "seed": 0, "held_out_macro_f1": score(ho.y, ho_pred)["macro_f1"]})
        print(f"fold {fi} held={held} val={sorted(set(va.subject.tolist()))} hgb F1={runs[-1]['held_out_macro_f1']:.4f}")

        for name, cfg in CONFIGS.items():
            for seed in SEEDS:
                with mlflow.start_run(run_name=f"{name}-fold{fi}-seed{seed}"):
                    model, info = train_cnn(tr, va, cfg, seed, device)
                    s = score(ho.y, predict(model, ho.X, device))
                    r = {"config": name, "fold": fi, "seed": seed, "held_out_macro_f1": s["macro_f1"],
                         "held_out_accuracy": s["accuracy"], "best_epoch": info["best_epoch"],
                         "epochs_run": info["epochs_run"], "train_seconds": info["train_seconds"]}
                    mlflow.log_params({**cfg.as_params(), "config": name, "fold": fi, "seed": seed,
                                       "device": str(device), "held_out": str(held)})
                    mlflow.set_tags({f"machine.{k}": v for k, v in machine.items()})
                    mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, float | int)
                                        and k not in ("fold", "seed")})
                runs.append(r)
                print(f"  {name} seed {seed}: F1={r['held_out_macro_f1']:.4f} "
                      f"epoch {r['best_epoch']}/{r['epochs_run']} {r['train_seconds']:.0f}s")

    def f1s(cfg, fold=None):
        return [r["held_out_macro_f1"] for r in runs if r["config"] == cfg and (fold is None or r["fold"] == fold)]

    summary = {c: mean_std(f1s(c)) for c in ["hgb", *CONFIGS]}
    fold_means = {c: [float(np.mean(f1s(c, f))) for f in range(K)] for c in ["hgb", *CONFIGS]}
    wins = sum(b > a for a, b in zip(fold_means["v1"], fold_means["v2"]))
    adopt = summary["v2"][0] > summary["v1"][0] and wins >= 3
    for c, (m, s) in summary.items():
        print(f"{c}: held-out macro-F1 {m:.4f} ± {s:.4f}  per fold {[round(x, 4) for x in fold_means[c]]}")
    print(f"v2 beats v1 on {wins} of {K} folds; mean diff {summary['v2'][0] - summary['v1'][0]:+.4f}")
    print("DECISION:", "adopt v2" if adopt else "keep v1")

    out = ROOT / "artifacts" / "cnn_selection.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"machine": machine, "device": str(device), "folds": folds,
                               "configs": {k: v.as_params() for k, v in CONFIGS.items()},
                               "summary": summary, "fold_means": fold_means, "v2_fold_wins": wins,
                               "decision": "v2" if adopt else "v1", "runs": runs}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
