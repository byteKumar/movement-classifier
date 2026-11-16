"""Feature baseline: hand-crafted features + HistGradientBoosting, over 5 seeds.

The number of boosting rounds is picked on the validation subjects (the same
role early stopping plays for the CNN); the test subjects are scored once, at
that round, and never used for any choice.

Usage: python scripts/train_baseline.py [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from movement import tracking
from movement.baseline import PARAMS, fit_and_predict
from movement.data import load_cached, split_by_subject
from movement.features import extract_features, feature_names
from movement.metrics import mean_std, score

ROOT = Path(__file__).resolve().parent.parent


def run(seed: int, F_tr, y_tr, F_va, y_va, F_te, y_te) -> dict:
    out = fit_and_predict(F_tr, y_tr, F_va, y_va, [F_te], seed)
    return {"seed": seed, "best_iter": out["best_iter"], "fit_seconds": out["fit_seconds"],
            "val": score(y_va, out["val_pred"]), "test": score(y_te, out["preds"][0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    train, val, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))
    F_tr, F_va, F_te = (extract_features(p.X) for p in (train, val, test))
    print(f"features: {F_tr.shape[1]}  train/val/test windows: {len(train)}/{len(val)}/{len(test)}")

    tracking.setup("baseline-hgb")
    machine = tracking.machine_info()
    results = []
    for seed in args.seeds:
        with mlflow.start_run(run_name=f"hgb-seed{seed}"):
            r = run(seed, F_tr, train.y, F_va, val.y, F_te, test.y)
            mlflow.log_params({**PARAMS, "seed": seed, "n_features": F_tr.shape[1], "device": "cpu"})
            mlflow.set_tags({"model": "hgb", **{f"machine.{k}": v for k, v in machine.items()}})
            mlflow.log_metrics({
                "best_iter": r["best_iter"], "fit_seconds": r["fit_seconds"],
                "val_macro_f1": r["val"]["macro_f1"], "val_accuracy": r["val"]["accuracy"],
                "test_macro_f1": r["test"]["macro_f1"], "test_accuracy": r["test"]["accuracy"],
            })
            mlflow.log_dict(r, "result.json")
        results.append(r)
        print(f"seed {seed}: best_iter={r['best_iter']:3d} val F1={r['val']['macro_f1']:.4f} "
              f"test F1={r['test']['macro_f1']:.4f} acc={r['test']['accuracy']:.4f} fit={r['fit_seconds']:.1f}s")

    f1_m, f1_s = mean_std([r["test"]["macro_f1"] for r in results])
    acc_m, acc_s = mean_std([r["test"]["accuracy"] for r in results])
    print(f"TEST macro-F1 {f1_m:.4f} ± {f1_s:.4f}   accuracy {acc_m:.4f} ± {acc_s:.4f}  (n={len(results)} seeds)")
    out = ROOT / "artifacts" / "baseline_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "machine": machine, "params": PARAMS, "feature_names": feature_names(),
        "test_macro_f1_mean": f1_m, "test_macro_f1_std": f1_s,
        "test_accuracy_mean": acc_m, "test_accuracy_std": acc_s, "runs": results,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
