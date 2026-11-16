"""Final evaluation of the baseline and the CNN (v2 defaults).

Part 1, fixed split (17 train / 4 val / 9 test subjects), seeds 0-4:
  macro-F1 and accuracy on the test subjects as mean ± std over seeds,
  per-class F1, confusion matrices summed over seeds, the hardest class pair,
  and per-subject test accuracy.
Part 2, leave-subjects-out cross-validation over all 30 subjects:
  5 folds of 6 held-out subjects; in each fold 4 of the other subjects are
  used for early stopping (HGB: choosing the number of rounds). Seeds 0-4.

The model served later is the seed-0 CNN from part 1. That choice is fixed
here, before any result exists, so the served model is not the luckiest seed.

Usage: python scripts/evaluate.py [--device mps] [--part all|seeds|cv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch

from movement import baseline, tracking
from movement.cnn import TrainConfig, count_params, pick_device, predict, train_cnn
from movement.data import CLASS_NAMES, fold_split, load_cached, split_by_subject, subject_folds
from movement.evaluation import hardest_pair, plot_confusion
from movement.features import extract_features
from movement.metrics import mean_std, score

ROOT = Path(__file__).resolve().parent.parent
ART, DOCS = ROOT / "artifacts", ROOT / "docs"
SEEDS = [0, 1, 2, 3, 4]
SERVED_SEED = 0
K, N_VAL = 5, 4


def log_run(name: str, params: dict, metrics: dict, machine: dict) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.set_tags({f"machine.{k}": v for k, v in machine.items()})
        mlflow.log_metrics(metrics)


def summarize(results: list[dict]) -> dict:
    f1 = mean_std([r["macro_f1"] for r in results])
    acc = mean_std([r["accuracy"] for r in results])
    conf = np.sum([r["confusion"] for r in results], axis=0)
    per_class = {c: mean_std([r["per_class_f1"][c] for r in results]) for c in CLASS_NAMES}
    return {"macro_f1": f1, "accuracy": acc, "per_class_f1": per_class,
            "confusion_sum": conf.tolist(), "hardest_pair": hardest_pair(conf), "n": len(results)}


def part_seeds(w, device, cfg, machine) -> dict:
    train, val, test = split_by_subject(w)
    F_tr, F_va, F_te = (extract_features(p.X) for p in (train, val, test))
    out = {"baseline": [], "cnn": []}
    subj_acc = {int(s): [] for s in np.unique(test.subject)}
    for seed in SEEDS:
        b = baseline.fit_and_predict(F_tr, train.y, F_va, val.y, [F_te], seed)
        sb = score(test.y, b["preds"][0])
        out["baseline"].append(sb)
        log_run(f"seeds-hgb-{seed}", {**baseline.PARAMS, "seed": seed, "part": "seeds", "model": "hgb"},
                {"test_macro_f1": sb["macro_f1"], "test_accuracy": sb["accuracy"], "best_iter": b["best_iter"]},
                machine)

        model, info = train_cnn(train, val, cfg, seed, device)
        pred = predict(model, test.X, device)
        sc = score(test.y, pred)
        out["cnn"].append(sc)
        for s in subj_acc:
            k = test.subject == s
            subj_acc[s].append(float((pred[k] == test.y[k]).mean()))
        log_run(f"seeds-cnn-{seed}", {**cfg.as_params(), "seed": seed, "part": "seeds", "model": "cnn",
                                       "device": str(device)},
                {"test_macro_f1": sc["macro_f1"], "test_accuracy": sc["accuracy"],
                 "best_epoch": info["best_epoch"], "train_seconds": info["train_seconds"]}, machine)
        print(f"seed {seed}: HGB F1={sb['macro_f1']:.4f}  CNN F1={sc['macro_f1']:.4f} acc={sc['accuracy']:.4f} "
              f"(epoch {info['best_epoch']}/{info['epochs_run']}, {info['train_seconds']:.0f}s {device})")
        if seed == SERVED_SEED:
            ART.mkdir(exist_ok=True)
            torch.save({"state_dict": model.cpu().state_dict(), "config": cfg.as_params(), "seed": seed,
                        "class_names": CLASS_NAMES, "params": count_params(model),
                        "test_macro_f1": sc["macro_f1"]}, ART / "cnn_final.pt")
            model.to(device)

    res = {m: summarize(v) for m, v in out.items()}
    res["cnn"]["per_subject_accuracy"] = {s: mean_std(v) for s, v in subj_acc.items()}
    res["runs"] = out
    DOCS.mkdir(exist_ok=True)
    for m, title in [("cnn", "1D-CNN"), ("baseline", "Features + HistGradientBoosting")]:
        plot_confusion(res[m]["confusion_sum"], f"{title}: test subjects, summed over {len(SEEDS)} seeds",
                       DOCS / f"confusion_{m}.png")
    return res


def part_cv(w, device, cfg, machine) -> dict:
    folds = subject_folds(np.unique(w.subject), K, seed=0)
    per_fold = []
    for fi, held in enumerate(folds):
        tr, va, ho = fold_split(w, held, N_VAL, seed=fi)
        F_tr, F_va, F_ho = (extract_features(p.X) for p in (tr, va, ho))
        b = baseline.fit_and_predict(F_tr, tr.y, F_va, va.y, [F_ho], seed=0)
        hgb_f1 = score(ho.y, b["preds"][0])["macro_f1"]
        log_run(f"cv-hgb-fold{fi}", {**baseline.PARAMS, "fold": fi, "part": "cv", "model": "hgb",
                                      "held_out": str(held)}, {"held_out_macro_f1": hgb_f1}, machine)
        cnn = []
        for seed in SEEDS:
            model, info = train_cnn(tr, va, cfg, seed, device)
            s = score(ho.y, predict(model, ho.X, device))
            cnn.append(s["macro_f1"])
            log_run(f"cv-cnn-fold{fi}-seed{seed}", {**cfg.as_params(), "fold": fi, "seed": seed, "part": "cv",
                                                     "model": "cnn", "device": str(device), "held_out": str(held)},
                    {"held_out_macro_f1": s["macro_f1"], "held_out_accuracy": s["accuracy"],
                     "best_epoch": info["best_epoch"]}, machine)
        per_fold.append({"fold": fi, "held_out": held, "hgb_macro_f1": hgb_f1, "cnn_macro_f1_seeds": cnn,
                         "cnn_macro_f1_mean": float(np.mean(cnn))})
        print(f"fold {fi} held={held}: HGB F1={hgb_f1:.4f}  CNN F1 seeds={[round(x, 4) for x in cnn]} "
              f"mean={np.mean(cnn):.4f}")
    return {
        "folds": folds, "per_fold": per_fold,
        "cnn_macro_f1_over_folds": mean_std([f["cnn_macro_f1_mean"] for f in per_fold]),
        "cnn_macro_f1_all_runs": mean_std([x for f in per_fold for x in f["cnn_macro_f1_seeds"]]),
        "hgb_macro_f1_over_folds": mean_std([f["hgb_macro_f1"] for f in per_fold]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--part", default="all", choices=["all", "seeds", "cv"])
    args = ap.parse_args()
    device, cfg = pick_device(args.device), TrainConfig()
    w = load_cached(ROOT / "data" / "windows.npz")
    tracking.setup("evaluation")
    machine = tracking.machine_info()

    path = ART / "evaluation.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result.update({"machine": machine, "device": str(device), "cnn_config": cfg.as_params(),
                   "baseline_params": baseline.PARAMS})
    if args.part in ("all", "seeds"):
        result["seeds"] = part_seeds(w, device, cfg, machine)
        for m in ("baseline", "cnn"):
            r = result["seeds"][m]
            hp = r["hardest_pair"]
            print(f"{m}: test macro-F1 {r['macro_f1'][0]:.4f} ± {r['macro_f1'][1]:.4f}  "
                  f"accuracy {r['accuracy'][0]:.4f} ± {r['accuracy'][1]:.4f}  hardest: {hp['classes']} "
                  f"({hp['a_as_b']} / {hp['b_as_a']} windows)")
    if args.part in ("all", "cv"):
        result["cv"] = part_cv(w, device, cfg, machine)
        cv = result["cv"]
        print(f"CV over 30 subjects: CNN {cv['cnn_macro_f1_over_folds'][0]:.4f} ± "
              f"{cv['cnn_macro_f1_over_folds'][1]:.4f} (fold means)  HGB {cv['hgb_macro_f1_over_folds'][0]:.4f} ± "
              f"{cv['hgb_macro_f1_over_folds'][1]:.4f}")
    ART.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
