"""Train the 1D-CNN for one or more seeds, logging every run to MLflow.

Early stopping uses the validation subjects; the test subjects are scored
once, with the best-epoch weights, and never influence training.

Usage: python scripts/train_cnn.py [--seeds 0] [--device auto|mps|cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch

from movement import tracking
from movement.cnn import TrainConfig, count_params, pick_device, predict, train_cnn
from movement.data import load_cached, split_by_subject
from movement.metrics import score

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def train_one(seed: int, device: torch.device, cfg: TrainConfig, train, val, test, machine: dict) -> dict:
    with mlflow.start_run(run_name=f"cnn-seed{seed}"):
        mlflow.log_params({**cfg.as_params(), "seed": seed, "device": str(device),
                           "train_windows": len(train), "val_windows": len(val)})
        mlflow.set_tags({"model": "cnn", **{f"machine.{k}": v for k, v in machine.items()}})
        model, info = train_cnn(train, val, cfg, seed, device,
                                on_epoch=lambda e, m: mlflow.log_metrics(m, step=e))
        r = {
            "seed": seed, **{k: v for k, v in info.items() if k != "history"},
            "params": count_params(model),
            "val": score(val.y, predict(model, val.X, device)),
            "test": score(test.y, predict(model, test.X, device)),
        }
        mlflow.log_metrics({
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "train_seconds": r["train_seconds"],
            "params": r["params"], "val_macro_f1_best": r["val"]["macro_f1"],
            "test_macro_f1": r["test"]["macro_f1"], "test_accuracy": r["test"]["accuracy"],
        })
        mlflow.log_dict(r, "result.json")
        ART.mkdir(exist_ok=True)
        ckpt = ART / f"cnn_seed{seed}.pt"
        torch.save({"state_dict": model.cpu().state_dict(), "config": cfg.as_params(), "seed": seed}, ckpt)
        mlflow.log_artifact(str(ckpt))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    train, val, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))
    device, cfg = pick_device(args.device), TrainConfig()
    tracking.setup("cnn")
    machine = tracking.machine_info()
    for seed in args.seeds:
        r = train_one(seed, device, cfg, train, val, test, machine)
        print(f"seed {seed} [{r['device']}]: params={r['params']} best_epoch={r['best_epoch']}/{r['epochs_run']} "
              f"val F1={r['val']['macro_f1']:.4f} test F1={r['test']['macro_f1']:.4f} "
              f"acc={r['test']['accuracy']:.4f} train={r['train_seconds']:.0f}s")
        (ART / f"cnn_seed{seed}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
