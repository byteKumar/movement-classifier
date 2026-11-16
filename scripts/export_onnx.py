"""Export artifacts/cnn_final.pt to artifacts/model.onnx and check parity.

Parity is measured on every test-subject window, against PyTorch on CPU (the
reference) and, when available, PyTorch on MPS.

Usage: python scripts/export_onnx.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from movement.cnn import count_params
from movement.data import CLASS_NAMES, load_cached, split_by_subject
from movement.export import PARITY_ATOL, export_onnx, load_checkpoint, parity, session, sha256

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def main() -> None:
    model, ck = load_checkpoint(ART / "cnn_final.pt")
    onnx_path = ART / "model.onnx"
    export_onnx(model, onnx_path)
    sess = session(onnx_path)
    _, _, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))

    results = [parity(model, sess, test.X, "cpu")]
    if torch.backends.mps.is_available():
        results.append(parity(model, sess, test.X, "mps"))
    for r in results:
        print(f"ONNX Runtime vs PyTorch[{r['torch_device']}] on {r['n']} test windows: "
              f"max |logit diff| = {r['max_abs_diff_logits']:.3e}, max |prob diff| = {r['max_abs_diff_probs']:.3e}, "
              f"argmax agreement = {r['argmax_agreement']:.4%}")
    if results[0]["max_abs_diff_logits"] > PARITY_ATOL:
        raise SystemExit(f"parity check FAILED (tolerance {PARITY_ATOL})")

    size = onnx_path.stat().st_size
    meta = {
        "model_version": sha256(onnx_path)[:12], "source_checkpoint_seed": ck["seed"],
        "class_names": CLASS_NAMES, "params": count_params(model),
        "onnx_bytes": size, "onnx_mb": size / 1e6,
        "pt_state_dict_bytes": (ART / "cnn_final.pt").stat().st_size,
        "parity": results, "parity_atol": PARITY_ATOL,
    }
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"model.onnx: {size} bytes ({size / 1e6:.3f} MB), params={meta['params']}, version={meta['model_version']}")


if __name__ == "__main__":
    main()
