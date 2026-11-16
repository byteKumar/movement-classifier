"""Export the CNN to ONNX and compare ONNX Runtime with PyTorch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from movement.cnn import MovementCNN
from movement.data import CHANNELS, WINDOW

INPUT, OUTPUT = "windows", "logits"
PARITY_ATOL = 1e-4  # max allowed |onnx - torch| on logits


def load_checkpoint(path: str | Path) -> tuple[MovementCNN, dict]:
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    model = MovementCNN(cfg["channels"], cfg["kernel"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def export_onnx(model: MovementCNN, path: str | Path) -> None:
    model = model.eval()
    example = torch.zeros(2, WINDOW, len(CHANNELS))
    torch.onnx.export(
        model, (example,), str(path),
        input_names=[INPUT], output_names=[OUTPUT],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
        dynamo=True, external_data=False,
    )


def session(path: str | Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def onnx_logits(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    return sess.run([OUTPUT], {INPUT: np.ascontiguousarray(X, dtype=np.float32)})[0]


@torch.no_grad()
def torch_logits(model: MovementCNN, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model = model.to(device).eval()
    out = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)).cpu().numpy()
    model.to("cpu")
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def parity(model: MovementCNN, sess: ort.InferenceSession, X: np.ndarray, device: str = "cpu",
           batch: int = 256) -> dict:
    """Max abs difference between ONNX Runtime and PyTorch over all of X."""
    a = np.concatenate([onnx_logits(sess, X[i : i + batch]) for i in range(0, len(X), batch)])
    b = np.concatenate([torch_logits(model, X[i : i + batch], device) for i in range(0, len(X), batch)])
    return {
        "n": len(X), "torch_device": device,
        "max_abs_diff_logits": float(np.abs(a - b).max()),
        "max_abs_diff_probs": float(np.abs(softmax(a) - softmax(b)).max()),
        "argmax_agreement": float((a.argmax(1) == b.argmax(1)).mean()),
    }


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
