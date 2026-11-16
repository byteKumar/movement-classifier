"""FastAPI service: POST /predict with a batch of raw windows, scored by ONNX Runtime.

Run:  uvicorn movement.serve:create_app --factory --port 8000
Env:  MODEL_PATH (default artifacts/model.onnx), ORT_THREADS (default 1)

Request:  {"windows": [[[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] x 128] x batch]}
          raw sensor values: acc in g, gyro in rad/s, 50 Hz.
Response: {"model_version": ..., "predictions": [{"label", "class_id", "probabilities"}, ...]}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from movement.data import CHANNELS, CLASS_NAMES, WINDOW
from movement.export import onnx_logits, session, sha256, softmax

ROOT = Path(__file__).resolve().parent.parent
MAX_BATCH = 256

PREDICT_SCHEMA = {
    "requestBody": {"required": True, "content": {"application/json": {"schema": {
        "type": "object", "required": ["windows"],
        "properties": {"windows": {
            "type": "array", "minItems": 1, "maxItems": MAX_BATCH,
            "description": f"batch of windows, each {WINDOW} samples x {len(CHANNELS)} channels {CHANNELS}",
            "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        }},
    }}}},
}


def parse_windows(body: bytes) -> np.ndarray:
    """Body -> (batch, 128, 6) float32, or HTTPException with a clear message.

    Parsed by hand with json + numpy instead of a nested pydantic model: for a
    batch of 32 that is 24,576 floats, and per-element pydantic validation
    would cost more than the model itself.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(422, "body must be JSON")
    if not isinstance(payload, dict) or "windows" not in payload:
        raise HTTPException(422, 'body must be an object with a "windows" field')
    try:
        X = np.asarray(payload["windows"], dtype=np.float32)
    except (ValueError, TypeError):
        raise HTTPException(422, "windows must be a rectangular array of numbers")
    if X.ndim != 3 or X.shape[1:] != (WINDOW, len(CHANNELS)):
        raise HTTPException(422, f"windows must have shape (batch, {WINDOW}, {len(CHANNELS)}), got {X.shape}")
    if len(X) == 0:
        raise HTTPException(422, "windows is empty")
    if len(X) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} windows per request")
    if not np.isfinite(X).all():
        raise HTTPException(422, "windows contain NaN or infinite values")
    return X


def create_app(model_path: str | Path | None = None, threads: int | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("MODEL_PATH", ROOT / "artifacts" / "model.onnx"))
    threads = threads or int(os.environ.get("ORT_THREADS", "1"))
    sess = session(model_path, threads=threads)
    version = sha256(model_path)[:12]
    app = FastAPI(title="movement-classifier", version=version)

    def infer(X: np.ndarray) -> list[dict]:
        probs = softmax(onnx_logits(sess, X))
        ids = probs.argmax(axis=1)
        return [
            {"label": CLASS_NAMES[i], "class_id": int(i),
             "probabilities": {c: round(float(p), 6) for c, p in zip(CLASS_NAMES, row)}}
            for i, row in zip(ids, probs)
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": version}

    @app.post("/predict", openapi_extra=PREDICT_SCHEMA)
    async def predict(request: Request) -> dict:
        X = parse_windows(await request.body())
        # ONNX Runtime releases the GIL, so running it in the thread pool lets
        # concurrent requests use more than one core.
        predictions = await run_in_threadpool(infer, X)
        return {"model_version": version, "predictions": predictions}

    return app
