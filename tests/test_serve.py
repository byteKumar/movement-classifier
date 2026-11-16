from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from movement.data import CLASS_NAMES, load_cached, split_by_subject
from movement.export import torch_logits
from movement.serve import MAX_BATCH, create_app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client(onnx_model):
    _, path = onnx_model
    return TestClient(create_app(path))


def _windows(n, seed=0):
    return np.random.default_rng(seed).normal(0, 1, size=(n, 128, 6)).astype(np.float32)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and len(r.json()["model_version"]) == 12


@pytest.mark.parametrize("n", [1, 32])
def test_predict_matches_the_pytorch_model(client, onnx_model, n):
    model, _ = onnx_model
    X = _windows(n)
    r = client.post("/predict", json={"windows": X.tolist()})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == n
    expected = torch_logits(model, X).argmax(axis=1)
    assert [p["class_id"] for p in preds] == expected.tolist()
    for p in preds:
        assert p["label"] == CLASS_NAMES[p["class_id"]]
        assert abs(sum(p["probabilities"].values()) - 1) < 1e-4


@pytest.mark.parametrize("body, status", [
    (b"not json", 422),
    (b"[1, 2]", 422),
    (b'{"x": []}', 422),
    (b'{"windows": []}', 422),
    (b'{"windows": [[1, 2, 3]]}', 422),
    (b'{"windows": [[[1, 2], [3]]]}', 422),
    (b'{"windows": "abc"}', 422),
])
def test_bad_requests_are_rejected(client, body, status):
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == status, r.text
    assert "detail" in r.json()


def test_wrong_window_length_is_rejected_with_shape_message(client):
    r = client.post("/predict", json={"windows": np.zeros((2, 100, 6)).tolist()})
    assert r.status_code == 422 and "(batch, 128, 6)" in r.json()["detail"]


def test_nan_is_rejected(client):
    X = _windows(1).tolist()
    body = '{"windows": ' + str(X).replace(str(X[0][5][2]), "NaN", 1) + "}"
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "NaN" in r.json()["detail"]


def test_too_large_batch(client):
    r = client.post("/predict", json={"windows": np.zeros((MAX_BATCH + 1, 128, 6)).tolist()})
    assert r.status_code == 413


@pytest.mark.trained_model
@pytest.mark.skipif(not (ROOT / "artifacts" / "model.onnx").exists()
                    or not (ROOT / "data" / "windows.npz").exists(), reason="needs exported model and data")
def test_real_model_through_the_api_matches_pytorch_on_test_windows():
    from movement.export import load_checkpoint

    model, _ = load_checkpoint(ROOT / "artifacts" / "cnn_final.pt")
    c = TestClient(create_app(ROOT / "artifacts" / "model.onnx"))
    _, _, test = split_by_subject(load_cached(ROOT / "data" / "windows.npz"))
    X = test.X[::25][:120]  # spread over subjects and classes
    got = [p["class_id"] for i in range(0, len(X), 32)
           for p in c.post("/predict", json={"windows": X[i:i + 32].tolist()}).json()["predictions"]]
    assert got == torch_logits(model, X).argmax(axis=1).tolist()


def test_metrics_and_drift_endpoints(onnx_model, tmp_path):
    from movement.drift import window_summary

    _, path = onnx_model
    ref = np.random.default_rng(5).normal(0, 1, size=(1000, 128, 6)).astype(np.float32)
    np.savez(tmp_path / "ref.npz", summary=window_summary(ref))
    c = TestClient(create_app(path, drift_reference=tmp_path / "ref.npz"))
    assert c.get("/drift").json()["drift"] is None
    for i in range(9):
        assert c.post("/predict", json={"windows": _windows(32, seed=i).tolist()}).status_code == 200
    m = c.get("/metrics").json()
    assert m["requests"] == 9 and m["windows"] == 288
    assert sum(m["predicted_class_counts"].values()) == 288
    d = c.get("/drift").json()
    assert d["n_windows"] == 256 and isinstance(d["drift"], bool)
