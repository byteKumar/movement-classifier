import numpy as np
import pytest

from movement.export import PARITY_ATOL, onnx_logits, parity, session, torch_logits


@pytest.mark.parametrize("batch", [1, 7, 32, 300])
def test_onnx_matches_torch_for_any_batch_size(onnx_model, batch):
    model, path = onnx_model
    X = np.random.default_rng(batch).normal(0, 1, size=(batch, 128, 6)).astype(np.float32)
    out = onnx_logits(session(path), X)
    assert out.shape == (batch, 6)
    assert np.abs(out - torch_logits(model, X)).max() < PARITY_ATOL


def test_parity_report(onnx_model):
    model, path = onnx_model
    X = np.random.default_rng(1).normal(0, 1, size=(100, 128, 6)).astype(np.float32)
    r = parity(model, session(path), X, batch=33)  # batch not dividing n on purpose
    assert r["n"] == 100 and r["argmax_agreement"] == 1.0
    assert r["max_abs_diff_logits"] < PARITY_ATOL


def test_normalization_is_inside_the_onnx_graph(onnx_model):
    # Shifting the input by the training mean must change the output: the graph
    # normalizes raw values itself rather than expecting pre-normalized input.
    model, path = onnx_model
    sess = session(path)
    raw = np.zeros((1, 128, 6), np.float32)
    shifted = raw + model.mean.numpy()
    assert not np.allclose(onnx_logits(sess, raw), onnx_logits(sess, shifted))
    assert np.abs(onnx_logits(sess, shifted) - torch_logits(model, shifted)).max() < PARITY_ATOL
