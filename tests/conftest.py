import numpy as np
import pytest
import torch

from movement.cnn import MovementCNN
from movement.export import export_onnx


@pytest.fixture(scope="session")
def onnx_model(tmp_path_factory):
    """A small exported model with random (but fixed) weights and real-looking normalization."""
    torch.manual_seed(0)
    model = MovementCNN().eval()
    X = np.random.default_rng(0).normal([0, 0.8, 0.3, 0, 0, 0], 0.5, size=(64, 128, 6)).astype(np.float32)
    model.set_normalization(X)
    path = tmp_path_factory.mktemp("model") / "model.onnx"
    export_onnx(model, path)
    return model, path
