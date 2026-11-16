import numpy as np
import pytest
import torch

from movement.cnn import MovementCNN, TrainConfig, augment, count_params, predict, random_rotations, train_cnn
from movement.data import WINDOW, Windows

CPU = torch.device("cpu")


def _windows(n, seed=0, separable=True):
    """Toy data: class k has a sine at a class-specific frequency on acc_x."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    t = np.arange(WINDOW) / 50
    X = rng.normal(0, 0.3, size=(n, WINDOW, 6)).astype(np.float32)
    if separable:
        X[:, :, 0] += np.sin(2 * np.pi * (1 + y[:, None]) * t)
    return Windows(X, y.astype(np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64))


def test_output_shape_and_param_count():
    m = MovementCNN()
    assert m(torch.zeros(4, WINDOW, 6)).shape == (4, 6)
    # conv 6->32, 32->64, 64->64 (k=5) + 3 batchnorms + linear 64->6
    expected = (6 * 32 * 5 + 32) + (32 * 64 * 5 + 64) + (64 * 64 * 5 + 64) + 2 * (32 + 64 + 64) + (64 * 6 + 6)
    assert count_params(m) == expected


def test_normalization_lives_in_the_model():
    X = (np.random.default_rng(0).normal(5, 2, size=(50, WINDOW, 6))).astype(np.float32)
    m = MovementCNN()
    m.set_normalization(X)
    assert torch.allclose(m.mean, torch.tensor(X.reshape(-1, 6).mean(0)), atol=1e-4)
    assert "mean" in m.state_dict() and "std" in m.state_dict()  # saved and exported with the weights


def test_learns_a_separable_toy_problem():
    cfg = TrainConfig(max_epochs=15, patience=15)
    model, info = train_cnn(_windows(600, 0), _windows(200, 1), cfg, seed=0, device=CPU)
    assert info["best_val_macro_f1"] > 0.95
    test = _windows(200, 2)
    assert (predict(model, test.X, CPU) == test.y).mean() > 0.95


def test_early_stopping_stops_and_keeps_best_epoch():
    # labels are pure noise, so val F1 stalls and training must stop early
    cfg = TrainConfig(max_epochs=50, patience=3)
    _, info = train_cnn(_windows(200, 0, separable=False), _windows(100, 1, separable=False), cfg, 0, CPU)
    assert info["epochs_run"] < 50
    assert info["epochs_run"] - info["best_epoch"] == 3
    assert info["best_val_macro_f1"] == max(h["val_macro_f1"] for h in info["history"])


def test_same_seed_same_result_on_cpu():
    cfg = TrainConfig(max_epochs=2, patience=5)
    a, _ = train_cnn(_windows(128), _windows(64, 1), cfg, seed=3, device=CPU)
    b, _ = train_cnn(_windows(128), _windows(64, 1), cfg, seed=3, device=CPU)
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(va, vb), k


def test_different_seeds_differ():
    cfg = TrainConfig(max_epochs=1, patience=5)
    a, _ = train_cnn(_windows(128), _windows(64, 1), cfg, seed=1, device=CPU)
    b, _ = train_cnn(_windows(128), _windows(64, 1), cfg, seed=2, device=CPU)
    assert not torch.equal(a.head[2].weight, b.head[2].weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_and_cpu_agree_on_forward_pass():
    m = MovementCNN().eval()
    x = torch.randn(8, WINDOW, 6)
    cpu_out = m(x)
    mps_out = m.to("mps")(x.to("mps")).cpu()
    assert torch.allclose(cpu_out, mps_out, atol=1e-4)


# ---------- augmentation ----------

def test_rotations_are_proper_and_within_angle():
    R = random_rotations(500, 30, torch.Generator().manual_seed(0))
    eye = torch.eye(3).expand(500, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(500), atol=1e-5)
    angle = torch.rad2deg(torch.arccos(((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)))
    assert angle.max() <= 30 + 1e-3 and angle.max() > 25


def test_rotation_keeps_vector_lengths_and_applies_same_rotation_to_both_sensors():
    x = torch.randn(16, WINDOW, 6)
    x[:, :, 3:] = x[:, :, :3]  # gyro == acc, so a shared rotation keeps them equal
    out = augment(x, TrainConfig(rotate_deg=30), torch.Generator().manual_seed(0))
    assert torch.allclose(out[:, :, :3].norm(dim=2), x[:, :, :3].norm(dim=2), atol=1e-5)
    assert torch.allclose(out[:, :, :3], out[:, :, 3:], atol=1e-6)
    assert not torch.allclose(out, x)


def test_gain_jitter_bounds_and_off_by_default():
    x = torch.ones(64, WINDOW, 6)
    out = augment(x, TrainConfig(gain_jitter=0.1), torch.Generator().manual_seed(0))
    assert out.min() >= 0.9 and out.max() <= 1.1 and out.std() > 0
    assert torch.equal(augment(x, TrainConfig(), torch.Generator()), x)
