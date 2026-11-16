"""1D-CNN over raw windows, and its training loop with early stopping."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from movement.data import CHANNELS, CLASS_NAMES, Windows
from movement.metrics import score


class MovementCNN(nn.Module):
    """Input: raw windows (batch, 128, 6) exactly as the pipeline produces them.

    The per-channel mean/std from the training set live inside the model as
    buffers, so the exported ONNX file takes raw sensor values and nobody has
    to remember to normalize before calling it.
    """

    def __init__(self, channels: int = 32, kernel: int = 5, dropout: float = 0.3):
        super().__init__()
        c_in, n_cls = len(CHANNELS), len(CLASS_NAMES)
        self.register_buffer("mean", torch.zeros(c_in))
        self.register_buffer("std", torch.ones(c_in))

        def block(cin, cout):
            return [nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.BatchNorm1d(cout), nn.ReLU()]

        self.features = nn.Sequential(
            *block(c_in, channels), nn.MaxPool1d(2),           # 128 -> 64 samples
            *block(channels, 2 * channels), nn.MaxPool1d(2),   # 64 -> 32
            *block(2 * channels, 2 * channels),
            nn.AdaptiveAvgPool1d(1),                           # average over time
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * channels, n_cls))

    def set_normalization(self, X_train: np.ndarray) -> None:
        self.mean.copy_(torch.from_numpy(X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)))
        self.std.copy_(torch.from_numpy(X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 128, 6) -> logits (B, 6)
        x = (x - self.mean) / self.std
        return self.head(self.features(x.transpose(1, 2)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class TrainConfig:
    channels: int = 32
    kernel: int = 5
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10  # epochs without val macro-F1 improvement before stopping
    # Training-time augmentation (0 = off). Applied to each training batch only.
    rotate_deg: float = 0.0  # max angle of a random 3D rotation of the phone
    gain_jitter: float = 0.0  # each channel scaled by U(1 - g, 1 + g)

    def as_params(self) -> dict:
        return asdict(self)


def random_rotations(n: int, max_deg: float, gen: torch.Generator) -> torch.Tensor:
    """n rotation matrices about random axes by angles uniform in [-max_deg, max_deg]."""
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=1, keepdim=True)
    angle = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(max_deg)
    x, y, z = axis.unbind(1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).view(n, 3, 3)
    s, c = torch.sin(angle).view(n, 1, 1), torch.cos(angle).view(n, 1, 1)
    return torch.eye(3).expand(n, 3, 3) + s * K + (1 - c) * (K @ K)  # Rodrigues' formula


def augment(xb: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> torch.Tensor:
    """Simulate a phone worn at a different angle / with different sensor gain.

    The accelerometer and gyroscope sit on the same board, so both get the
    same rotation. Rotation angles are kept small: large ones would erase
    the gravity direction that tells sitting, standing and lying apart.
    """
    if cfg.rotate_deg > 0:
        R = random_rotations(len(xb), cfg.rotate_deg, gen)  # (B, 3, 3)
        acc = xb[:, :, 0:3] @ R.transpose(1, 2)
        gyro = xb[:, :, 3:6] @ R.transpose(1, 2)
        xb = torch.cat([acc, gyro], dim=2)
    if cfg.gain_jitter > 0:
        g = 1 + (torch.rand(len(xb), 1, xb.shape[2], generator=gen) * 2 - 1) * cfg.gain_jitter
        xb = xb * g
    return xb


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def train_cnn(train: Windows, val: Windows, cfg: TrainConfig, seed: int, device: torch.device,
              on_epoch=None) -> tuple[MovementCNN, dict]:
    """Train on `train`, early-stop on `val` macro-F1, return the best-epoch model.

    `on_epoch(epoch, metrics_dict)` is called after every epoch (used for MLflow).
    """
    set_seed(seed)
    model = MovementCNN(cfg.channels, cfg.kernel, cfg.dropout)
    model.set_normalization(train.X)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    X_tr = torch.from_numpy(train.X)
    y_tr = torch.from_numpy(train.y)
    gen = torch.Generator().manual_seed(seed)  # shuffling order depends only on the seed

    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = augment(X_tr[idx], cfg, gen)  # done on CPU so the random draws are reproducible
            xb, yb = xb.to(device), y_tr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_f1 = score(val.y, predict(model, val.X, device))["macro_f1"]
        row = {"train_loss": total / len(perm), "val_macro_f1": val_f1}
        history.append(row)
        if on_epoch:
            on_epoch(epoch, row)
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state = val_f1, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_macro_f1": best_f1,
            "train_seconds": time.perf_counter() - t0, "device": str(device), "history": history}
    return model, info
