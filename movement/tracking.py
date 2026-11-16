"""MLflow setup and machine info, so every run records where it ran."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()


def setup(experiment: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"{ARTIFACT_ROOT}/{experiment}")
    mlflow.set_experiment(experiment)


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_info() -> dict:
    info = {
        "chip": _sysctl("machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor(),
        "ram_gb": round(int(_sysctl("hw.memsize")) / 2**30) if sys.platform == "darwin" else None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info
