"""Utility functions for dflash: logging, device management, and tensor helpers."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch

logger = logging.getLogger(__name__)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(level)
    return log


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve a torch device string, defaulting to CUDA if available."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count the total (or trainable) parameters of a model."""
    params = (
        model.parameters()
        if not trainable_only
        else filter(lambda p: p.requires_grad, model.parameters())
    )
    return sum(p.numel() for p in params)


@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """Context manager that logs wall-clock time for a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("%s took %.4f s", label, elapsed)


def move_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Recursively move all tensors in a dict to *device*."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch (and optionally numpy)."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def human_size(num_bytes: int, suffix: str = "B") -> str:
    """Convert a byte count to a human-readable string (e.g. 1.4 GiB)."""
    for unit in ("  ", "Ki", "Mi", "Gi", "Ti", "Pi"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f} {unit.strip()}{suffix}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.1f} Ei{suffix}"


def gpu_memory_summary(device: Optional[torch.device] = None) -> str:
    """Return a short string summarising current GPU memory usage."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    dev = device or torch.device("cuda")
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    return (
        f"allocated={human_size(allocated)}, "
        f"reserved={human_size(reserved)}"
    )


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from an environment variable.

    Truthy values: '1', 'true', 'yes' (case-insensitive).
    """
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes")
