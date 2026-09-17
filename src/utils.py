"""Reproducibility, device selection, logging and timing helpers."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True) -> str:
    try:
        import torch

        return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_logger(name: str = "gold_forecast", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Measure wall-clock time; the yielded dict receives ``seconds`` on exit."""
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - start
        if logger is not None:
            logger.info("%s finished in %.2fs", label, info["seconds"])
