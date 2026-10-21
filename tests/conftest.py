"""Shared pytest fixtures and tolerances."""

from __future__ import annotations

import pytest
import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if cuda_available() else "cpu")


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    if not cuda_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda")


def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
