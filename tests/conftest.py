"""Pytest fixtures for AgriPerceiver tests."""

import pytest
import torch


@pytest.fixture
def dummy_pixel_values():
    """Dummy tiled image tensor [B=1, T=5, C=3, H=384, W=384]."""
    return torch.randn(1, 5, 3, 384, 384, dtype=torch.bfloat16)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
