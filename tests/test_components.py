"""Unit tests for model components (CPU-only, no backbones needed)."""

import torch
import pytest

from agri_perceiver.model.perceiver_resampler import PerceiverResampler
from agri_perceiver.model.vision_projector import VisionProjector
from agri_perceiver.model.tile_embeddings import TileEmbeddings


class TestPerceiverResampler:
    def test_output_shape(self):
        pr = PerceiverResampler(dim=256, num_latents=16, depth=1, heads=4)
        x = torch.randn(2, 100, 256)
        out = pr(x)
        assert out.shape == (2, 16, 256)

    def test_compression_ratio(self):
        pr = PerceiverResampler(dim=256, num_latents=8, depth=2, heads=4)
        x = torch.randn(1, 200, 256)
        out = pr(x)
        assert out.shape[1] == 8  # 200 -> 8 = 25x compression


class TestVisionProjector:
    def test_output_shape(self):
        proj = VisionProjector(in_dim=128, out_dim=256)
        x = torch.randn(2, 10, 128)
        out = proj(x)
        assert out.shape == (2, 10, 256)


class TestTileEmbeddings:
    def test_output_shape(self):
        te = TileEmbeddings(dim=128, num_tiles=5)
        x = torch.randn(2, 5, 10, 128)
        out = te(x)
        assert out.shape == (2, 5, 10, 128)

    def test_tile_count_mismatch(self):
        te = TileEmbeddings(dim=128, num_tiles=5)
        x = torch.randn(2, 3, 10, 128)
        with pytest.raises(AssertionError):
            te(x)
