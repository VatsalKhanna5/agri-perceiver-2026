"""
Tile Embeddings — Learned spatial position encoding per tile.

Adds a distinct learnable embedding to each tile's patch tokens, enabling
the model to distinguish spatial origin (top-left, top-right, etc.).
"""

import torch
import torch.nn as nn


class TileEmbeddings(nn.Module):
    """
    Spatial grounding via per-tile learned embeddings.

    Args:
        dim: Feature dimension (must match vision encoder output dim).
        num_tiles: Number of AnyRes tiles (default: 5 = 4 quadrants + 1 global).
        dropout: Dropout rate.
    """

    def __init__(self, dim: int = 1152, num_tiles: int = 5, dropout: float = 0.0):
        super().__init__()

        self.num_tiles = num_tiles
        self.dim = dim

        self.tile_embeddings = nn.Parameter(torch.randn(num_tiles, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, P, D] patch tokens per tile.

        Returns:
            [B, T, P, D] spatially grounded patch tokens.
        """
        B, T, P, D = x.shape
        assert T == self.num_tiles, f"Expected {self.num_tiles} tiles, got {T}"
        assert D == self.dim, f"Expected dim {self.dim}, got {D}"

        tile_embed = self.tile_embeddings.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
        return self.dropout(x + tile_embed)
