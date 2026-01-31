# models/spatial/tile_embeddings.py

import torch
import torch.nn as nn


class TileEmbeddings(nn.Module):
    """
    Adds learned spatial embeddings to each tile's patch tokens.

    Input:  x -> [B, 5, 729, D]
    Output: x -> [B, 5, 729, D] (spatially grounded)
    """

    def __init__(self, dim: int = 1152, num_tiles: int = 5, dropout: float = 0.0):
        super().__init__()

        self.num_tiles = num_tiles
        self.dim = dim

        # Learned embedding per tile
        self.tile_embeddings = nn.Parameter(
            torch.randn(num_tiles, dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape

        assert T == self.num_tiles, f"Expected {self.num_tiles} tiles, got {T}"
        assert D == self.dim, f"Expected feature dim {self.dim}, got {D}"

        # [T, D] → [1, T, 1, D]
        tile_embed = self.tile_embeddings.unsqueeze(0).unsqueeze(2)

        # Broadcast add
        x = x + tile_embed

        return self.dropout(x)
