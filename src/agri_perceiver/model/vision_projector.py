"""
Vision Projector — Linear projection from vision to language space.

Two-layer MLP with GELU activation: [B, N, D_vision] → [B, N, D_language].
"""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """
    Projects visual features into the language model embedding space.

    Args:
        in_dim: Input dimension (vision encoder output, e.g., 1152 for SigLIP).
        out_dim: Output dimension (language model hidden, e.g., 3072 for Phi-3).
        dropout: Dropout rate.
    """

    def __init__(self, in_dim: int = 1152, out_dim: int = 3072, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
