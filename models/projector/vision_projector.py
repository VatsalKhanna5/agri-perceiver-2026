# models/projector/vision_projector.py

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """
    Projects SigLIP visual features into the language model dimension.

    Input:  [B, N, 1152]
    Output: [B, N, 3072]
    """

    def __init__(self, in_dim: int = 1152, out_dim: int = 3072, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )

        # Stable initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
