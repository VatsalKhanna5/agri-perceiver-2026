"""
Perceiver Resampler — Cross-attention based token compression.

Compresses T*P visual tokens (e.g., 3,645) into a fixed set of N learned latents (e.g., 128).
Each PerceiverBlock: CrossAttn(latents ← visual) → SelfAttn(latents) → FFN(GLU).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.scale * x * torch.rsqrt(norm + self.eps)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key-value pairs."""

    def __init__(self, dim: int, heads: int = 24):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, D = q.shape
        Nk = kv.size(1)
        H = self.heads

        q = self.to_q(q).view(B, Nq, H, D // H).transpose(1, 2)
        k = self.to_k(kv).view(B, Nk, H, D // H).transpose(1, 2)
        v = self.to_v(kv).view(B, Nk, H, D // H).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, D)
        return self.proj(out)


class PerceiverBlock(nn.Module):
    """Single Perceiver block: cross-attention + self-attention + FFN with GLU."""

    def __init__(self, dim: int, heads: int = 24, mlp_ratio: int = 4):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads)

        self.norm2 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm3 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio * 2),
            nn.GLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, latents: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross_attn(self.norm1(latents), visual_tokens)
        normed = self.norm2(latents)
        latents = latents + self.self_attn(normed, normed, normed)[0]
        latents = latents + self.mlp(self.norm3(latents))
        return latents


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for visual token compression.

    Uses learned latent queries that cross-attend to visual tokens,
    compressing variable-length visual sequences into a fixed-size representation.

    Args:
        dim: Feature dimension (must match language model hidden dim).
        num_latents: Number of output latent vectors (compression target).
        depth: Number of stacked PerceiverBlocks.
        heads: Number of attention heads.
    """

    def __init__(self, dim: int = 3072, num_latents: int = 128, depth: int = 2, heads: int = 24):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([PerceiverBlock(dim, heads) for _ in range(depth)])

        self.norm = RMSNorm(dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N_visual, dim] projected visual features.

        Returns:
            [B, num_latents, dim] compressed latent representation.
        """
        B = visual_tokens.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            latents = layer(latents, visual_tokens)

        return self.norm(latents)
