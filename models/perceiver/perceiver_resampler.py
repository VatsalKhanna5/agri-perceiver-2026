# models/perceiver/perceiver_resampler.py

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=24):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        B, Nq, D = q.shape
        Nk = kv.size(1)
        H = self.heads

        q = self.to_q(q).view(B, Nq, H, D//H).transpose(1,2)
        k = self.to_k(kv).view(B, Nk, H, D//H).transpose(1,2)
        v = self.to_v(kv).view(B, Nk, H, D//H).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1,2).contiguous().view(B, Nq, D)
        return self.proj(out)

class PerceiverBlock(nn.Module):
    def __init__(self, dim, heads=24, mlp_ratio=4):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads)

        self.norm2 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm3 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio * 2),
            nn.GLU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, latents, visual_tokens):
        latents = latents + self.cross_attn(self.norm1(latents), visual_tokens)
        latents = latents + self.self_attn(self.norm2(latents),
                                           self.norm2(latents),
                                           self.norm2(latents))[0]
        latents = latents + self.mlp(self.norm3(latents))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, dim=3072, num_latents=128, depth=2, heads=24):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([
            PerceiverBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = RMSNorm(dim)

    def forward(self, visual_tokens):
        B = visual_tokens.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            latents = layer(latents, visual_tokens)

        return self.norm(latents)
