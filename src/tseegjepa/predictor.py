"""JEPA predictor: forecast target-token latents from the context view.

Narrow transformer.  Input sequence = context tokens (their context-encoder
representations) + mask tokens placed at every target position.  Both carry
position/identity embeddings so the predictor knows *where* it is forecasting.
Factorized temporal/spatial attention avoids a dense token-grid attention
matrix; read-out at target positions is projected back to encoder dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def _attend(self, x: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        G, M, D = x.shape
        original = key_pad
        key_pad = key_pad.clone()
        empty = ~key_pad.any(-1)
        if empty.any():
            key_pad[empty, 0] = True
        qkv = self.qkv(x).reshape(G, M, 3, self.heads, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = key_pad.view(G, 1, 1, M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.proj(out.transpose(1, 2).reshape(G, M, D))
        return out * original.unsqueeze(-1)

    def forward(
        self, x: torch.Tensor, key_pad: torch.Tensor, grid_shape: tuple[int, int]
    ) -> torch.Tensor:
        B, _, D = x.shape
        C, T = grid_shape
        temporal = x.reshape(B, C, T, D).reshape(B * C, T, D)
        temporal_m = key_pad.reshape(B, C, T).reshape(B * C, T)
        temporal = self._attend(temporal, temporal_m).reshape(B, C, T, D)

        spatial = x.reshape(B, C, T, D).permute(0, 2, 1, 3).reshape(B * T, C, D)
        spatial_m = key_pad.reshape(B, C, T).permute(0, 2, 1).reshape(B * T, C)
        spatial = self._attend(spatial, spatial_m)
        spatial = spatial.reshape(B, T, C, D).permute(0, 2, 1, 3)
        return (temporal + spatial).reshape(B, C * T, D) * (2.0 ** -0.5)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _SelfAttn(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, key_pad, grid_shape):
        x = x + self.attn(self.n1(x), key_pad, grid_shape)
        x = x + self.mlp(self.n2(x))
        return x


class Predictor(nn.Module):
    def __init__(self, enc_dim: int, pred_dim: int, depth: int, heads: int):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, pred_dim)
        self.pos_proj = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([_Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, enc_dim)

    def forward(
        self,
        context_repr: torch.Tensor,   # (B, N, enc_dim) context-encoder output
        pos_emb: torch.Tensor,        # (B, N, enc_dim) position/identity embeddings
        context_mask: torch.Tensor,   # (B, N) True = context token
        target_mask: torch.Tensor,    # (B, N) True = target token
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Return predicted target latents (B, N, enc_dim) (only target rows used)."""
        B, N, _ = context_repr.shape
        pos = self.pos_proj(pos_emb)
        ctx = self.in_proj(context_repr) + pos
        tgt = self.mask_token.view(1, 1, -1) + pos

        cmask = context_mask.unsqueeze(-1)
        tmask = target_mask.unsqueeze(-1)
        x = torch.where(cmask, ctx, torch.zeros_like(ctx))
        x = torch.where(tmask, tgt, x)

        key_pad = context_mask | target_mask          # valid tokens for attention
        for blk in self.blocks:
            x = blk(x, key_pad, grid_shape)
        x = self.norm(x)
        return self.out_proj(x)
