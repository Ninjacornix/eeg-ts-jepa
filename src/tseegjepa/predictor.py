"""JEPA predictor: forecast target-token latents from the context view.

Narrow transformer.  Input sequence = context tokens (their context-encoder
representations) + mask tokens placed at every target position.  Both carry
position/identity embeddings so the predictor knows *where* it is forecasting.
Full self-attention (padding-masked); read-out at target positions is projected
back to the encoder's latent dimension for the latent-space loss.
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

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        # key_pad: (B, M) True = valid key
        B, M, D = x.shape
        qkv = self.qkv(x).reshape(B, M, 3, self.heads, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = key_pad.view(B, 1, 1, M).expand(B, self.heads, M, M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, M, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _SelfAttn(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, key_pad):
        x = x + self.attn(self.n1(x), key_pad)
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
            x = blk(x, key_pad)
        x = self.norm(x)
        return self.out_proj(x)
