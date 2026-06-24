"""Factorized temporal and electrode-graph attention.

Attention is computed as ``C`` temporal sequences of length ``T`` plus ``T``
spatial graphs of size ``C``. It therefore avoids materializing a dense
``(C*T) x (C*T)`` mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchMHA(nn.Module):
    """MHA over grouped temporal sequences or grouped spatial graphs."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def _attend(
        self,
        x: torch.Tensor,          # (G,L,D)
        key_valid: torch.Tensor,  # (G,L)
        pair_mask: torch.Tensor | None = None,  # (1|G,1,L,L)
    ) -> torch.Tensor:
        G, L, D = x.shape
        original_valid = key_valid
        key_valid = key_valid.clone()
        empty = ~key_valid.any(-1)
        if empty.any():
            key_valid[empty, 0] = True
        qkv = self.qkv(x).reshape(G, L, 3, self.heads, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = key_valid[:, None, None, :]
        if pair_mask is not None:
            attn_mask = attn_mask & pair_mask
        if attn_mask.ndim == 4:
            none = ~attn_mask.any(-1, keepdim=True)
            if none.any():
                eye = torch.eye(L, dtype=torch.bool, device=x.device)[None, None]
                attn_mask = attn_mask | (none & eye)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self.proj(out.transpose(1, 2).reshape(G, L, D))
        return out * original_valid.unsqueeze(-1)

    def temporal(
        self, x: torch.Tensor, valid: torch.Tensor, C: int, T: int, window: int
    ) -> torch.Tensor:
        B, _, D = x.shape
        seq = x.reshape(B, C, T, D).reshape(B * C, T, D)
        mask = valid.reshape(B, C, T).reshape(B * C, T)
        ti = torch.arange(T, device=x.device)
        local = ((ti[:, None] - ti[None, :]).abs() <= window)[None, None]
        return self._attend(seq, mask, local).reshape(B, C * T, D)

    def spatial(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        ch_pos: torch.Tensor | None,
        C: int,
        T: int,
        k_neighbors: int,
    ) -> torch.Tensor:
        B, _, D = x.shape
        seq = x.reshape(B, C, T, D).permute(0, 2, 1, 3).reshape(B * T, C, D)
        mask = valid.reshape(B, C, T).permute(0, 2, 1).reshape(B * T, C)
        graph = None
        if ch_pos is not None:
            channel_valid = valid.reshape(B, C, T).any(-1)
            dist = torch.cdist(ch_pos.float(), ch_pos.float())
            dist = dist.masked_fill(~channel_valid[:, None, :], float("inf"))
            k = min(C, k_neighbors)
            nearest = dist.topk(k, largest=False).indices
            graph_b = torch.zeros(B, C, C, dtype=torch.bool, device=x.device)
            graph_b.scatter_(2, nearest, True)
            graph_b &= channel_valid[:, None, :]
            graph = graph_b[:, None].expand(B, T, C, C).reshape(B * T, 1, C, C)
        out = self._attend(seq, mask, graph)
        return out.reshape(B, T, C, D).permute(0, 2, 1, 3).reshape(B, C * T, D)
