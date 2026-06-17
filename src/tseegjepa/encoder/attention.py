"""Branch attention masks + multi-head attention.

Branches are realised as *structured attention patterns* over an arbitrary set
of tokens, each token carrying a channel index and a time index:

  * temporal branches (one per scale window `w`): a token attends only to
    tokens on the SAME channel within +/- w time patches.  Different `w` give
    short / medium / long temporal receptive fields.
  * spatial branch: a token attends to tokens at the SAME time patch on ANY
    channel (electrode-graph / channel attention).

Because masks are derived from indices, they work for any montage and for any
masked subset of tokens (I-JEPA-style gathered context/target sets).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_branch_masks(
    ch_index: torch.Tensor,     # (B, M)
    time_index: torch.Tensor,   # (B, M)
    token_mask: torch.Tensor,   # (B, M) True = valid
    temporal_windows: tuple[int, ...],
    use_spatial: bool,
) -> list[torch.Tensor]:
    """Return a list of boolean attn masks, each (B, 1, M, M); True = attend."""
    same_ch = ch_index.unsqueeze(2) == ch_index.unsqueeze(1)      # (B,M,M)
    same_t = time_index.unsqueeze(2) == time_index.unsqueeze(1)
    dt = (time_index.unsqueeze(2) - time_index.unsqueeze(1)).abs()
    valid_kv = token_mask.unsqueeze(1)                            # (B,1,M) over keys

    masks: list[torch.Tensor] = []
    for w in temporal_windows:
        m = same_ch & (dt <= w)
        m = m & valid_kv
        masks.append(m.unsqueeze(1))
    if use_spatial:
        m = same_t & valid_kv
        masks.append(m.unsqueeze(1))

    # guard against all-False rows (a valid query with no valid keys) which would
    # make softmax produce NaNs: let such a query attend to itself.
    eye = torch.eye(ch_index.shape[1], dtype=torch.bool, device=ch_index.device)
    eye = eye.view(1, 1, ch_index.shape[1], ch_index.shape[1])
    fixed = []
    for m in masks:
        none_attended = ~m.any(dim=-1, keepdim=True)              # (B,1,M,1)
        fixed.append(m | (none_attended & eye))
    return fixed


class BranchMHA(nn.Module):
    """Standard MHA that consumes a precomputed boolean attention mask."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, M, D = x.shape
        qkv = self.qkv(x).reshape(B, M, 3, self.heads, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)           # each (B,H,M,dh)
        # attn_mask: (B,1,M,M) bool -> broadcast over heads
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, M, D)
        return self.proj(out)
