"""Multi-scale shared encoder.

Every block runs the temporal-scale branches and the spatial branch *in
parallel* over the same tokens, then fuses their outputs into one latent space
(a learned linear combination), followed by an MLP.  Stacking blocks lets the
fused representation mix scales hierarchically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import BranchMHA, build_branch_masks


class MultiScaleBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, n_branches: int):
        super().__init__()
        self.n_branches = n_branches
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.branches = nn.ModuleList(
            [BranchMHA(cfg.dim, cfg.heads, cfg.dropout) for _ in range(n_branches)]
        )
        # fuse parallel branch outputs -> single latent
        self.fuse = nn.Linear(n_branches * cfg.dim, cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(hidden, cfg.dim),
        )

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        outs = [branch(h, m) for branch, m in zip(self.branches, masks)]
        x = x + self.fuse(torch.cat(outs, dim=-1))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleEncoder(nn.Module):
    """Token-set encoder; operates on (B, M, D) tokens + index tensors.

    Accepts an arbitrary (possibly masked / gathered) set of tokens, so the
    same module serves as both the context encoder and the EMA target encoder.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_branches = len(cfg.temporal_windows) + (1 if cfg.use_spatial_branch else 0)
        self.blocks = nn.ModuleList(
            [MultiScaleBlock(cfg, self.n_branches) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, M, D)
        ch_index: torch.Tensor,     # (B, M)
        time_index: torch.Tensor,   # (B, M)
        token_mask: torch.Tensor,   # (B, M) True = valid
    ) -> torch.Tensor:
        masks = build_branch_masks(
            ch_index, time_index, token_mask,
            self.cfg.temporal_windows, self.cfg.use_spatial_branch,
        )
        x = tokens
        for blk in self.blocks:
            x = blk(x, masks)
        return self.norm(x)

    @torch.no_grad()
    def pooled(self, *args, **kwargs) -> torch.Tensor:
        """Masked mean-pool over valid tokens -> (B, D) sample embedding."""
        token_mask = kwargs.get("token_mask", args[3] if len(args) > 3 else None)
        x = self.forward(*args, **kwargs)
        w = token_mask.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)
