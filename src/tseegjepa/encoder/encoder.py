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
from .attention import BranchMHA


class MultiScaleBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_branches = len(cfg.temporal_windows) + int(cfg.use_spatial_branch)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.temporal = nn.ModuleList(
            [BranchMHA(cfg.dim, cfg.heads, cfg.dropout) for _ in cfg.temporal_windows]
        )
        self.spatial = (
            BranchMHA(cfg.dim, cfg.heads, cfg.dropout)
            if cfg.use_spatial_branch else None
        )
        # fuse parallel branch outputs -> single latent
        self.fuse = nn.Linear(self.n_branches * cfg.dim, cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(hidden, cfg.dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        ch_pos: torch.Tensor | None,
        C: int,
        T: int,
    ) -> torch.Tensor:
        h = self.norm1(x)
        outs = [
            branch.temporal(h, token_mask, C, T, window)
            for branch, window in zip(self.temporal, self.cfg.temporal_windows)
        ]
        if self.spatial is not None:
            outs.append(
                self.spatial.spatial(
                    h, token_mask, ch_pos, C, T, self.cfg.spatial_k
                )
            )
        x = x + self.fuse(torch.cat(outs, dim=-1))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleEncoder(nn.Module):
    """Factorized encoder over a padded, masked ``(channel, time)`` token grid."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_branches = len(cfg.temporal_windows) + (1 if cfg.use_spatial_branch else 0)
        self.blocks = nn.ModuleList(
            [MultiScaleBlock(cfg) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, M, D)
        ch_index: torch.Tensor,     # (B, M)
        time_index: torch.Tensor,   # (B, M)
        token_mask: torch.Tensor,   # (B, M) True = valid
        ch_pos: torch.Tensor | None = None,  # (B,C,3)
    ) -> torch.Tensor:
        C = int(ch_index.max().item()) + 1
        T = int(time_index.max().item()) + 1
        x_tokens = tokens.shape[1]
        if C * T != x_tokens:
            raise ValueError(f"factorized encoder expected C*T={C*T}, got {x_tokens}")
        x = tokens
        for blk in self.blocks:
            x = blk(x, token_mask, ch_pos, C, T)
        return self.norm(x)

    @torch.no_grad()
    def pooled(self, *args, **kwargs) -> torch.Tensor:
        """Masked mean-pool over valid tokens -> (B, D) sample embedding."""
        token_mask = kwargs.get("token_mask", args[3] if len(args) > 3 else None)
        x = self.forward(*args, **kwargs)
        w = token_mask.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)
