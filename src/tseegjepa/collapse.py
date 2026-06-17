"""Representation-collapse diagnostics for JEPA pretraining.

Latent-prediction objectives can collapse (encoder maps everything to a near
constant).  We monitor two cheap signals on the *target* embeddings:

  * mean per-dimension std  -> shrinks toward 0 on collapse
  * effective rank (entropy of normalized singular values) -> shrinks toward 1

Call `collapse_stats` periodically; alert when below thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CollapseStats:
    embed_std: float        # mean per-dim std of (L2-normalized) embeddings
    eff_rank: float         # effective rank (>=1)
    rank_ratio: float       # eff_rank / dim  in (0, 1]
    collapsed: bool

    def __str__(self) -> str:
        flag = "  <-- COLLAPSE" if self.collapsed else ""
        return (f"std={self.embed_std:.4f} eff_rank={self.eff_rank:.1f} "
                f"ratio={self.rank_ratio:.3f}{flag}")


@torch.no_grad()
def effective_rank(x: torch.Tensor) -> float:
    """Entropy-based effective rank of a (M, D) matrix."""
    x = x - x.mean(0, keepdim=True)
    if x.shape[0] < 2:
        return 1.0
    # svdvals unimplemented on MPS -> run on CPU
    s = torch.linalg.svdvals(x.float().cpu())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    ent = -(p * p.log()).sum()
    return float(ent.exp())


@torch.no_grad()
def collapse_stats(
    embeddings: torch.Tensor,   # (M, D) target embeddings (flattened over batch/tokens)
    std_thresh: float = 0.01,
    rank_ratio_thresh: float = 0.10,
    max_rows: int = 4096,
) -> CollapseStats:
    x = embeddings.detach().float()
    if x.shape[0] > max_rows:
        sel = torch.randperm(x.shape[0], device=x.device)[:max_rows]
        x = x[sel]
    D = x.shape[1]
    xn = torch.nn.functional.normalize(x, dim=-1)
    std = float(xn.std(0).mean())
    er = effective_rank(x)
    ratio = er / D
    collapsed = (std < std_thresh) or (ratio < rank_ratio_thresh)
    return CollapseStats(std, er, ratio, collapsed)
