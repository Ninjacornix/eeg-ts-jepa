"""Masking for EEG-JEPA.

Targets are *blocks* = (a spatially-contiguous block of electrodes) x (a
variable-duration temporal span).  Several such blocks are unioned per sample:

  * spatial-block masking -> channels nearest to sampled electrode centers
  * temporal masking -> exact configured coverage around sampled time centers

The context view is everything valid that is NOT a target token.  We keep full
(B, N) boolean masks (no ragged gather): the context encoder simply hides target
tokens from attention, and the predictor replaces them with mask tokens.
"""

from __future__ import annotations

import torch

from .config import MaskConfig


def _randperm(n: int, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randperm(n, generator=generator, device="cpu")


def make_jepa_masks(
    ch_pos: torch.Tensor,       # (B, C, 3)
    ch_mask: torch.Tensor,      # (B, C) valid channels
    n_time: int,                # Tp
    cfg: MaskConfig,
    generator: torch.Generator | None = None,
    time_mask: torch.Tensor | None = None,  # (B,Tp)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (context_mask, target_mask), each (B, C*Tp) boolean.

    context_mask: tokens visible to the context encoder (valid & not target).
    target_mask:  tokens the predictor must forecast.
    """
    B, C, _ = ch_pos.shape
    dev = ch_pos.device

    target = torch.zeros(B, C, n_time, dtype=torch.bool, device=dev)
    for b in range(B):
        chans = ch_mask[b].nonzero(as_tuple=False).flatten()
        available_t = (
            time_mask[b].nonzero(as_tuple=False).flatten()
            if time_mask is not None
            else torch.arange(n_time, device=dev)
        )
        if chans.numel() == 0 or available_t.numel() == 0:
            continue
        n_ch = max(1, min(chans.numel(), round(chans.numel() * cfg.spatial_block_frac)))
        n_centers = min(cfg.n_spatial_blocks, chans.numel())
        centers = chans[_randperm(chans.numel(), generator)[:n_centers].to(chans.device)]
        # True spatial blocks: choose channels nearest to one of several centers.
        dist = torch.cdist(ch_pos[b, chans].float(), ch_pos[b, centers].float())
        nearest_center = dist.min(-1).values
        selected_ch = chans[nearest_center.topk(n_ch, largest=False).indices]

        n_available = available_t.numel()
        target_t = max(1, round(n_available * cfg.temporal_mask_frac))
        target_t = min(target_t, n_available)
        target_t = max(target_t, min(n_available, cfg.min_span))
        # Add centers when needed so the requested fraction does not silently
        # stop growing on long recordings because of max_span.
        needed_centers = (target_t + cfg.max_span - 1) // cfg.max_span
        n_t_centers = min(
            max(cfg.n_target_blocks, needed_centers), n_available
        )
        centers_t = available_t[
            _randperm(n_available, generator)[:n_t_centers].to(available_t.device)
        ]
        nearest_t = (
            available_t[:, None] - centers_t[None, :]
        ).abs().min(-1).values
        selected_t = available_t[nearest_t.topk(target_t, largest=False).indices]

        target[b, selected_ch[:, None], selected_t[None, :]] = True

    valid_t = (
        time_mask if time_mask is not None
        else torch.ones(B, n_time, dtype=torch.bool, device=dev)
    )
    valid = ch_mask.unsqueeze(-1) & valid_t.unsqueeze(1)
    target = target & valid

    # guarantee non-degenerate split: if a sample has no target or no context,
    # fall back to masking a single central temporal span.
    for b in range(B):
        if not target[b].any() or not (valid[b] & ~target[b]).any():
            target[b] = False
            times = valid_t[b].nonzero(as_tuple=False).flatten()
            keep = times[:max(1, times.numel() // 2)]
            target[b, :, keep] = valid[b, :, keep]

    context = valid & ~target
    return (context.reshape(B, -1), target.reshape(B, -1))
