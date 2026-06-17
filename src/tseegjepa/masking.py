"""Masking for EEG-JEPA.

Targets are *blocks* = (a spatially-contiguous block of electrodes) x (a
variable-duration temporal span).  Several such blocks are unioned per sample:

  * spatial-block masking  -> contiguous runs of nearby electrodes
  * variable-duration temporal masking -> spans of random length in [min,max]

The context view is everything valid that is NOT a target token.  We keep full
(B, N) boolean masks (no ragged gather): the context encoder simply hides target
tokens from attention, and the predictor replaces them with mask tokens.
"""

from __future__ import annotations

import torch

from .config import MaskConfig


def _ordered_channels(ch_pos: torch.Tensor, valid: torch.Tensor) -> list[int]:
    """Valid channel slots ordered by scalp angle -> spatial contiguity."""
    ang = torch.atan2(ch_pos[:, 1], ch_pos[:, 0])
    idx = [i for i in range(ch_pos.shape[0]) if bool(valid[i])]
    idx.sort(key=lambda i: float(ang[i]))
    return idx


def make_jepa_masks(
    ch_pos: torch.Tensor,       # (B, C, 3)
    ch_mask: torch.Tensor,      # (B, C) valid channels
    n_time: int,                # Tp
    cfg: MaskConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (context_mask, target_mask), each (B, C*Tp) boolean.

    context_mask: tokens visible to the context encoder (valid & not target).
    target_mask:  tokens the predictor must forecast.
    """
    B, C, _ = ch_pos.shape
    dev = ch_pos.device

    def rint(lo, hi):
        if hi <= lo:
            return lo
        return int(torch.randint(lo, hi, (1,), generator=generator, device="cpu").item())

    target = torch.zeros(B, C, n_time, dtype=torch.bool, device=dev)
    for b in range(B):
        chans = _ordered_channels(ch_pos[b], ch_mask[b])
        if not chans:
            continue
        n_block_ch = max(1, int(round(len(chans) * cfg.spatial_block_frac
                                      / max(1, cfg.n_spatial_blocks))))
        for _ in range(cfg.n_target_blocks):
            # spatial block: contiguous run in angle-ordered channel list
            start = rint(0, max(1, len(chans) - n_block_ch + 1))
            block_ch = chans[start:start + n_block_ch]
            # temporal span: variable duration
            span = rint(cfg.min_span, cfg.max_span + 1)
            span = min(span, n_time)
            t0 = rint(0, max(1, n_time - span + 1))
            for c in block_ch:
                target[b, c, t0:t0 + span] = True

    valid = ch_mask.unsqueeze(-1).expand(B, C, n_time)
    target = target & valid

    # guarantee non-degenerate split: if a sample has no target or no context,
    # fall back to masking a single central temporal span.
    for b in range(B):
        if not target[b].any() or not (valid[b] & ~target[b]).any():
            target[b] = False
            t0 = n_time // 4
            target[b, :, t0:t0 + max(1, n_time // 2)] = valid[b, :, t0:t0 + max(1, n_time // 2)]

    context = valid & ~target
    return (context.reshape(B, -1), target.reshape(B, -1))
