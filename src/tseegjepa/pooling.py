"""Montage-invariant spatial pooling over electrode representations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _fibonacci_sphere(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float32)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    theta = i * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack((r * theta.cos(), r * theta.sin(), z), dim=-1)


class SpatialAnchorPool(nn.Module):
    """Project arbitrary electrodes onto fixed spherical anchors.

    Output size is always ``n_anchors * dim``. This preserves coarse scalp
    topology without depending on channel count or input channel ordering.
    """

    def __init__(self, n_anchors: int, temperature: float = 0.25):
        super().__init__()
        self.n_anchors = n_anchors
        self.temperature = temperature
        self.register_buffer("anchors", _fibonacci_sphere(n_anchors))

    def forward(
        self,
        channel_repr: torch.Tensor,  # (B,C,D)
        ch_pos: torch.Tensor,        # (B,C,3)
        ch_mask: torch.Tensor,       # (B,C)
    ) -> torch.Tensor:
        pos = torch.nn.functional.normalize(ch_pos.float(), dim=-1)
        anchors = self.anchors.to(device=pos.device, dtype=pos.dtype)
        scores = torch.einsum("bcd,kd->bkc", pos, anchors) / self.temperature
        scores = scores.masked_fill(~ch_mask[:, None, :], float("-inf"))
        weights = scores.softmax(dim=-1).to(channel_repr.dtype)
        pooled = torch.einsum("bkc,bcd->bkd", weights, channel_repr)
        return pooled.flatten(1)
