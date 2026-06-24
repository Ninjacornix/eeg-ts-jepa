"""Typed EEG sample/batch contracts and runtime validation."""

from __future__ import annotations

from typing import TypedDict

import torch


class EEGSample(TypedDict):
    signal: torch.Tensor
    ch_ids: torch.Tensor
    ch_pos: torch.Tensor
    label: int
    subgroup: int
    domain: int


class EEGBatch(TypedDict):
    signal: torch.Tensor
    sample_mask: torch.Tensor
    ch_ids: torch.Tensor
    ch_pos: torch.Tensor
    ch_mask: torch.Tensor
    label: torch.Tensor
    subgroup: torch.Tensor
    domain: torch.Tensor


def validate_eeg_batch(batch: dict) -> None:
    required = {
        "signal", "ch_ids", "ch_pos", "ch_mask", "label", "subgroup", "domain"
    }
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"EEG batch missing fields: {sorted(missing)}")
    signal = batch["signal"]
    ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
    if signal.ndim != 3:
        raise ValueError(f"signal must be (B,C,T), got {tuple(signal.shape)}")
    B, C, T = signal.shape
    if ch_ids.shape != (B, C) or ch_mask.shape != (B, C):
        raise ValueError("ch_ids/ch_mask must match signal's (B,C)")
    if ch_pos.shape != (B, C, 3):
        raise ValueError("ch_pos must be (B,C,3)")
    if T < 1 or not ch_mask.any(dim=1).all():
        raise ValueError("every EEG sample needs time samples and at least one channel")
    sample_mask = batch.get("sample_mask")
    if sample_mask is not None and sample_mask.shape != (B, T):
        raise ValueError("sample_mask must match signal's (B,T)")
