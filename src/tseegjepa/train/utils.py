"""Shared device, schedule, and batch utilities."""

from __future__ import annotations

import math

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    def available(kind: str) -> bool:
        if kind == "cuda":
            return torch.cuda.is_available()
        if kind == "mps":
            return (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            )
        return True

    if prefer == "auto":
        dev = torch.device(
            "cuda" if available("cuda") else "mps" if available("mps") else "cpu"
        )
    else:
        dev = torch.device(prefer)
        if not available(dev.type):
            print(f"[device] {prefer!r} unavailable -> falling back to cpu")
            dev = torch.device("cpu")
    if dev.type == "cuda":
        idx = dev.index or 0
        print(f"[device] cuda:{idx} ({torch.cuda.get_device_name(idx)})")
    else:
        print(f"[device] {dev.type}")
    return dev


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def ema_at(step: int, total: int, base: float, final: float) -> float:
    return base + (final - base) * (step / max(1, total))


def move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
