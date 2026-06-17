"""OOD / corruption stress tests.

Each corruption simulates a realistic acquisition shift.  Wrap any dataset with
`CorruptedDataset(ds, name, severity)` to measure how much downstream accuracy
degrades vs. the clean baseline.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def _gaussian(sig, sev, g):
    return sig + sev * 0.5 * torch.randn(sig.shape, generator=g)


def _channel_dropout(sig, sev, g):
    C = sig.shape[0]
    k = int(round(sev * 0.5 * C))
    if k <= 0:
        return sig
    drop = torch.randperm(C, generator=g)[:k]
    sig = sig.clone()
    sig[drop] = 0.0
    return sig


def _amplitude(sig, sev, g):
    scale = 1.0 + sev * (torch.rand(sig.shape[0], 1, generator=g) - 0.5) * 2.0
    return sig * scale


def _line_noise(sig, sev, g):
    T = sig.shape[1]
    t = torch.arange(T).float() / T
    hz = 50.0
    return sig + sev * torch.sin(2 * torch.pi * hz * t * 100).unsqueeze(0)


def _temporal_shift(sig, sev, g):
    s = int(round(sev * 0.2 * sig.shape[1]))
    if s == 0:
        return sig
    return torch.roll(sig, shifts=s, dims=1)


def _drift(sig, sev, g):
    T = sig.shape[1]
    walk = torch.cumsum(torch.randn(sig.shape[0], T, generator=g), dim=1) / T
    return sig + sev * walk


CORRUPTIONS = {
    "gaussian": _gaussian,
    "channel_dropout": _channel_dropout,
    "amplitude": _amplitude,
    "line_noise": _line_noise,
    "temporal_shift": _temporal_shift,
    "drift": _drift,
}


class CorruptedDataset(Dataset):
    def __init__(self, base: Dataset, name: str, severity: float = 1.0, seed: int = 0):
        if name not in CORRUPTIONS:
            raise KeyError(f"unknown corruption {name!r}; options: {list(CORRUPTIONS)}")
        self.base = base
        self.fn = CORRUPTIONS[name]
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        sig = item["signal"]
        sig = self.fn(sig, self.severity, g)
        # re-standardize so corruption changes shape/SNR, not just global scale
        sig = (sig - sig.mean(1, keepdim=True)) / (sig.std(1, keepdim=True) + 1e-6)
        item["signal"] = sig
        return item
