"""Synthetic multi-montage EEG for end-to-end smoke tests and eval harnesses.

Generates band-limited oscillatory signals with per-"site" nuisance
characteristics (gain, line-noise, drift, sensor noise) so that
leave-one-dataset-out and corruption evaluations are meaningful.  Each sample
carries its electrode names, so montages vary across (and within) sites.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .montage import MONTAGES, Montage, channel_ids, channel_positions

# canonical frequency bands (Hz)
_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}
_BAND_NAMES = list(_BANDS.keys())


class SiteProfile:
    """Per-site nuisance parameters -> simulate cross-device domain shift."""

    def __init__(self, site_id: int, rng: np.random.Generator):
        self.site_id = site_id
        self.gain = float(rng.uniform(0.6, 1.6))
        self.line_hz = float(rng.choice([50.0, 60.0]))
        self.line_amp = float(rng.uniform(0.0, 0.4))
        self.drift_amp = float(rng.uniform(0.0, 0.3))
        self.noise_std = float(rng.uniform(0.05, 0.25))
        self.montage_name = str(rng.choice(list(MONTAGES.keys())))


class SyntheticEEGDataset(Dataset):
    """Variable-montage EEG with a discrete class label per window.

    The class is encoded as the dominant frequency band so that a linear probe
    on a good representation should recover it regardless of montage/site.
    """

    def __init__(
        self,
        n_samples: int = 512,
        seconds: float = 4.0,
        sample_rate: int = 200,
        site_id: int = 0,
        n_classes: int = 5,
        seed: int = 0,
        fixed_montage: Montage | None = None,
    ):
        self.sr = sample_rate
        self.T = int(round(seconds * sample_rate))
        self.n_classes = min(n_classes, len(_BAND_NAMES))
        rng = np.random.default_rng(seed + 1000 * site_id)
        self.site = SiteProfile(site_id, rng)
        self.fixed_montage = fixed_montage
        # subgroup attribute (e.g. simulated age band) for disaggregated metrics
        self._meta = [
            {
                "label": int(rng.integers(self.n_classes)),
                "subgroup": int(rng.integers(2)),  # 0/1 simulated demographic split
                "seed": int(rng.integers(1 << 31)),
            }
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self._meta)

    def _montage(self, rng: np.random.Generator) -> Montage:
        if self.fixed_montage is not None:
            return self.fixed_montage
        return MONTAGES[self.site.montage_name]

    def __getitem__(self, idx: int) -> dict:
        m = self._meta[idx]
        rng = np.random.default_rng(m["seed"])
        montage = self._montage(rng)
        C = montage.n_channels
        t = np.arange(self.T) / self.sr

        lo, hi = _BANDS[_BAND_NAMES[m["label"]]]
        # shared latent source -> spatially mixed across channels
        n_src = 3
        srcs = np.zeros((n_src, self.T), dtype=np.float32)
        for s in range(n_src):
            f = rng.uniform(lo, hi)
            ph = rng.uniform(0, 2 * np.pi)
            srcs[s] = np.sin(2 * np.pi * f * t + ph)
        mix = rng.normal(0, 1, size=(C, n_src)).astype(np.float32)
        x = mix @ srcs                                   # (C, T)

        # site nuisances
        x *= self.site.gain
        x += self.site.line_amp * np.sin(2 * np.pi * self.site.line_hz * t)[None, :]
        x += self.site.drift_amp * np.cumsum(
            rng.normal(0, 1, size=(C, self.T)).astype(np.float32), axis=1
        ) / self.T
        x += rng.normal(0, self.site.noise_std, size=(C, self.T)).astype(np.float32)

        # per-channel z-score (robust to gain) -> still keeps domain shift in shape
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

        return {
            "signal": torch.from_numpy(x.astype(np.float32)),        # (C, T)
            "ch_ids": torch.from_numpy(channel_ids(list(montage.channels))),   # (C,)
            "ch_pos": torch.from_numpy(channel_positions(list(montage.channels))),  # (C,3)
            "label": int(m["label"]),
            "subgroup": int(m["subgroup"]),
            "domain": int(self.site.site_id),
        }


def collate_variable_montage(batch: list[dict]) -> dict:
    """Pad ragged channel/time dims and build a key-padding mask.

    Returns tensors shaped (B, Cmax, T) etc. plus `ch_mask` (B, Cmax) True=valid.
    Time length is assumed constant within a batch (fixed `seconds`).
    """
    B = len(batch)
    Cmax = max(b["signal"].shape[0] for b in batch)
    T = batch[0]["signal"].shape[1]

    signal = torch.zeros(B, Cmax, T)
    ch_ids = torch.zeros(B, Cmax, dtype=torch.long)
    ch_pos = torch.zeros(B, Cmax, 3)
    ch_mask = torch.zeros(B, Cmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        c = b["signal"].shape[0]
        signal[i, :c] = b["signal"]
        ch_ids[i, :c] = b["ch_ids"]
        ch_pos[i, :c] = b["ch_pos"]
        ch_mask[i, :c] = True

    return {
        "signal": signal,
        "ch_ids": ch_ids,
        "ch_pos": ch_pos,
        "ch_mask": ch_mask,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "subgroup": torch.tensor([b["subgroup"] for b in batch], dtype=torch.long),
        "domain": torch.tensor([b["domain"] for b in batch], dtype=torch.long),
    }
