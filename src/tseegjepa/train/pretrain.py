"""Self-supervised pretraining loop for EEG-JEPA.

Multi-site synthetic data by default (diverse montages/devices) -> the model is
pushed toward cross-site generalization. Latent-space prediction loss, a full
EMA target tower, cosine LR + warmup, and collapse-aware validation.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..config import PretrainConfig
from ..data import SyntheticEEGDataset, collate_variable_montage
from ..jepa import EEGJepa
from .checkpoint import save_checkpoint
from .engine import PretrainTrainer
from .utils import ema_at, lr_at, move, pick_device

# Backward-compatible names used by external scripts.
_lr_at = lr_at
_ema_at = ema_at


def build_pretrain_loader(cfg: PretrainConfig, n_sites: int = 4,
                          per_site: int = 256, seconds: float = 4.0):
    sites = [
        SyntheticEEGDataset(
            n_samples=per_site, seconds=seconds,
            sample_rate=cfg.model.sample_rate, site_id=s,
            seed=cfg.seed + s,
        )
        for s in range(n_sites)
    ]
    ds = ConcatDataset(sites)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_variable_montage, drop_last=True,
    )


def pretrain(cfg: PretrainConfig, device: torch.device | None = None,
             n_sites: int = 4, per_site: int = 256, seconds: float = 4.0,
             verbose: bool = True) -> EEGJepa:
    torch.manual_seed(cfg.seed)
    device = device or pick_device()
    cfg.n_domains = max(cfg.n_domains, n_sites)

    loader = build_pretrain_loader(cfg, n_sites, per_site, seconds)
    model = EEGJepa(cfg).to(device)
    trainer = PretrainTrainer(model, cfg, device)
    trainer.fit(loader.dataset, verbose=verbose)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain EEG-JEPA")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sites", type=int, default=4)
    p.add_argument("--per-site", type=int, default=256)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--patch-ms", type=float, default=125.0)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--domain-inv", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--save", default="eegjepa.pt")
    a = p.parse_args()

    cfg = PretrainConfig()
    cfg.epochs = a.epochs
    cfg.batch_size = a.batch_size
    cfg.model.patch_ms = a.patch_ms
    cfg.model.dim = a.dim
    cfg.model.depth = a.depth
    cfg.use_domain_invariance = a.domain_inv

    model = pretrain(cfg, device=pick_device(a.device),
                     n_sites=a.sites, per_site=a.per_site, seconds=a.seconds)
    save_checkpoint(a.save, model, cfg)
    print(f"saved -> {a.save}")


if __name__ == "__main__":
    main()
