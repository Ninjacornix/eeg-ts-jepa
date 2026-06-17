"""Self-supervised pretraining loop for EEG-JEPA.

Multi-site synthetic data by default (diverse montages/devices) -> the model is
pushed toward cross-site generalization.  Latent-space prediction loss, EMA
target encoder, cosine LR + warmup, and periodic collapse monitoring.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..config import PretrainConfig
from ..data import SyntheticEEGDataset, collate_variable_montage
from ..jepa import EEGJepa


def pick_device(prefer: str = "auto") -> torch.device:
    """Resolve a device. 'auto' picks cuda > mps > cpu.

    Explicit choices ('cuda', 'cuda:1', 'mps', 'cpu') are honored but validated:
    if the requested accelerator is unavailable we warn and fall back to cpu
    instead of crashing mid-run.
    """
    def _available(kind: str) -> bool:
        if kind == "cuda":
            return torch.cuda.is_available()
        if kind == "mps":
            return getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available()
        return True  # cpu

    if prefer == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif _available("mps"):
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(prefer)
        if not _available(dev.type):
            print(f"[device] {prefer!r} unavailable -> falling back to cpu")
            dev = torch.device("cpu")

    if dev.type == "cuda":
        idx = dev.index or 0
        print(f"[device] cuda:{idx} ({torch.cuda.get_device_name(idx)})")
    else:
        print(f"[device] {dev.type}")
    return dev


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


def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def _ema_at(step: int, total: int, base: float, final: float) -> float:
    return base + (final - base) * (step / max(1, total))


def move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pretrain(cfg: PretrainConfig, device: torch.device | None = None,
             n_sites: int = 4, per_site: int = 256, seconds: float = 4.0,
             verbose: bool = True) -> EEGJepa:
    torch.manual_seed(cfg.seed)
    device = device or pick_device()
    cfg.n_domains = max(cfg.n_domains, n_sites)

    loader = build_pretrain_loader(cfg, n_sites, per_site, seconds)
    model = EEGJepa(cfg).to(device)
    gen = torch.Generator().manual_seed(cfg.seed)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(loader)
    warmup = cfg.warmup_epochs * len(loader)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        run = 0.0
        for batch in loader:
            batch = move(batch, device)
            lr = _lr_at(step, total_steps, warmup, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            out = model(batch, generator=gen)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            model.update_target(_ema_at(step, total_steps, cfg.ema_base, cfg.ema_final))

            run += float(loss.detach())
            if verbose and step % cfg.collapse_log_every == 0:
                cs = model.collapse_report(out)
                dom = f" dom={float(out['loss_domain']):.3f}" if "loss_domain" in out else ""
                print(f"[ep {epoch:02d} step {step:05d}] lr={lr:.2e} "
                      f"loss={float(loss):.4f}{dom} | {cs} "
                      f"(ctx={out['n_context']} tgt={out['n_targets']})")
                if cs.collapsed:
                    print("  !! collapse warning: embedding variance/rank low")
            step += 1
        if verbose:
            print(f"== epoch {epoch:02d} mean loss {run / len(loader):.4f} ==")
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
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, a.save)
    print(f"saved -> {a.save}")


if __name__ == "__main__":
    main()
