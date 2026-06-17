"""Leave-one-dataset-out (LODO) cross-site evaluation.

For each held-out site:
  * pretrain (or reuse) the encoder on the remaining sites,
  * fit a frozen linear probe on a labeled split from the held-out site,
  * report clean accuracy, subgroup-disaggregated accuracy, and the accuracy
    drop under each corruption (OOD stress).

This is the headline generalization metric: performance on a device/site never
seen during pretraining.
"""

from __future__ import annotations

import argparse

import torch

from ..config import PretrainConfig
from ..data import SyntheticEEGDataset
from ..jepa import EEGJepa
from ..train.linear_probe import fit_linear_probe
from ..train.pretrain import pick_device, pretrain
from .corruptions import CORRUPTIONS, CorruptedDataset


def leave_one_dataset_out(
    n_sites: int = 4,
    cfg: PretrainConfig | None = None,
    device: torch.device | None = None,
    n_classes: int = 5,
    pretrain_epochs: int = 4,
    per_site: int = 128,
    probe_n: int = 256,
    corruptions: list[str] | None = None,
    severity: float = 1.0,
    verbose: bool = True,
) -> dict:
    device = device or pick_device()
    corruptions = corruptions if corruptions is not None else list(CORRUPTIONS)
    results = {}

    for held in range(n_sites):
        train_sites = [s for s in range(n_sites) if s != held]
        cfg_h = cfg or PretrainConfig()
        cfg_h.epochs = pretrain_epochs
        cfg_h.model.patch_ms = max(cfg_h.model.patch_ms, 125.0)

        # pretrain on train sites only (remap ids 0..n-2)
        model = EEGJepa(cfg_h).to(device)
        from torch.utils.data import ConcatDataset, DataLoader
        from ..data import collate_variable_montage
        from ..train.pretrain import _ema_at, _lr_at, move

        ds = ConcatDataset([
            SyntheticEEGDataset(per_site, site_id=s, seed=s)
            for s in train_sites
        ])
        loader = DataLoader(ds, batch_size=cfg_h.batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_variable_montage)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg_h.lr, weight_decay=cfg_h.weight_decay)
        gen = torch.Generator().manual_seed(held)
        total = cfg_h.epochs * len(loader)
        warm = cfg_h.warmup_epochs * len(loader)
        step = 0
        model.train()
        for _ in range(cfg_h.epochs):
            for batch in loader:
                batch = move(batch, device)
                for g in opt.param_groups:
                    g["lr"] = _lr_at(step, total, warm, cfg_h.lr)
                out = model(batch, generator=gen)
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, cfg_h.grad_clip)
                opt.step()
                model.update_target(_ema_at(step, total, cfg_h.ema_base, cfg_h.ema_final))
                step += 1

        # downstream on held-out site
        probe_tr = SyntheticEEGDataset(probe_n, site_id=held, seed=100 + held)
        probe_te = SyntheticEEGDataset(probe_n, site_id=held, seed=200 + held)
        clean = fit_linear_probe(model, probe_tr, probe_te, n_classes, device)

        corrupt = {}
        for name in corruptions:
            cte = CorruptedDataset(probe_te, name, severity=severity, seed=held)
            r = fit_linear_probe(model, probe_tr, cte, n_classes, device)
            corrupt[name] = {
                "accuracy": r["accuracy"],
                "drop": clean["accuracy"] - r["accuracy"],
            }

        results[f"held_site_{held}"] = {
            "clean": clean,
            "ood": corrupt,
        }
        if verbose:
            print(f"\n=== held-out site {held} (pretrained on {train_sites}) ===")
            print(f"  clean acc={clean['accuracy']:.3f} "
                  f"bal={clean['balanced_accuracy']:.3f} "
                  f"subgroup_gap={clean['subgroup_gap']:.3f}")
            print(f"  subgroup acc: {clean['subgroup_accuracy']}")
            for name, r in corrupt.items():
                print(f"  OOD[{name:15s}] acc={r['accuracy']:.3f} drop={r['drop']:+.3f}")

    # aggregate
    accs = [results[k]["clean"]["accuracy"] for k in results]
    gaps = [results[k]["clean"]["subgroup_gap"] for k in results]
    results["_summary"] = {
        "mean_clean_accuracy": sum(accs) / len(accs),
        "worst_clean_accuracy": min(accs),
        "max_subgroup_gap": max(gaps),
    }
    if verbose:
        print("\n=== LODO summary ===", results["_summary"])
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-one-dataset-out EEG-JEPA eval")
    p.add_argument("--sites", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--per-site", type=int, default=128)
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    leave_one_dataset_out(
        n_sites=a.sites, pretrain_epochs=a.epochs,
        per_site=a.per_site, severity=a.severity,
        device=pick_device(a.device),
    )


if __name__ == "__main__":
    main()
