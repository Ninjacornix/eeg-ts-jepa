"""Downstream evaluation: freeze the encoder, fit a linear probe.

Also supports full fine-tuning (unfreeze the encoder).  Returns accuracy and a
per-subgroup breakdown for disaggregated reporting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import collate_variable_montage
from ..jepa import EEGJepa
from .pretrain import move


class LinearProbe(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(x)


def _loader(ds, bs=64, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      collate_fn=collate_variable_montage)


@torch.no_grad()
def _features(model: EEGJepa, loader, device, pool="mean"):
    feats, labels, subgroups = [], [], []
    model.eval()
    for batch in loader:
        b = move(batch, device)
        feats.append(model.encode(b, pool=pool).cpu())
        labels.append(batch["label"])
        subgroups.append(batch["subgroup"])
    return (torch.cat(feats), torch.cat(labels), torch.cat(subgroups))


def fit_linear_probe(
    model: EEGJepa, train_ds, test_ds, n_classes: int,
    device: torch.device, epochs: int = 50, lr: float = 1e-2,
    finetune: bool = False, verbose: bool = False, pool: str = "mean",
) -> dict:
    if finetune:
        return _finetune(model, train_ds, test_ds, n_classes, device, epochs, lr, verbose)

    # frozen encoder -> precompute features once
    Xtr, ytr, _ = _features(model, _loader(train_ds, shuffle=False), device, pool)
    Xte, yte, gte = _features(model, _loader(test_ds), device, pool)

    # standardize features with TRAIN statistics (crucial for a linear probe)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # PCA-reduce when features >> samples (e.g. channel-preserving pooling) to
    # avoid p>>n overfitting; components fit on TRAIN only (svd on CPU for MPS).
    if Xtr.shape[1] > Xtr.shape[0]:
        k = min(64, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, V = torch.linalg.svd(Xtr.cpu(), full_matrices=False)
        comp = V[:k]                                   # (k, D)
        Xtr = Xtr @ comp.t()
        Xte = Xte @ comp.t()
        if verbose:
            print(f"  PCA: {comp.shape[1]} -> {k} dims")

    Xtr = Xtr.to(device)
    Xte = Xte.to(device)
    ytr = ytr.to(device)

    probe = LinearProbe(Xtr.shape[1], n_classes).to(device)
    # stronger weight decay for high-dim (channel-preserving) features
    wd = 1e-2 if Xtr.shape[1] > 512 else 1e-3
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    bs = min(128, n)
    steps_per_epoch = max(1, n // bs)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, generator=g)
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            loss = F.cross_entropy(probe(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if verbose and ep % 10 == 0:
            print(f"  probe ep{ep} loss {float(loss):.3f}")

    probe.eval()
    with torch.no_grad():
        train_acc = float((probe(Xtr).argmax(-1).cpu() == ytr.cpu()).float().mean())
        pred = probe(Xte).argmax(-1).cpu()
    res = _metrics(pred, yte, gte, n_classes)
    res["train_accuracy"] = train_acc          # diagnose probe-fit vs feature quality
    return res


def _finetune(model, train_ds, test_ds, n_classes, device, epochs, lr, verbose):
    probe = LinearProbe(model.cfg.model.dim, n_classes).to(device)
    # unfreeze context encoder + tokenizer
    enc_params = list(model.tokenizer.parameters()) + list(model.context_encoder.parameters())
    for p in enc_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr * 0.1},
         {"params": probe.parameters(), "lr": lr}], weight_decay=1e-4
    )
    tl = _loader(train_ds, shuffle=True)
    for ep in range(epochs):
        model.train(); probe.train()
        for batch in tl:
            b = move(batch, device)
            grid = model.tokenizer(b["signal"], b["ch_ids"], b["ch_pos"], b["ch_mask"])
            r = model.context_encoder(grid.tokens, grid.ch_index, grid.time_index, grid.token_mask)
            w = grid.token_mask.float().unsqueeze(-1)
            pooled = (r * w).sum(1) / w.sum(1).clamp_min(1.0)
            loss = F.cross_entropy(probe(pooled), b["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if verbose:
            print(f"  ft ep{ep} loss {float(loss):.3f}")

    model.eval(); probe.eval()
    preds, ys, gs = [], [], []
    with torch.no_grad():
        for batch in _loader(test_ds):
            b = move(batch, device)
            grid = model.tokenizer(b["signal"], b["ch_ids"], b["ch_pos"], b["ch_mask"])
            r = model.context_encoder(grid.tokens, grid.ch_index, grid.time_index, grid.token_mask)
            w = grid.token_mask.float().unsqueeze(-1)
            pooled = (r * w).sum(1) / w.sum(1).clamp_min(1.0)
            preds.append(probe(pooled).argmax(-1).cpu())
            ys.append(batch["label"]); gs.append(batch["subgroup"])
    return _metrics(torch.cat(preds), torch.cat(ys), torch.cat(gs), n_classes)


@torch.no_grad()
def raw_logvar_features(ds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classic MI feature: log-variance (log band-power) per channel.

    Baseline that bypasses the model entirely -> tells you the achievable
    accuracy ceiling on this data/montage, so you know whether a chance-level
    probe means a weak encoder or an inherently hard task/split.
    """
    feats, labels, subs = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        x = item["signal"]                               # (C, T)
        feats.append(torch.log(x.var(dim=1) + 1e-6))     # (C,)
        labels.append(item["label"]); subs.append(item["subgroup"])
    return torch.stack(feats), torch.tensor(labels), torch.tensor(subs)


def fit_raw_baseline(train_ds, test_ds, n_classes, device, epochs=100, lr=1e-2) -> dict:
    Xtr, ytr, _ = raw_logvar_features(train_ds)
    Xte, yte, gte = raw_logvar_features(test_ds)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device); Xte = ((Xte - mu) / sd).to(device)
    ytr = ytr.to(device)
    probe = LinearProbe(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(Xtr), ytr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        pred = probe(Xte).argmax(-1).cpu()
    return _metrics(pred, yte, gte, n_classes)


def _metrics(pred, y, g, n_classes) -> dict:
    acc = float((pred == y).float().mean())
    subgroup_acc = {}
    for grp in g.unique().tolist():
        m = g == grp
        subgroup_acc[int(grp)] = float((pred[m] == y[m]).float().mean())
    # balanced accuracy (macro recall) -> robust to class imbalance
    recalls = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            recalls.append(float((pred[m] == y[m]).float().mean()))
    return {
        "accuracy": acc,
        "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
        "subgroup_accuracy": subgroup_acc,
        "subgroup_gap": (max(subgroup_acc.values()) - min(subgroup_acc.values()))
        if subgroup_acc else 0.0,
    }


def main() -> None:
    import argparse
    from ..config import PretrainConfig
    from ..data import SyntheticEEGDataset
    from .pretrain import pick_device, pretrain

    p = argparse.ArgumentParser(description="Linear-probe a pretrained EEG-JEPA")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = pick_device(a.device)
    if a.ckpt:
        blob = torch.load(a.ckpt, map_location=device, weights_only=False)
        cfg = blob["cfg"]
        model = EEGJepa(cfg).to(device)
        model.load_state_dict(blob["state_dict"])
    else:
        cfg = PretrainConfig(); cfg.epochs = 5; cfg.model.patch_ms = 125
        model = pretrain(cfg, device=device, n_sites=3, per_site=128)

    tr = SyntheticEEGDataset(256, site_id=10, seed=7)
    te = SyntheticEEGDataset(256, site_id=11, seed=8)
    res = fit_linear_probe(model, tr, te, cfg_n := 5, device, finetune=a.finetune)
    print("downstream:", res)


if __name__ == "__main__":
    main()
