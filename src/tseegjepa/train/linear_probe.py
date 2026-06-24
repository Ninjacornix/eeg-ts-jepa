"""Downstream evaluation: freeze the encoder, fit a linear probe.

Also supports full fine-tuning (unfreeze the encoder).  Returns accuracy and a
per-subgroup breakdown for disaggregated reporting.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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
def _features_and_groups(model, loader, device, pool="mean", group_field="subgroup"):
    feats, labels, groups = [], [], []
    model.eval()
    for batch in loader:
        b = move(batch, device)
        feats.append(model.encode(b, pool=pool).cpu())
        labels.append(batch["label"])
        groups.append(batch[group_field])
    try:
        features = torch.cat(feats)
    except RuntimeError as exc:
        if pool == "chan":
            raise ValueError(
                "pool='chan' requires one fixed montage; use pool='spatial' "
                "for mixed channel counts"
            ) from exc
        raise
    return features, torch.cat(labels), torch.cat(groups)


def _features(model: EEGJepa, loader, device, pool="mean"):
    return _features_and_groups(model, loader, device, pool, "subgroup")


def fit_linear_probe(
    model: EEGJepa, train_ds, test_ds, n_classes: int,
    device: torch.device, epochs: int = 50, lr: float = 1e-2,
    finetune: bool = False, verbose: bool = False, pool: str = "mean",
    seed: int = 0, early_stop: bool = True, val_frac: float = 0.2,
) -> dict:
    """seed controls head init + batch shuffle (reproducible, vary across attempts).
    early_stop: hold out val_frac of TRAIN, keep the probe weights at best val loss
    instead of training a fixed number of epochs -> guards against overfitting."""
    if finetune:
        # Fine-tuning is an evaluation-local adaptation. Never mutate the shared
        # pretrained encoder used for another subject/fold.
        local_model = copy.deepcopy(model).to(device)
        return _finetune(local_model, train_ds, test_ds, n_classes, device, epochs, lr,
                         verbose, pool, seed)

    # frozen encoder -> precompute features once
    Xtr, ytr, _ = _features(model, _loader(train_ds, shuffle=False), device, pool)
    Xte, yte, gte = _features(model, _loader(test_ds), device, pool)

    # Split before fitting normalization/PCA so early-stop validation remains
    # genuinely unseen by the probe's preprocessing pipeline.
    g = torch.Generator().manual_seed(seed)
    n_all = Xtr.shape[0]
    val_idx, tr_idx = _stratified_indices(
        ytr, val_frac if early_stop and n_all > 10 else 0.0, g
    )
    fit_raw = Xtr[tr_idx]

    # standardize with FIT statistics only
    mu, sd = fit_raw.mean(0, keepdim=True), fit_raw.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # PCA-reduce when features >> samples (e.g. channel-preserving pooling) to
    # avoid p>>n overfitting; components fit on TRAIN only (svd on CPU for MPS).
    if Xtr.shape[1] > len(tr_idx):
        k = min(64, len(tr_idx) - 1, Xtr.shape[1])
        _, _, V = torch.linalg.svd(Xtr[tr_idx].cpu(), full_matrices=False)
        comp = V[:k]                                   # (k, D)
        Xtr = Xtr @ comp.t()
        Xte = Xte @ comp.t()
        if verbose:
            print(f"  PCA: {comp.shape[1]} -> {k} dims")

    Xtr = Xtr.to(device)
    Xte = Xte.to(device)
    ytr = ytr.to(device)
    tr_idx = tr_idx.to(device)
    val_idx = val_idx.to(device)

    n_val = len(val_idx)
    Xfit, yfit = Xtr[tr_idx], ytr[tr_idx]
    Xval, yval = Xtr[val_idx], ytr[val_idx]

    torch.manual_seed(seed)                            # deterministic head init
    probe = LinearProbe(Xtr.shape[1], n_classes).to(device)
    wd = 1e-2 if Xtr.shape[1] > 512 else 1e-3          # high-dim -> stronger decay
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    n = Xfit.shape[0]
    bs = min(128, n)
    steps_per_epoch = max(1, math.ceil(n / bs))

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, generator=g)
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            loss = F.cross_entropy(probe(Xfit[idx]), yfit[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if n_val:                                      # track best-on-val weights
            probe.eval()
            with torch.no_grad():
                vl = float(F.cross_entropy(probe(Xval), yval))
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        if verbose and ep % 10 == 0:
            print(f"  probe ep{ep} loss {float(loss):.3f}"
                  + (f" val {best_val:.3f}" if n_val else ""))

    if best_state is not None:                         # restore best (not last) weights
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        train_acc = float((probe(Xtr).argmax(-1).cpu() == ytr.cpu()).float().mean())
        pred = probe(Xte).argmax(-1).cpu()
    res = _metrics(pred, yte, gte, n_classes)
    res["train_accuracy"] = train_acc          # diagnose probe-fit vs feature quality
    return res


def fit_cross_subject_probe(
    model,
    train_ds,
    val_ds,
    test_ds,
    n_classes: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 3e-3,
    patience: int = 25,
    pool: str = "spatial",
    seed: int = 0,
    verbose: bool = False,
    return_artifact: bool = False,
) -> dict:
    """Fit one decoder on train subjects and evaluate zero-shot on test subjects.

    Validation subjects choose the decoder checkpoint. No test-subject examples
    participate in fitting, normalization, PCA, checkpoint selection, or tuning.
    """
    Xtr, ytr, str_ = _features_and_groups(
        model, _loader(train_ds, shuffle=False), device, pool, "domain"
    )
    Xva, yva, sva = _features_and_groups(
        model, _loader(val_ds, shuffle=False), device, pool, "domain"
    )
    Xte, yte, ste = _features_and_groups(
        model, _loader(test_ds, shuffle=False), device, pool, "domain"
    )
    return _fit_shared_probe(
        Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste,
        n_classes, device, epochs, lr, patience, seed, verbose,
        return_artifact=return_artifact,
    )


def _fit_shared_probe(
    Xtr,
    ytr,
    str_,
    Xva,
    yva,
    sva,
    Xte,
    yte,
    ste,
    n_classes,
    device,
    epochs,
    lr,
    patience,
    seed,
    verbose=False,
    return_artifact=False,
):
    if not len(Xtr) or not len(Xva) or not len(Xte):
        raise ValueError("cross-subject protocol requires non-empty train/val/test data")

    # All representation preprocessing is fit on labeled training subjects only.
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    # PCA is only necessary in the p>>n regime and is likewise train-only.
    comp = None
    if Xtr.shape[1] > Xtr.shape[0]:
        k = min(256, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, V = torch.linalg.svd(Xtr, full_matrices=False)
        comp = V[:k]
        Xtr, Xva, Xte = Xtr @ comp.t(), Xva @ comp.t(), Xte @ comp.t()
        if verbose:
            print(f"  shared-probe PCA: {comp.shape[1]} -> {k}")

    torch.manual_seed(seed)
    probe = LinearProbe(Xtr.shape[1], n_classes).to(device)
    weight_decay = 1e-2 if Xtr.shape[1] > 512 else 1e-3
    opt = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)

    # Equalize subjects and classes so a large/easy subject cannot dominate the
    # shared decoder learned for unseen people.
    subject_counts = torch.bincount(str_).float().clamp_min(1)
    class_counts = torch.bincount(ytr, minlength=n_classes).float().clamp_min(1)
    sample_weight = (
        subject_counts[str_].reciprocal()
        * class_counts[ytr].reciprocal()
    )
    sample_weight /= sample_weight.mean()

    batch_size = min(256, len(Xtr))
    best_score, best_state, best_epoch, stale = -float("inf"), None, -1, 0
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(Xtr), generator=generator)
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            wb = sample_weight[idx].to(device)
            loss = (F.cross_entropy(probe(xb), yb, reduction="none") * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_pred = _predict_feature_batches(probe, Xva, device)
        val_metrics = _subject_metrics(val_pred, yva, sva, n_classes)
        score = val_metrics["subject_balanced_accuracy_mean"]
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        else:
            stale += 1
        if verbose and (epoch % 10 == 0 or stale == 0):
            print(
                f"  shared probe ep{epoch:03d} loss={float(loss.detach()):.4f} "
                f"val_subj_bal={score:.3f}"
            )
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("shared cross-subject decoder found no valid checkpoint")
    probe.load_state_dict(best_state)
    test_pred = _predict_feature_batches(probe, Xte, device)
    result = _subject_metrics(test_pred, yte, ste, n_classes)
    result.update({
        "validation_subject_balanced_accuracy": best_score,
        "best_epoch": best_epoch,
        "n_train_subjects": int(str_.unique().numel()),
        "n_val_subjects": int(sva.unique().numel()),
        "n_test_subjects": int(ste.unique().numel()),
        "weight_decay": weight_decay,
    })
    if return_artifact:
        result["_decoder_artifact"] = {
            "feature_mean": mu.cpu(),
            "feature_std": sd.cpu(),
            "pca_components": comp.cpu() if comp is not None else None,
            "weight": probe.head.weight.detach().cpu(),
            "bias": probe.head.bias.detach().cpu(),
        }
    return result


@torch.no_grad()
def _predict_feature_batches(probe, features, device, batch_size=1024):
    probe.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        logits = probe(features[start:start + batch_size].to(device))
        predictions.append(logits.argmax(-1).cpu())
    return torch.cat(predictions)


def _finetune(model, train_ds, test_ds, n_classes, device, epochs, lr,
              verbose, pool="chan", seed=0):
    """End-to-end fine-tune: unfreeze encoder + tokenizer, train with a head on
    the SAME pooling as the probe (chan keeps C3/C4 lateralization -- mean-pooling
    here silently destroys the MI signal). Head sized to the pooled feature dim."""
    torch.manual_seed(seed)
    enc_params = list(model.feature_parameters())
    for p in enc_params:
        p.requires_grad_(True)

    fit_ds, val_ds = _dataset_train_val_split(train_ds, seed)
    # size the head from one batch of features
    sample = next(iter(_loader(fit_ds, shuffle=False)))
    with torch.no_grad():
        feat_dim = model.encode_features(move(sample, device), pool).shape[1]
    probe = LinearProbe(feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr * 0.1},
         {"params": probe.parameters(), "lr": lr}], weight_decay=1e-4
    )

    tl = _loader(fit_ds, shuffle=True)
    best_val, best_model, best_probe, stale = float("inf"), None, None, 0
    for ep in range(epochs):
        model.train(); probe.train()
        for batch in tl:
            b = move(batch, device)
            feat = model.encode_features(b, pool)         # grad flows
            loss = F.cross_entropy(probe(feat), b["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if verbose:
            print(f"  ft ep{ep} loss {float(loss):.3f}")
        if val_ds is not None:
            model.eval(); probe.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for batch in _loader(val_ds):
                    b = move(batch, device)
                    val_loss += float(F.cross_entropy(
                        probe(model.encode_features(b, pool)), b["label"]
                    ))
                    n_val += 1
            val_loss /= max(1, n_val)
            if val_loss < best_val:
                best_val, stale = val_loss, 0
                best_model = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_probe = {
                    k: v.detach().cpu().clone()
                    for k, v in probe.state_dict().items()
                }
            else:
                stale += 1
            if stale >= 8:
                break

    if best_model is not None:
        model.load_state_dict(best_model)
        probe.load_state_dict(best_probe)

    model.eval(); probe.eval()
    preds, ys, gs = [], [], []
    with torch.no_grad():
        for batch in _loader(test_ds):
            b = move(batch, device)
            feat = model.encode_features(b, pool)
            preds.append(probe(feat).argmax(-1).cpu())
            ys.append(batch["label"]); gs.append(batch["subgroup"])
    return _metrics(torch.cat(preds), torch.cat(ys), torch.cat(gs), n_classes)


def _stratified_indices(y, val_frac, generator):
    if val_frac <= 0:
        return torch.empty(0, dtype=torch.long), torch.arange(len(y))
    val, train = [], []
    for cls in y.unique(sorted=True):
        idx = (y == cls).nonzero(as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=generator)]
        n_val = min(max(1, round(val_frac * len(idx))), max(0, len(idx) - 1))
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return torch.cat(val), torch.cat(train)


def _dataset_train_val_split(ds, seed, val_frac=0.2):
    if len(ds) <= 10:
        return ds, None
    labels = torch.tensor([int(ds[i]["label"]) for i in range(len(ds))])
    val_idx, train_idx = _stratified_indices(
        labels, val_frac, torch.Generator().manual_seed(seed)
    )
    if not len(val_idx) or not len(train_idx):
        return ds, None
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


@torch.no_grad()
def raw_logvar_features(
    ds, group_field: str = "subgroup"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        labels.append(item["label"]); subs.append(item[group_field])
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


def fit_cross_subject_raw_baseline(
    train_ds,
    val_ds,
    test_ds,
    n_classes,
    device,
    epochs=200,
    lr=3e-3,
    patience=25,
    seed=0,
    verbose=False,
) -> dict:
    Xtr, ytr, str_ = raw_logvar_features(train_ds, "domain")
    Xva, yva, sva = raw_logvar_features(val_ds, "domain")
    Xte, yte, ste = raw_logvar_features(test_ds, "domain")
    return _fit_shared_probe(
        Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste,
        n_classes, device, epochs, lr, patience, seed, verbose,
    )


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


def _subject_metrics(pred, y, subject, n_classes) -> dict:
    pooled = _metrics(pred, y, subject, n_classes)
    per_subject = {}
    subject_acc, subject_bal = [], []
    for sid in subject.unique(sorted=True).tolist():
        mask = subject == sid
        sm = _metrics(
            pred[mask], y[mask], torch.zeros(mask.sum(), dtype=torch.long), n_classes
        )
        per_subject[int(sid)] = {
            "accuracy": sm["accuracy"],
            "balanced_accuracy": sm["balanced_accuracy"],
            "n_trials": int(mask.sum()),
        }
        subject_acc.append(sm["accuracy"])
        subject_bal.append(sm["balanced_accuracy"])
    return {
        "accuracy": pooled["accuracy"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "subject_accuracy_mean": float(torch.tensor(subject_acc).mean()),
        "subject_accuracy_std": float(torch.tensor(subject_acc).std(unbiased=False)),
        "subject_balanced_accuracy_mean": float(torch.tensor(subject_bal).mean()),
        "subject_balanced_accuracy_std": float(
            torch.tensor(subject_bal).std(unbiased=False)
        ),
        "per_subject": per_subject,
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
        from .checkpoint import load_checkpoint
        model, cfg, _, _, missing, unexpected = load_checkpoint(a.ckpt, device)
        if missing or unexpected:
            print(f"checkpoint compatibility: missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
    else:
        cfg = PretrainConfig(); cfg.epochs = 5; cfg.model.patch_ms = 125
        model = pretrain(cfg, device=device, n_sites=3, per_site=128)

    tr = SyntheticEEGDataset(256, site_id=10, seed=7)
    te = SyntheticEEGDataset(256, site_id=11, seed=8)
    res = fit_linear_probe(
        model, tr, te, cfg_n := 5, device,
        finetune=a.finetune, pool="spatial",
    )
    print("downstream:", res)


if __name__ == "__main__":
    main()
