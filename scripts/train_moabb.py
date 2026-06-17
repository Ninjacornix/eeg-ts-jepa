"""Pretrain EEG-JEPA on a small MOABB dataset, then evaluate.

Default: BNCI2014_001 (motor imagery, 22 EEG ch, 4 classes). Use a few subjects
for a small, fast run.  Self-supervised pretraining ignores labels; the linear
probe and leave-one-SUBJECT-out evaluation use them.

  # quick: 3 subjects, pretrain on 2, probe held-out subject
  python scripts/train_moabb.py --subjects 1 2 3 --epochs 15

  # cross-subject generalization (LODO over subjects)
  python scripts/train_moabb.py --subjects 1 2 3 4 --epochs 20 --lodo
"""

import argparse

import torch

from tseegjepa.config import PretrainConfig
from tseegjepa.data.moabb_eeg import MoabbEEGDataset, load_moabb
from tseegjepa.jepa import EEGJepa
from tseegjepa.train.linear_probe import fit_linear_probe, fit_raw_baseline
from tseegjepa.train.pretrain import (
    _ema_at, _lr_at, move, pick_device,
)
from tseegjepa.data import collate_variable_montage
from torch.utils.data import DataLoader


def build_cfg(args, n_subjects, n_classes) -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.model.sample_rate = args.sample_rate
    cfg.model.patch_ms = args.patch_ms
    cfg.model.dim = args.dim
    cfg.model.heads = args.heads
    assert args.dim % args.heads == 0, "dim must be divisible by heads"
    cfg.model.depth = args.depth
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.warmup_epochs = max(1, args.epochs // 10)
    cfg.use_domain_invariance = args.domain_inv
    cfg.n_domains = n_subjects
    return cfg


def pretrain_on(model, ds, cfg, device, verbose=True):
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate_variable_montage)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    total = cfg.epochs * len(loader)
    warm = cfg.warmup_epochs * len(loader)
    step = 0
    for ep in range(cfg.epochs):
        model.train()
        run = 0.0
        for batch in loader:
            batch = move(batch, device)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total, warm, cfg.lr)
            out = model(batch, generator=gen)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            model.update_target(_ema_at(step, total, cfg.ema_base, cfg.ema_final))
            run += float(out["loss"].detach())
            step += 1
        if verbose:
            cs = model.collapse_report(out)
            v = f" var={float(out['loss_var']):.3f}" if "loss_var" in out else ""
            cov = f" cov={float(out['loss_cov']):.3f}" if "loss_cov" in out else ""
            print(f"  [pretrain ep{ep:02d}] loss={run/len(loader):.4f}{v}{cov} | {cs}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="BNCI2014_001")
    p.add_argument("--paradigm", default="MotorImagery")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument("--tmax", type=float, default=None, help="crop trial to N seconds")
    p.add_argument("--patch-ms", type=float, default=250.0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--domain-inv", action="store_true")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--pool", default="chan", choices=["mean", "chan"],
                   help="downstream pooling: 'chan' keeps per-channel structure "
                        "(better for motor imagery / lateralized tasks)")
    p.add_argument("--norm", default="global", choices=["global", "perchan", "none"],
                   help="signal norm; 'global' preserves inter-channel band-power (MI)")
    p.add_argument("--subgroup", default="sex", choices=["sex", "age", "session"],
                   help="axis for disaggregated (fairness) metrics")
    p.add_argument("--probe-pool", action="store_true",
                   help="probe on ALL subjects pooled (sexes mixed) so subgroup gap "
                        "is meaningful; pretrain still excludes held-out subject")
    p.add_argument("--raw-baseline", action="store_true",
                   help="also fit a log-bandpower baseline (no model) = achievable ceiling")
    p.add_argument("--lodo", action="store_true", help="leave-one-subject-out loop")
    p.add_argument("--device", default="auto")
    p.add_argument("--save", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"loading MOABB {args.dataset} subjects={args.subjects} ...")
    X, y, meta, ch_names, label_names = load_moabb(
        args.dataset, subjects=args.subjects, paradigm_name=args.paradigm,
        n_classes=args.n_classes, sample_rate=args.sample_rate, tmax=args.tmax,
    )
    full = MoabbEEGDataset(X, y, meta, ch_names, norm=args.norm,
                           subgroup_by=args.subgroup)
    print(f"subgroup axis '{args.subgroup}': {full.subgroup_names}")
    n_cls = len(label_names)
    print(f"  trials={len(full)} channels={len(ch_names)} classes={label_names} "
          f"T={X.shape[-1]} ({X.shape[-1]/args.sample_rate:.1f}s)")
    cfg = build_cfg(args, len(args.subjects), n_cls)

    if not args.lodo:
        # pretrain on all-but-last subject, probe last
        held = args.subjects[-1]
        train_subs = args.subjects[:-1] or args.subjects
        print(f"pretrain on {train_subs}, probe held-out subject {held}")
        model = EEGJepa(cfg).to(device)
        pretrain_on(model, full.subset_by_subject(train_subs), cfg, device)
        # probe set: held-out subject (one sex), or ALL subjects pooled so the
        # subgroup gap spans both sexes (--probe-pool).
        probe_src = full if args.probe_pool else full.subset_by_subject([held])
        if args.probe_pool:
            print(f"probe on ALL subjects pooled (subgroup gap is meaningful)")
        n = len(probe_src); k = int(0.7 * n)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        tr = _index(probe_src, idx[:k]); te = _index(probe_src, idx[k:])
        if args.raw_baseline:
            rb = fit_raw_baseline(tr, te, n_cls, device)
            print(f"raw log-bandpower baseline: acc={rb['accuracy']:.3f} "
                  f"bal={rb['balanced_accuracy']:.3f}  (chance={1/n_cls:.3f})")
        res = fit_linear_probe(model, tr, te, n_cls, device, finetune=args.finetune, pool=args.pool)
        print("downstream:", {k_: v for k_, v in res.items() if k_ != "subgroup_accuracy"})
        print("subgroup acc:", res["subgroup_accuracy"])
        if args.save:
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, args.save)
            print("saved ->", args.save)
        return

    accs = []
    for held in args.subjects:
        train_subs = [s for s in args.subjects if s != held]
        print(f"\n=== held-out subject {held} (pretrain on {train_subs}) ===")
        model = EEGJepa(cfg).to(device)
        pretrain_on(model, full.subset_by_subject(train_subs), cfg, device, verbose=False)
        ho = full.subset_by_subject([held])
        n = len(ho); k = int(0.7 * n)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(held))
        tr = _index(ho, idx[:k]); te = _index(ho, idx[k:])
        res = fit_linear_probe(model, tr, te, n_cls, device, finetune=args.finetune, pool=args.pool)
        accs.append(res["accuracy"])
        print(f"  acc={res['accuracy']:.3f} bal={res['balanced_accuracy']:.3f} "
              f"subgroup_gap={res['subgroup_gap']:.3f}")
    print(f"\nLOSO mean acc={sum(accs)/len(accs):.3f} worst={min(accs):.3f} "
          f"(chance={1/n_cls:.3f})")


def _index(ds: MoabbEEGDataset, idx) -> MoabbEEGDataset:
    idx = idx.tolist()
    return MoabbEEGDataset(ds.X[idx], ds.y[idx], [ds.meta[i] for i in idx],
                           ds.ch_names, norm=ds.norm, subgroup_by=ds.subgroup_by)


if __name__ == "__main__":
    main()
