"""Intra-cohort train/val/test protocol for ts-eeg-jepa.

People are first split into demographic COHORTS (e.g. young-male, old-female).
Everything stays INSIDE a cohort -- we do NOT train on one group and test on
another. For each cohort independently:

  * its subjects are split into disjoint train / val / test people
  * the JEPA encoder is pretrained on that cohort's TRAIN people
  * pretraining is monitored on that cohort's VAL people (loss + collapse)
  * each TEST person is calibrated (personal probe on their own data) and scored
    on their own held-out data

So a finding for "young males" is trained, validated, and tested entirely within
young males -- validated intra-group, never generalized across demographics.
Reported per cohort (mean +/- std over that cohort's held-out people).

Cohorts are defined by --cohort-by (sex, age, or both). With --cohort-by none
you get a single all-subjects cohort.

Run (venv python so torch + tseegjepa resolve; Dreyer2023 has sex+age):
  .venv/bin/python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
      --cohort-by sex age --split 0.6 0.2 0.2 --epochs 20 --device mps --pool chan
"""

from __future__ import annotations

import argparse
import os
import sys

# make `tseegjepa` importable without an editable install (run with any python
# that has torch; e.g. .venv/bin/python scripts/experiment.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from tseegjepa.config import PretrainConfig
from tseegjepa.data import collate_variable_montage
from tseegjepa.data.moabb_eeg import MoabbEEGDataset, load_moabb
from tseegjepa.jepa import EEGJepa
from tseegjepa.jepa_hier import HierarchicalEEGJepa
from tseegjepa.train.linear_probe import fit_linear_probe, fit_raw_baseline
from tseegjepa.train.pretrain import _ema_at, _lr_at, move, pick_device

# subjects with known acquisition issues (different sfreq / damaged runs)
PHYSIONET_BAD = {88, 89, 92, 100, 104}


# ----------------------------- splitting ----------------------------------
def _ratio_split(s, ratios):
    n = len(s)
    n_tr = max(1, int(round(ratios[0] * n)))
    n_va = max(1, int(round(ratios[1] * n)))
    n_va = min(n_va, max(0, n - n_tr - 1))
    return s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]


def cohort_labels(subjects, demo, cohort_by):
    """Map each subject -> a cohort label from the chosen demographic axes.

    cohort_by is a list e.g. ['sex'], ['age'], or ['sex','age']. age is binned by
    the GLOBAL median (so bins are consistent across cohorts). [] -> single 'all'.
    Subjects missing any requested attribute get label None (excluded).
    """
    if not cohort_by:
        return {s: "all" for s in subjects}
    med = None
    if "age" in cohort_by:
        ages = [demo.get(s, {}).get("age") for s in subjects]
        valid = [a for a in ages if a is not None]
        med = float(np.median(valid)) if valid else None

    def part(s):
        parts = []
        d = demo.get(s, {})
        for axis in cohort_by:
            if axis == "sex":
                v = d.get("sex", "U")
                if v == "U":
                    return None
                parts.append(f"sex={v}")
            elif axis == "age":
                a = d.get("age")
                if a is None or med is None:
                    return None
                parts.append(f"age<={med:.0f}" if a <= med else f"age>{med:.0f}")
        return "|".join(parts)

    return {s: part(s) for s in subjects}


def load_extra_datasets(specs, paradigm, sample_rate, norm):
    """Load extra MOABB datasets (NAME[:N]) as MoabbEEGDatasets for SSL pooling.

    Loaded PER SUBJECT and resilient: a subject that fails to download (network
    timeout, corrupt run, channel mismatch) is skipped, keeping the ones that
    succeeded. One flaky source/subject must not kill a multi-hour run.
    """
    import moabb.datasets as MD
    out = []
    for spec in specs:
        name, _, n = spec.partition(":")
        try:
            subs = (list(range(1, int(n) + 1)) if n
                    else list(getattr(MD, name)().subject_list))
        except Exception as e:
            print(f"  [skip dataset] {name}: {repr(e)[:100]}")
            continue
        print(f"  loading extra SSL dataset {name} ({len(subs)} subjects) ...")
        Xs, ys, metas, chn, ok = [], [], [], None, 0
        for s in subs:
            try:
                X, y, meta, ch_names, _ = load_moabb(
                    name, subjects=[s], paradigm_name=paradigm,
                    sample_rate=sample_rate)
            except Exception as e:
                print(f"    [skip {name} s{s}] {repr(e)[:80]}")
                continue
            if chn is None:
                chn = ch_names
            if ch_names != chn:                      # montage differs -> skip subject
                print(f"    [skip {name} s{s}] channel mismatch")
                continue
            Xs.append(X); ys.append(y); metas += meta; ok += 1
        if not Xs:
            print(f"  [skip dataset] {name}: no subjects loaded")
            continue
        import numpy as np
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        ds = MoabbEEGDataset(X, y, metas, chn, norm=norm)
        print(f"    {name}: {ok}/{len(subs)} subjects, trials={len(ds)} "
              f"ch={len(chn)} T={X.shape[-1]}")
        out.append(ds)
    return out


def split_within(subjects, ratios, seed):
    """Disjoint train/val/test split WITHIN a set of subjects (no leakage)."""
    rng = np.random.default_rng(seed)
    s = list(subjects); rng.shuffle(s)
    tr, va, te = _ratio_split(s, ratios)
    return sorted(tr), sorted(va), sorted(te)


# --------------------------- pretraining ----------------------------------
def _amp_ctx(amp, device):
    """bf16 autocast on cuda (halves activation memory, ~2x faster); else no-op."""
    import contextlib
    if amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def jepa_eval(model, ds, cfg, device, gen, amp=False, workers=0):
    """Mean JEPA pred-loss + collapse on an unseen (val) subject group."""
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_variable_montage,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
    model.eval()
    tot, nb, last = 0.0, 0, None
    for b in loader:
        with _amp_ctx(amp, device):
            out = model(move(b, device), generator=gen)
        tot += float(out["loss_pred"]); nb += 1; last = out
    cs = model.collapse_report(last) if last is not None else None
    return (tot / max(1, nb)), cs


def pretrain(model, train_ds, val_ds, cfg, device, amp=False, workers=0):
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate_variable_montage,
                        num_workers=workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(workers > 0))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    total = cfg.epochs * len(loader)
    warm = cfg.warmup_epochs * len(loader)
    step = 0
    best_val = float("inf")
    for ep in range(cfg.epochs):
        model.train(); run = 0.0
        for b in loader:
            b = move(b, device)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total, warm, cfg.lr)
            with _amp_ctx(amp, device):
                out = model(b, generator=gen)
            opt.zero_grad(set_to_none=True); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
            model.update_target(_ema_at(step, total, cfg.ema_base, cfg.ema_final))
            run += float(out["loss"].detach()); step += 1
        vloss, vcs = jepa_eval(model, val_ds, cfg, device, gen, amp, workers)
        best_val = min(best_val, vloss)
        print(f"  ep{ep:02d} train_loss={run/len(loader):.4f}  "
              f"val_pred={vloss:.4f} | val {vcs}")
    return best_val


# --------------------------- downstream -----------------------------------
def _calib_eval_split(ho: MoabbEEGDataset, calib_frac, seed):
    """Split ONE new subject's trials into (calibration, evaluation).

    Deployment model: a new person submits some labeled data -> we fit their
    PERSONAL probe on the calibration slice, then score the held-out evaluation
    slice. Uses sessions if available (calib=earliest session, eval=later) so
    eval is a genuinely later recording; else a fraction split.
    """
    sessions = sorted({m["session"] for m in ho.meta})
    if len(sessions) >= 2:
        cal_idx = [i for i, m in enumerate(ho.meta) if m["session"] == sessions[0]]
        ev_idx = [i for i, m in enumerate(ho.meta) if m["session"] != sessions[0]]
    else:
        n = len(ho); k = max(1, int(calib_frac * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        cal_idx, ev_idx = perm[:k], perm[k:]
    mk = lambda idx: MoabbEEGDataset(ho.X[idx], ho.y[idx], [ho.meta[i] for i in idx],
                                     ho.ch_names, norm=ho.norm, subgroup_by=ho.subgroup_by)
    return mk(cal_idx), mk(ev_idx)


def calibrate_per_subject(model, full, subjects, n_cls, device, pool,
                          calib_frac=0.5, raw=False, finetune=False):
    """For each NEW person: calibrate a personal probe, score their held-out data.

    Encoder is frozen (shared, pretrained). Only the per-subject linear probe is
    fit -> this is per-person calibration, NOT cross-subject generalization.
    Reported as mean +/- std over people (each validated on their own eval slice).
    """
    accs, raws = [], []
    for s in subjects:
        ho = full.subset_by_subject([s])
        if len(ho) < 12:
            continue
        cal, ev = _calib_eval_split(ho, calib_frac, seed=s)
        res = fit_linear_probe(model, cal, ev, n_cls, device, pool=pool,
                               finetune=finetune)
        accs.append(res["accuracy"])
        line = (f"  subj {s:>3d}: calib={len(cal):>3d} eval={len(ev):>3d}  "
                f"acc={res['accuracy']:.3f} bal={res['balanced_accuracy']:.3f}")
        if raw:
            rb = fit_raw_baseline(cal, ev, n_cls, device)
            raws.append(rb["accuracy"]); line += f"  (raw={rb['accuracy']:.3f})"
        print(line)
    out = {"subjects": subjects, "acc_mean": float(np.mean(accs)) if accs else 0.0,
           "acc_std": float(np.std(accs)) if accs else 0.0, "n": len(accs)}
    if raws:
        out["raw_mean"] = float(np.mean(raws))
    return out


# ------------------------------- main -------------------------------------
def build_cfg(a, n_train) -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.model.sample_rate = a.sample_rate
    cfg.model.patch_ms = a.patch_ms
    cfg.model.dim = a.dim; cfg.model.heads = a.heads; cfg.model.depth = a.depth
    cfg.model.dropout = a.dropout
    if a.mask_frac is not None:                  # harder pretext -> richer features
        cfg.mask.temporal_mask_frac = a.mask_frac
        cfg.mask.spatial_block_frac = a.mask_frac
    if a.n_target_blocks is not None:
        cfg.mask.n_target_blocks = a.n_target_blocks
    assert a.dim % a.heads == 0, "dim % heads != 0"
    cfg.epochs = a.epochs; cfg.batch_size = a.batch_size
    cfg.warmup_epochs = a.warmup_epochs if a.warmup_epochs is not None \
        else max(1, a.epochs // 10)
    cfg.lr = a.lr; cfg.seed = a.seed
    cfg.n_domains = n_train
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="PhysionetMI")
    p.add_argument("--paradigm", default="MotorImagery")
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--n-subjects", type=int, default=30, help="first N (minus bad)")
    p.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--cohort-by", nargs="*", default=[], choices=["sex", "age"],
                   help="demographic axes defining cohorts; each cohort is "
                        "trained+validated+tested entirely within itself. "
                        "e.g. --cohort-by sex age. empty = one all-subjects cohort")
    p.add_argument("--min-cohort", type=int, default=6,
                   help="skip cohorts with fewer subjects than this (need >=3 to "
                        "split train/val/test)")
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument("--patch-ms", type=float, default=125.0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-3, help="peak learning rate")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="encoder dropout; raise (0.1-0.2) to fight train-subject "
                        "overfitting (the train/val SSL-loss gap)")
    p.add_argument("--mask-frac", type=float, default=None,
                   help="fraction masked (temporal+spatial). Higher = harder pretext "
                        "= richer, less subject-specific features. default 0.5")
    p.add_argument("--n-target-blocks", type=int, default=None,
                   help="number of JEPA target blocks (default 4); more = harder")
    p.add_argument("--warmup-epochs", type=int, default=None,
                   help="LR warmup epochs (default: epochs//10). Raise to avoid the "
                        "post-warmup collapse spike on large SSL pools")
    p.add_argument("--pool", default="chan", choices=["mean", "chan"])
    p.add_argument("--norm", default="global", choices=["global", "perchan", "none"])
    p.add_argument("--calib-frac", type=float, default=0.5,
                   help="fraction of a new subject's data used to calibrate their "
                        "personal probe (ignored if subject has >=2 sessions)")
    p.add_argument("--pretrain-extra", nargs="*", default=[],
                   help="extra MOABB datasets pooled into SSL pretraining only "
                        "(labels ignored). Format NAME[:N]. e.g. PhysionetMI:40 "
                        "BNCI2014_001:9 . Downstream eval stays on --dataset.")
    p.add_argument("--pretrain-seconds", type=float, default=None,
                   help="crop all pretraining trials to this length (default: min "
                        "trial length across primary+extra datasets)")
    p.add_argument("--crop-align", default="end", choices=["end", "center", "start"],
                   help="which part of each trial to keep when cropping the SSL pool. "
                        "'end' = sustained MI (aligns datasets with different cue "
                        "offsets, e.g. IV-2a [2,6] vs Dreyer [0,5])")
    p.add_argument("--raw-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--hierarchical", action="store_true",
                   help="use Hierarchical EEG-JEPA (temporal pyramid, per-level "
                        "prediction) instead of the flat model")
    p.add_argument("--levels", type=int, default=3, help="hierarchy levels")
    p.add_argument("--pool-factor", type=int, default=2,
                   help="temporal pooling factor between hierarchy levels")
    p.add_argument("--finetune", action="store_true",
                   help="per-subject calibration unfreezes the encoder (fine-tune) "
                        "instead of frozen linear probe")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast on cuda: ~halves activation memory, ~2x faster")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader worker processes (keep the GPU fed)")
    p.add_argument("--save", default=None,
                   help="save per-cohort encoders (suffixed with the cohort label)")
    a = p.parse_args()
    device = pick_device(a.device)
    if a.finetune and a.hierarchical:
        print("[warn] --finetune not supported with --hierarchical -> frozen probe")
        a.finetune = False

    if a.subjects:
        subjects = a.subjects
    else:
        bad = PHYSIONET_BAD if a.dataset == "PhysionetMI" else set()
        subjects = [s for s in range(1, a.n_subjects + 1) if s not in bad]

    print(f"loading {a.dataset} subjects={subjects} ...")
    X, y, meta, ch_names, label_names = load_moabb(
        a.dataset, subjects=subjects, paradigm_name=a.paradigm,
        sample_rate=a.sample_rate)
    full = MoabbEEGDataset(X, y, meta, ch_names, norm=a.norm)
    n_cls = len(label_names)
    print(f"  trials={len(full)} ch={len(ch_names)} classes={label_names} "
          f"chance={1/n_cls:.3f}")

    # demographics from the loaded meta (Dreyer2023/BNCI carry sex+age)
    demo = {m["subject"]: {"sex": m.get("sex", "U"), "age": m.get("age")}
            for m in meta}

    # extra datasets for SSL pretraining (labels ignored); crop all to common T
    extras = load_extra_datasets(a.pretrain_extra, a.paradigm, a.sample_rate, a.norm)
    if extras:
        lengths = [X.shape[-1]] + [e.X.shape[-1] for e in extras]
        common_t = (int(a.pretrain_seconds * a.sample_rate)
                    if a.pretrain_seconds else min(lengths))
        extras = [e.with_crop(common_t, a.crop_align) for e in extras]
        print(f"  multi-dataset SSL: +{len(extras)} datasets, crop to {common_t} "
              f"samples ({common_t/a.sample_rate:.2f}s, align={a.crop_align})")
    else:
        common_t = None

    # build cohorts (each handled fully intra-group)
    labels = cohort_labels(subjects, demo, a.cohort_by)
    cohorts: dict[str, list] = {}
    for s in subjects:
        if labels[s] is None:
            continue
        cohorts.setdefault(labels[s], []).append(s)
    cohorts = {c: sorted(v) for c, v in sorted(cohorts.items())}
    if a.cohort_by and all(c == "all" for c in cohorts):
        print(f"  WARNING: dataset lacks {a.cohort_by} -> single cohort")
    print(f"\ncohorts ({a.cohort_by or 'all'}): "
          f"{ {c: len(v) for c, v in cohorts.items()} }")

    summary = {}
    for cohort, c_subs in cohorts.items():
        print("\n" + "#" * 70)
        print(f"# COHORT {cohort}  ({len(c_subs)} subjects: {c_subs})")
        print("#" * 70)
        if len(c_subs) < max(3, a.min_cohort):
            print(f"  skip: < min-cohort ({a.min_cohort})")
            continue

        train_s, val_s, test_s = split_within(c_subs, a.split, a.seed)
        print(f"  intra-cohort split (disjoint people):")
        print(f"    train({len(train_s)})={train_s} val({len(val_s)})={val_s} "
              f"test({len(test_s)})={test_s}")
        if not test_s:
            print("  skip: empty test split"); continue

        cfg = build_cfg(a, len(train_s))
        if a.hierarchical:
            model = HierarchicalEEGJepa(cfg, n_levels=a.levels,
                                        pool_factor=a.pool_factor).to(device)
        else:
            model = EEGJepa(cfg).to(device)
        # SSL pretraining pool = cohort train subjects (+ extra datasets, cropped)
        train_primary = full.subset_by_subject(train_s)
        if extras:
            train_pool = ConcatDataset(
                [train_primary.with_crop(common_t, a.crop_align)] + extras)
            print(f"  SSL pool: {len(train_primary)} primary + "
                  f"{sum(len(e) for e in extras)} extra = {len(train_pool)} trials")
        else:
            train_pool = train_primary
        print(f"  == pretrain on {cohort} train / monitor {cohort} val "
              f"({'hierarchical' if a.hierarchical else 'flat'}) ==")
        pretrain(model, train_pool, full.subset_by_subject(val_s), cfg, device,
                 amp=a.amp, workers=a.workers)

        print(f"  == {cohort} TEST: per-subject calibration ==")
        tres = calibrate_per_subject(model, full, test_s, n_cls, device, a.pool,
                                     a.calib_frac, a.raw_baseline, a.finetune)
        print(f"  COHORT {cohort} acc = {tres['acc_mean']:.3f} +- "
              f"{tres['acc_std']:.3f} (n={tres['n']}, chance={1/n_cls:.3f})")
        summary[cohort] = tres
        if a.save:
            path = a.save.replace(".pt", f"_{cohort.replace('|','_').replace('=','')}.pt")
            torch.save({"cfg": cfg, "state_dict": model.state_dict(),
                        "cohort": cohort,
                        "splits": {"train": train_s, "val": val_s, "test": test_s}}, path)
            print(f"  saved -> {path}")

    print("\n" + "=" * 70 + "\nINTRA-COHORT SUMMARY (chance "
          f"{1/n_cls:.3f})\n" + "=" * 70)
    for c, r in summary.items():
        extra = f"  raw={r['raw_mean']:.3f}" if "raw_mean" in r else ""
        print(f"  {c:22s} acc={r['acc_mean']:.3f} +- {r['acc_std']:.3f} "
              f"(n={r['n']}){extra}")
    print("\nEach cohort trained+validated+tested within itself; no cross-group "
          "transfer. Per-person calibration inside each cohort.")


if __name__ == "__main__":
    main()
