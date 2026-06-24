"""Subject-disjoint pretraining and downstream evaluation for ts-eeg-jepa.

People are first split into demographic COHORTS (e.g. young-male, old-female).
Everything stays INSIDE a cohort -- we do NOT train on one group and test on
another. For each cohort independently:

  * its subjects are split into disjoint train / val / test people
  * the JEPA encoder is pretrained on that cohort's TRAIN people
  * pretraining is monitored on that cohort's VAL people (loss + collapse)
  * cross-subject mode fits one shared decoder on labeled TRAIN people, selects
    it on VAL people, and directly scores TEST people with zero test-person labels
  * calibration mode remains available as a secondary personalized evaluation

So a finding for "young males" is trained, validated, and tested entirely within
young males -- validated intra-group, never generalized across demographics.
The primary metric is mean per-subject balanced accuracy on unseen test people.

Cohorts are defined by --cohort-by (sex, age, or both). With --cohort-by none
you get a single all-subjects cohort.

Run (venv python so torch + tseegjepa resolve; Dreyer2023 has sex+age):
  .venv/bin/python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
      --cohort-by sex age --split 0.6 0.2 0.2 --epochs 20 --device mps --pool spatial
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# make `tseegjepa` importable without an editable install (run with any python
# that has torch; e.g. .venv/bin/python scripts/experiment.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from tseegjepa.config import PretrainConfig
from tseegjepa.data.moabb_eeg import MoabbEEGDataset, load_moabb
from tseegjepa.eval.mdfb import (
    compare_bands,
    decoder_spectral_saliency,
    mdfb_like,
)
from tseegjepa.jepa import EEGJepa
from tseegjepa.jepa_hier import HierarchicalEEGJepa
from tseegjepa.train.checkpoint import load_checkpoint, save_checkpoint
from tseegjepa.train.engine import PretrainTrainer
from tseegjepa.train.linear_probe import (
    fit_cross_subject_probe,
    fit_cross_subject_raw_baseline,
    fit_linear_probe,
    fit_raw_baseline,
)
from tseegjepa.train.pretrain import pick_device

# subjects with known acquisition issues (different sfreq / damaged runs)
PHYSIONET_BAD = {88, 89, 92, 100, 104}


# ----------------------------- splitting ----------------------------------
def _frequency_pairs(values, name):
    if len(values) % 2:
        raise ValueError(f"{name} requires LOW HIGH pairs")
    pairs = tuple(
        (float(values[i]), float(values[i + 1]))
        for i in range(0, len(values), 2)
    )
    if any(lo >= hi for lo, hi in pairs):
        raise ValueError(f"{name} bands must satisfy LOW < HIGH")
    return pairs


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


def load_extra_datasets(specs, paradigm, sample_rate, norm, bandpass):
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
                    sample_rate=sample_rate,
                    fmin=bandpass[0], fmax=bandpass[1])
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


def split_within(subjects, ratios, seed, strata=None):
    """Disjoint train/val/test split, optionally preserving collection domains."""
    rng = np.random.default_rng(seed)
    if strata is None:
        s = list(subjects)
        rng.shuffle(s)
        tr, va, te = _ratio_split(s, ratios)
        return sorted(tr), sorted(va), sorted(te)

    missing = [s for s in subjects if not strata.get(s)]
    if missing:
        raise ValueError(f"missing split stratum for subjects {missing}")
    groups = {}
    for subject in subjects:
        groups.setdefault(strata[subject], []).append(subject)
    train, val, test = [], [], []
    for name in sorted(groups):
        group = groups[name]
        rng.shuffle(group)
        tr, va, te = _ratio_split(group, ratios)
        train.extend(tr)
        val.extend(va)
        test.extend(te)
    return sorted(train), sorted(val), sorted(test)


def _domain_counts(subjects, domains):
    counts = {}
    for subject in subjects:
        name = domains.get(subject)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _attach_domain_metrics(result, subject_lookup, domains):
    grouped = {}
    for local_id, metrics in result["per_subject"].items():
        actual = subject_lookup.get(int(local_id), int(local_id))
        domain = domains.get(actual)
        if domain:
            grouped.setdefault(domain, []).append((actual, metrics))
    summaries = {}
    for domain, rows in sorted(grouped.items()):
        acc = np.asarray([m["accuracy"] for _, m in rows], dtype=float)
        bal = np.asarray([m["balanced_accuracy"] for _, m in rows], dtype=float)
        summaries[domain] = {
            "subjects": [s for s, _ in rows],
            "n_subjects": len(rows),
            "subject_accuracy_mean": float(acc.mean()),
            "subject_accuracy_std": float(acc.std()),
            "subject_balanced_accuracy_mean": float(bal.mean()),
            "subject_balanced_accuracy_std": float(bal.std()),
        }
    result["subdataset_metrics"] = summaries
    return summaries


def _augment_summary(cfg: PretrainConfig) -> dict:
    aug = cfg.augment
    return {
        "enabled": aug.enabled,
        "crop_jitter_ms": aug.crop_jitter_ms,
        "time_jitter_ms": aug.time_jitter_ms,
        "amplitude_jitter": aug.amplitude_jitter,
        "gaussian_noise": aug.gaussian_noise,
        "channel_dropout": aug.channel_dropout,
        "min_channels": aug.min_channels,
        "freq_mask_prob": aug.freq_mask_prob,
        "freq_mask_width_hz": aug.freq_mask_width_hz,
        "freq_mask_range_hz": [aug.freq_mask_fmin, aug.freq_mask_fmax],
    }


# --------------------------- pretraining ----------------------------------
def pretrain(model, train_ds, val_ds, cfg, device, amp=False, workers=0,
             patience=0):
    """Shared collapse-aware training engine."""
    trainer = PretrainTrainer(model, cfg, device, amp=amp, workers=workers)
    history = trainer.fit(train_ds, val_ds, patience=patience)
    valid = [h for h in history if h.selection_score is not None]
    return min(valid, key=lambda h: h.selection_score).val_pred if valid else None


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


def _cap_dataset(ds: MoabbEEGDataset, n: int, seed: int) -> MoabbEEGDataset:
    """Stratified subsample of a calibration set to ~n trials (low-label regime)."""
    y = ds.y
    rng = np.random.default_rng(seed)
    classes = sorted(set(int(v) for v in y))
    per = max(1, n // len(classes))
    idx = []
    for c in classes:
        ci = [i for i in range(len(y)) if int(y[i]) == c]
        rng.shuffle(ci)
        idx += ci[:per]
    rng.shuffle(idx)
    idx = idx[:n]
    return MoabbEEGDataset(ds.X[idx], ds.y[idx], [ds.meta[i] for i in idx],
                           ds.ch_names, norm=ds.norm, subgroup_by=ds.subgroup_by)


def calibrate_per_subject(model, full, subjects, n_cls, device, pool,
                          calib_frac=0.5, raw=False, finetune=False, probe_seed=0,
                          calib_cap=None):
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
        if calib_cap is not None:                     # low-label regime
            cal = _cap_dataset(cal, calib_cap, seed=s)
        # per-subject seed, offset by --probe-seed so re-runs are fresh attempts
        res = fit_linear_probe(model, cal, ev, n_cls, device, pool=pool,
                               finetune=finetune, seed=1000 * probe_seed + s)
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


def evaluate_cross_subject(
    model,
    full,
    train_subjects,
    val_subjects,
    test_subjects,
    n_cls,
    device,
    pool,
    probe_epochs,
    probe_lr,
    probe_patience,
    probe_seed,
    raw_baseline=False,
    subject_domains=None,
    mdfb_analysis=False,
    mdfb_max_trials=80,
):
    """One decoder: labeled train people -> validation people -> unseen test people."""
    train_ds = full.subset_by_subject(train_subjects)
    val_ds = full.subset_by_subject(val_subjects)
    test_ds = full.subset_by_subject(test_subjects)
    print(
        f"  shared decoder: train={len(train_ds)} trials/{len(train_subjects)} subjects, "
        f"val={len(val_ds)}/{len(val_subjects)}, test={len(test_ds)}/{len(test_subjects)}"
    )
    result = fit_cross_subject_probe(
        model, train_ds, val_ds, test_ds, n_cls, device,
        epochs=probe_epochs, lr=probe_lr, patience=probe_patience,
        pool=pool, seed=probe_seed, verbose=True,
        return_artifact=mdfb_analysis,
    )
    decoder_artifact = result.pop("_decoder_artifact", None)
    subject_lookup = {i: s for i, s in enumerate(sorted(test_subjects))}
    print("  unseen test subjects (zero calibration labels):")
    for local_id, metrics in sorted(result["per_subject"].items()):
        actual = subject_lookup.get(local_id, local_id)
        print(
            f"    subj {actual:>3}: n={metrics['n_trials']:>3} "
            f"acc={metrics['accuracy']:.3f} "
            f"bal={metrics['balanced_accuracy']:.3f}"
        )
    print(
        f"  JEPA zero-shot: subject-bal="
        f"{result['subject_balanced_accuracy_mean']:.3f}+-"
        f"{result['subject_balanced_accuracy_std']:.3f} "
        f"pooled-bal={result['balanced_accuracy']:.3f} "
        f"pooled-acc={result['accuracy']:.3f}"
    )

    domain_metrics = {}
    if subject_domains:
        domain_metrics = _attach_domain_metrics(
            result, subject_lookup, subject_domains
        )

    raw_domain_metrics = {}
    if raw_baseline:
        raw = fit_cross_subject_raw_baseline(
            train_ds, val_ds, test_ds, n_cls, device,
            epochs=probe_epochs, lr=probe_lr, patience=probe_patience,
            seed=probe_seed, verbose=False,
        )
        if subject_domains:
            raw_domain_metrics = _attach_domain_metrics(
                raw, subject_lookup, subject_domains
            )
        result["band_power"] = raw
        delta = (
            result["subject_balanced_accuracy_mean"]
            - raw["subject_balanced_accuracy_mean"]
        )
        print(
            f"  band-power zero-shot: subject-bal="
            f"{raw['subject_balanced_accuracy_mean']:.3f}+-"
            f"{raw['subject_balanced_accuracy_std']:.3f} "
            f"delta={delta:+.3f}"
        )
    if domain_metrics:
        print("  zero-shot balanced accuracy by Dreyer subdataset:")
        for domain, metrics in domain_metrics.items():
            raw_metrics = raw_domain_metrics.get(domain)
            raw_text = (
                f" band-power={raw_metrics['subject_balanced_accuracy_mean']:.3f}"
                if raw_metrics else ""
            )
            print(
                f"    {domain}: n={metrics['n_subjects']} "
                f"JEPA={metrics['subject_balanced_accuracy_mean']:.3f}"
                f"{raw_text}"
            )
    if decoder_artifact is not None:
        print(
            "  post-hoc MDFB agreement (acquisition-run labels are diagnostic "
            "only; they do not fit the zero-shot decoder):"
        )
        per_subject_alignment = {}
        for subject in test_subjects:
            subject_ds = full.subset_by_subject([subject])
            acquisition_idx = [
                i for i, meta in enumerate(subject_ds.meta)
                if "acquisition" in meta.get("run", "").lower()
            ]
            if not acquisition_idx:
                print(f"    subj {subject:>3}: skipped (no acquisition-run metadata)")
                continue
            acquisition_idx = acquisition_idx[:mdfb_max_trials]
            acquisition = MoabbEEGDataset(
                subject_ds.X[acquisition_idx],
                subject_ds.y[acquisition_idx],
                [subject_ds.meta[i] for i in acquisition_idx],
                subject_ds.ch_names,
                norm=subject_ds.norm,
                subgroup_by=subject_ds.subgroup_by,
            )
            try:
                mdfb = mdfb_like(
                    acquisition.X,
                    acquisition.y,
                    acquisition.ch_names,
                    model.cfg.model.sample_rate,
                    fmin=model.cfg.model.spectral_fmin,
                    fmax=model.cfg.model.spectral_fmax,
                )
                learned = decoder_spectral_saliency(
                    model,
                    acquisition,
                    decoder_artifact,
                    device,
                    pool,
                    model.cfg.model.sample_rate,
                    fmin=model.cfg.model.spectral_fmin,
                    fmax=model.cfg.model.spectral_fmax,
                    max_trials=mdfb_max_trials,
                )
                comparison = compare_bands(mdfb, learned)
            except (ValueError, RuntimeError) as exc:
                print(f"    subj {subject:>3}: skipped ({exc})")
                continue
            per_subject_alignment[str(subject)] = {
                "mdfb": mdfb,
                "decoder_saliency": learned,
                **comparison,
            }
            print(
                f"    subj {subject:>3}: MDFB={mdfb['low_hz']:.1f}-"
                f"{mdfb['high_hz']:.1f}Hz decoder={learned['low_hz']:.1f}-"
                f"{learned['high_hz']:.1f}Hz peak={learned['peak_hz']:.1f}Hz "
                f"distance={comparison['peak_distance_hz']:.1f}Hz"
            )
        if per_subject_alignment:
            rows = list(per_subject_alignment.values())
            result["mdfb_alignment"] = {
                "diagnostic_only": True,
                "n_subjects": len(rows),
                "peak_in_mdfb_rate": float(np.mean([
                    row["learned_peak_in_mdfb"] for row in rows
                ])),
                "peak_distance_hz_mean": float(np.mean([
                    row["peak_distance_hz"] for row in rows
                ])),
                "band_iou_mean": float(np.mean([
                    row["band_iou"] for row in rows
                ])),
                "per_subject": per_subject_alignment,
            }
    result["test_subject_lookup"] = subject_lookup
    return result


# ------------------------------- main -------------------------------------
def build_cfg(a, n_train) -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.model.sample_rate = a.sample_rate
    cfg.model.patch_ms = a.patch_ms
    cfg.model.dim = a.dim; cfg.model.heads = a.heads; cfg.model.depth = a.depth
    cfg.model.dropout = a.dropout
    cfg.model.input_mode = a.input_mode
    cfg.model.spectral_frontend = a.spectral_frontend
    cfg.model.spectral_window_ms = a.spectral_window_ms
    cfg.model.spectral_fmin, cfg.model.spectral_fmax = a.spectral_range
    cfg.var_reg = a.var_reg                       # anti-collapse strength
    cfg.cov_reg = a.cov_reg
    cfg.spectral_aux = a.spectral_aux             # MI-aware band-power pretext
    cfg.spectral_aux_bands = _frequency_pairs(
        a.spectral_aux_bands, "--spectral-aux-bands"
    )
    cfg.augment.enabled = a.augment
    cfg.augment.crop_jitter_ms = a.aug_crop_jitter_ms
    cfg.augment.time_jitter_ms = a.aug_time_jitter_ms
    cfg.augment.amplitude_jitter = a.aug_amplitude_jitter
    cfg.augment.gaussian_noise = a.aug_gaussian_noise
    cfg.augment.channel_dropout = a.aug_channel_dropout
    cfg.augment.min_channels = a.aug_min_channels
    cfg.augment.freq_mask_prob = a.aug_freq_mask_prob
    cfg.augment.freq_mask_width_hz = a.aug_freq_mask_width_hz
    cfg.augment.freq_mask_fmin, cfg.augment.freq_mask_fmax = a.aug_freq_mask_range
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
    p.add_argument(
        "--split-stratify",
        default="auto",
        choices=["auto", "none", "subdataset"],
        help="preserve acquisition-domain proportions in subject splits. "
             "'auto' uses Dreyer2023 A/B/C subdatasets and otherwise no strata",
    )
    p.add_argument("--cohort-by", nargs="*", default=[], choices=["sex", "age"],
                   help="demographic axes defining cohorts; each cohort is "
                        "trained+validated+tested entirely within itself. "
                        "e.g. --cohort-by sex age. empty = one all-subjects cohort")
    p.add_argument("--min-cohort", type=int, default=6,
                   help="skip cohorts with fewer subjects than this (need >=3 to "
                        "split train/val/test)")
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument(
        "--bandpass",
        type=float,
        nargs=2,
        default=[0.5, 45.0],
        metavar=("LOW_HZ", "HIGH_HZ"),
        help="explicit MOABB preprocessing filter. Use 0.5 45 for broadband "
             "or 5 35 for the MI-prior condition. Previous implicit MOABB "
             "defaults were 8 32",
    )
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
    p.add_argument("--var-reg", type=float, default=1.0,
                   help="VICReg variance weight (anti-collapse). Raise (2-4) if the "
                        "EMA target collapses: val_pred -> 0, std -> 0")
    p.add_argument("--cov-reg", type=float, default=1.0,
                   help="VICReg covariance weight (decorrelate dims, fights rank "
                        "collapse). Raise (2-4) alongside --var-reg")
    p.add_argument(
        "--input-mode",
        default="raw",
        choices=["raw", "fft", "both", "spectral"],
        help="per-patch input to the encoder: time-domain (raw), rFFT "
             "log-magnitude (fft), both, or only the selected long-window "
             "spectral frontend",
    )
    p.add_argument(
        "--spectral-frontend",
        default="none",
        choices=["none", "filterbank", "learned"],
        help="none gives true raw/patch-FFT input; filterbank uses fixed MI bands; "
             "learned projects long-window FFT bins",
    )
    p.add_argument(
        "--spectral-window-ms",
        type=float,
        default=2000.0,
        help="context window for filterbank/learned spectra and spectral targets",
    )
    p.add_argument(
        "--spectral-range",
        type=float,
        nargs=2,
        default=[5.0, 35.0],
        metavar=("LOW_HZ", "HIGH_HZ"),
        help="frequency range used by the learned spectral frontend",
    )
    p.add_argument("--spectral-aux", type=float, default=0.0,
                   help="weight of the spectral auxiliary: also predict log "
                        "band-power of masked long-window targets")
    p.add_argument(
        "--spectral-aux-bands",
        type=float,
        nargs="+",
        default=[8.0, 13.0, 13.0, 30.0],
        metavar="HZ",
        help="LOW HIGH pairs for auxiliary targets; default is mu and beta",
    )
    p.add_argument("--mask-frac", type=float, default=None,
                   help="fraction masked (temporal+spatial). Higher = harder pretext "
                        "= richer, less subject-specific features. default 0.5")
    p.add_argument("--n-target-blocks", type=int, default=None,
                   help="number of JEPA target blocks (default 4); more = harder")
    p.add_argument("--warmup-epochs", type=int, default=None,
                   help="LR warmup epochs (default: epochs//10). Raise to avoid the "
                        "post-warmup collapse spike on large SSL pools")
    p.add_argument("--patience", type=int, default=0,
                   help="stop pretraining after N epochs without val_pred improvement "
                        "(0 = train all epochs). Best-val weights are saved regardless")
    p.add_argument("--pool", default="spatial", choices=["mean", "spatial", "chan"])
    p.add_argument(
        "--eval-protocol",
        default="calibration",
        choices=["cross-subject", "calibration"],
        help="cross-subject = one shared decoder and zero test-subject labels; "
             "calibration = personalized decoder using labels from each test subject",
    )
    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-lr", type=float, default=3e-3)
    p.add_argument("--probe-patience", type=int, default=25)
    p.add_argument("--norm", default="global", choices=["global", "perchan", "none"])
    p.add_argument("--calib-frac", type=float, default=0.5,
                   help="fraction of a new subject's data used to calibrate their "
                        "personal probe (ignored if subject has >=2 sessions)")
    p.add_argument("--calib-trials", type=int, nargs="*", default=None,
                   metavar="N",
                   help="LOW-LABEL SWEEP: cap calibration to each N (e.g. 5 10 20 "
                        "40), comparing JEPA vs band-power per label budget. "
                        "Shows label efficiency = where SSL pretraining wins")
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
    p.add_argument(
        "--augment",
        action="store_true",
        help="enable train-only SSL augmentations. Also auto-enabled when any "
             "--aug-* corruption knob is non-zero",
    )
    p.add_argument(
        "--aug-crop-jitter-ms",
        type=float,
        default=0.0,
        help="randomize SSL crop start by +/- this many ms around --crop-align",
    )
    p.add_argument(
        "--aug-time-jitter-ms",
        type=float,
        default=0.0,
        help="random circular shift inside each valid trial window",
    )
    p.add_argument(
        "--aug-amplitude-jitter",
        type=float,
        default=0.0,
        help="log-normal per-trial gain std; try 0.05-0.15",
    )
    p.add_argument(
        "--aug-gaussian-noise",
        type=float,
        default=0.0,
        help="additive noise std as a fraction of each trial std; try 0.01-0.03",
    )
    p.add_argument(
        "--aug-channel-dropout",
        type=float,
        default=0.0,
        help="probability of hiding a valid electrode during SSL; try 0.03-0.10",
    )
    p.add_argument(
        "--aug-min-channels",
        type=int,
        default=4,
        help="minimum valid electrodes retained after channel dropout",
    )
    p.add_argument(
        "--aug-freq-mask-prob",
        type=float,
        default=0.0,
        help="probability of applying one narrow random FFT band-stop mask",
    )
    p.add_argument(
        "--aug-freq-mask-width-hz",
        type=float,
        default=2.0,
        help="width of the random frequency mask in Hz",
    )
    p.add_argument(
        "--aug-freq-mask-range",
        type=float,
        nargs=2,
        default=[5.0, 35.0],
        metavar=("LOW_HZ", "HIGH_HZ"),
        help="range from which frequency-mask centers are sampled",
    )
    p.add_argument("--raw-baseline", action="store_true")
    p.add_argument(
        "--mdfb-analysis",
        action="store_true",
        help="post-hoc diagnostic: compare label-free shared-decoder frequency "
             "saliency with an MDFB-like band computed from each test subject's "
             "first two acquisition runs. Does not tune the model or decoder",
    )
    p.add_argument(
        "--mdfb-max-trials",
        type=int,
        default=80,
        help="maximum acquisition trials per test subject used by MDFB diagnostics",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-seed", type=int, default=None,
                   help="seed for the shared decoder or personalized probes; "
                        "default: 0, or the checkpoint value with --load")
    p.add_argument("--device", default="auto")
    p.add_argument("--hierarchical", action="store_true",
                   help="use Hierarchical EEG-JEPA (temporal pyramid, per-level "
                        "prediction) instead of the flat model")
    p.add_argument("--levels", type=int, default=3, help="hierarchy levels")
    p.add_argument("--pool-factor", type=int, default=2,
                   help="temporal pooling factor between hierarchy levels")
    p.add_argument("--finetune", action="store_true",
                   help="fine-tune an isolated encoder copy per subject")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast on cuda: ~halves activation memory, ~2x faster")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader worker processes (keep the GPU fed)")
    p.add_argument("--save", default=None,
                   help="save per-cohort encoders (suffixed with the cohort label)")
    p.add_argument("--load", default=None,
                   help="load a pretrained encoder and stored subject split; skip "
                        "SSL pretraining and run the selected downstream protocol")
    a = p.parse_args()
    try:
        _frequency_pairs(a.spectral_aux_bands, "--spectral-aux-bands")
    except ValueError as exc:
        p.error(str(exc))
    aug_knobs = [
        a.aug_crop_jitter_ms,
        a.aug_time_jitter_ms,
        a.aug_amplitude_jitter,
        a.aug_gaussian_noise,
        a.aug_channel_dropout,
        a.aug_freq_mask_prob,
    ]
    if any(v > 0 for v in aug_knobs):
        a.augment = True
    if a.augment and a.load:
        print("  NOTE: --load skips SSL pretraining, so --augment has no effect")
    if a.eval_protocol == "cross-subject" and a.calib_trials:
        p.error("--calib-trials belongs to calibration mode, not cross-subject mode")
    if a.eval_protocol == "cross-subject" and a.finetune:
        p.error(
            "--finetune is not part of the frozen-representation cross-subject "
            "benchmark; omit it"
        )
    if a.mdfb_analysis and a.eval_protocol != "cross-subject":
        p.error("--mdfb-analysis requires --eval-protocol cross-subject")
    device = pick_device(a.device)
    if a.subjects:
        subjects = a.subjects
    else:
        bad = PHYSIONET_BAD if a.dataset == "PhysionetMI" else set()
        subjects = [s for s in range(1, a.n_subjects + 1) if s not in bad]

    print(f"loading {a.dataset} subjects={subjects} ...")
    X, y, meta, ch_names, label_names = load_moabb(
        a.dataset, subjects=subjects, paradigm_name=a.paradigm,
        sample_rate=a.sample_rate, fmin=a.bandpass[0], fmax=a.bandpass[1])
    full = MoabbEEGDataset(X, y, meta, ch_names, norm=a.norm)
    n_cls = len(label_names)
    print(f"  trials={len(full)} ch={len(ch_names)} classes={label_names} "
          f"chance={1/n_cls:.3f}")

    # demographics from the loaded meta (Dreyer2023/BNCI carry sex+age)
    demo = {
        m["subject"]: {
            "sex": m.get("sex", "U"),
            "age": m.get("age"),
            "subdataset": m.get("subdataset"),
            "experimenter_sex": m.get("experimenter_sex", "U"),
        }
        for m in meta
    }
    split_stratify = a.split_stratify
    if split_stratify == "auto":
        split_stratify = (
            "subdataset"
            if all(demo.get(s, {}).get("subdataset") for s in subjects)
            else "none"
        )
    subject_domains = {}
    if split_stratify == "subdataset":
        missing_domains = [
            s for s in subjects if not demo.get(s, {}).get("subdataset")
        ]
        if missing_domains:
            p.error(
                "--split-stratify subdataset requires subdataset metadata for "
                f"every subject; missing {missing_domains}"
            )
        subject_domains = {s: demo[s]["subdataset"] for s in subjects}

    # extra datasets for SSL pretraining (labels ignored); crop all to common T
    extras = (
        [] if a.load
        else load_extra_datasets(
            a.pretrain_extra, a.paradigm, a.sample_rate, a.norm, a.bandpass
        )
    )
    if extras:
        lengths = [X.shape[-1]] + [e.X.shape[-1] for e in extras]
        common_t = (int(a.pretrain_seconds * a.sample_rate)
                    if a.pretrain_seconds else min(lengths))
        crop_jitter = int(round(a.aug_crop_jitter_ms * a.sample_rate / 1000.0))
        extras = [
            e.with_crop(common_t, a.crop_align, crop_jitter=crop_jitter)
            for e in extras
        ]
        print(f"  multi-dataset SSL: +{len(extras)} datasets, crop to {common_t} "
              f"samples ({common_t/a.sample_rate:.2f}s, align={a.crop_align})")
        if crop_jitter:
            print(
                f"  SSL crop jitter: +/-{crop_jitter} samples "
                f"({a.aug_crop_jitter_ms:g} ms)"
            )
    elif a.pretrain_seconds:
        common_t = int(a.pretrain_seconds * a.sample_rate)
        crop_jitter = int(round(a.aug_crop_jitter_ms * a.sample_rate / 1000.0))
        print(
            f"  SSL primary crop: {common_t} samples "
            f"({common_t/a.sample_rate:.2f}s, align={a.crop_align})"
        )
        if a.aug_crop_jitter_ms:
            print(
                f"  SSL crop jitter: +/-{crop_jitter} samples "
                f"({a.aug_crop_jitter_ms:g} ms)"
            )
    else:
        common_t = None
        crop_jitter = 0
        if a.aug_crop_jitter_ms:
            print(
                "  NOTE: --aug-crop-jitter-ms needs --pretrain-seconds or "
                "--pretrain-extra to create a cropped SSL window"
            )

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
    split_summary = {}
    for cohort, c_subs in cohorts.items():
        print("\n" + "#" * 70)
        print(f"# COHORT {cohort}  ({len(c_subs)} subjects: {c_subs})")
        print("#" * 70)
        if len(c_subs) < max(3, a.min_cohort):
            print(f"  skip: < min-cohort ({a.min_cohort})")
            continue

        cohort_domains = (
            {s: subject_domains[s] for s in c_subs} if subject_domains else None
        )
        train_s, val_s, test_s = split_within(
            c_subs, a.split, a.seed, strata=cohort_domains
        )
        effective_split_stratify = split_stratify

        if a.load:
            # restore a pretrained encoder (+ its stored seeds); skip pretraining
            model, cfg, blob, metadata, missing, unexpected = load_checkpoint(
                a.load, device
            )
            a.input_mode = cfg.model.input_mode
            a.spectral_frontend = cfg.model.spectral_frontend
            a.spectral_window_ms = cfg.model.spectral_window_ms
            a.spectral_range = [
                cfg.model.spectral_fmin, cfg.model.spectral_fmax
            ]
            a.spectral_aux = cfg.spectral_aux
            a.spectral_aux_bands = [
                value for band in cfg.spectral_aux_bands for value in band
            ]
            if a.probe_seed is None:
                a.probe_seed = metadata.get("probe_seed", 0)
            stored_bandpass = metadata.get("bandpass")
            if stored_bandpass is not None and not np.allclose(
                stored_bandpass, a.bandpass
            ):
                raise ValueError(
                    f"checkpoint was trained with bandpass {stored_bandpass}, "
                    f"but evaluation loaded {a.bandpass}; use the stored filter"
                )
            if stored_bandpass is None:
                print(
                    "  WARNING: checkpoint does not record its preprocessing "
                    "bandpass; historical MOABB MI runs usually used 8-32 Hz"
                )
            # use the checkpoint's splits -> avoid calibrating on pretrain subjects
            sp = metadata.get("splits")
            if sp:
                train_s, val_s, test_s = sp["train"], sp["val"], sp["test"]
                effective_split_stratify = metadata.get(
                    "split_stratify", "checkpoint-legacy"
                )
            print(f"  loaded {a.load} (seed={metadata.get('seed')}, "
                  f"probe_seed default={a.probe_seed}); using checkpoint split "
                  f"-> skip pretraining")
            if missing or unexpected:
                print(f"  checkpoint compatibility: missing={len(missing)} "
                      f"unexpected={len(unexpected)}")
        print(
            "  intra-cohort split "
            f"(disjoint people, stratify={effective_split_stratify}):"
        )
        print(f"    train({len(train_s)})={train_s} val({len(val_s)})={val_s} "
              f"test({len(test_s)})={test_s}")
        if subject_domains:
            print(
                "    subdatasets: "
                f"train={_domain_counts(train_s, subject_domains)} "
                f"val={_domain_counts(val_s, subject_domains)} "
                f"test={_domain_counts(test_s, subject_domains)}"
            )
        if not test_s:
            print("  skip: empty test split"); continue
        split_summary[cohort] = {
            "train": train_s,
            "validation": val_s,
            "test": test_s,
            "stratify": effective_split_stratify,
            "subdataset_counts": {
                "train": _domain_counts(train_s, subject_domains),
                "validation": _domain_counts(val_s, subject_domains),
                "test": _domain_counts(test_s, subject_domains),
            } if subject_domains else {},
        }

        if not a.load:
            cfg = build_cfg(a, len(train_s))
            model = (HierarchicalEEGJepa(cfg, n_levels=a.levels,
                                         pool_factor=a.pool_factor)
                     if a.hierarchical else EEGJepa(cfg)).to(device)
            # SSL pool = cohort train subjects (+ extra datasets, cropped)
            train_primary = full.subset_by_subject(train_s)
            train_primary_ssl = (
                train_primary.with_crop(
                    common_t, a.crop_align, crop_jitter=crop_jitter
                )
                if common_t is not None else train_primary
            )
            if extras:
                train_pool = ConcatDataset(
                    [train_primary_ssl] + extras)
                print(f"  SSL pool: {len(train_primary)} primary + "
                      f"{sum(len(e) for e in extras)} extra = {len(train_pool)} trials")
            else:
                train_pool = train_primary_ssl
            print(f"  == pretrain on {cohort} train / monitor {cohort} val "
                  f"({'hierarchical' if a.hierarchical else 'flat'}) ==")
            pretrain(model, train_pool, full.subset_by_subject(val_s), cfg, device,
                     amp=a.amp, workers=a.workers, patience=a.patience)
        if a.probe_seed is None:
            a.probe_seed = 0

        if a.save:
            path = a.save.replace(".pt", f"_{cohort.replace('|','_').replace('=','')}.pt")
            save_checkpoint(
                path, model, cfg, cohort=cohort, seed=a.seed,
                probe_seed=a.probe_seed,
                eval_protocol=a.eval_protocol,
                split_stratify=effective_split_stratify,
                bandpass=list(a.bandpass),
                augmentation=_augment_summary(cfg),
                splits={"train": train_s, "val": val_s, "test": test_s},
            )
            print(f"  saved encoder before downstream evaluation -> {path}")

        if a.eval_protocol == "cross-subject":
            print(f"  == {cohort} TEST: zero-shot cross-subject decoding ==")
            tres = evaluate_cross_subject(
                model, full, train_s, val_s, test_s, n_cls, device, a.pool,
                a.probe_epochs, a.probe_lr, a.probe_patience, a.probe_seed,
                raw_baseline=a.raw_baseline,
                subject_domains=subject_domains,
                mdfb_analysis=a.mdfb_analysis,
                mdfb_max_trials=a.mdfb_max_trials,
            )
            summary[cohort] = tres
        elif a.calib_trials:
            # LOW-LABEL SWEEP: vary #calibration trials, compare JEPA vs band-power
            # at each. SSL's value is label efficiency, not full-label peak.
            print(f"  == {cohort} TEST: low-label sweep (calib trials -> acc) ==")
            print(f"    {'n_calib':>8s}{'JEPA acc':>20s}{'band-power':>20s}")
            curve = {}
            for n_cal in a.calib_trials:
                r = calibrate_per_subject(model, full, test_s, n_cls, device, a.pool,
                                          a.calib_frac, raw=True, finetune=a.finetune,
                                          probe_seed=a.probe_seed, calib_cap=n_cal)
                jepa = f"{r['acc_mean']:.3f}+-{r['acc_std']:.3f}"
                rawm = f"{r.get('raw_mean', float('nan')):.3f}"
                print(f"    {n_cal:>8d}{jepa:>20s}{rawm:>20s}"
                      f"   delta={r['acc_mean'] - r.get('raw_mean', 0):+.3f}")
                curve[n_cal] = r
            tres = curve[a.calib_trials[-1]]
            summary[cohort] = {"sweep": {n: curve[n] for n in a.calib_trials}, **tres}
        else:
            print(f"  == {cohort} TEST: per-subject calibration ==")
            tres = calibrate_per_subject(model, full, test_s, n_cls, device, a.pool,
                                         a.calib_frac, a.raw_baseline, a.finetune,
                                         probe_seed=a.probe_seed)
            print(f"  COHORT {cohort} acc = {tres['acc_mean']:.3f} +- "
                  f"{tres['acc_std']:.3f} (n={tres['n']}, chance={1/n_cls:.3f})")
            summary[cohort] = tres
    summary_title = (
        "ZERO-SHOT CROSS-SUBJECT SUMMARY"
        if a.eval_protocol == "cross-subject"
        else "PER-SUBJECT CALIBRATION SUMMARY"
    )
    print("\n" + "=" * 70 + f"\n{summary_title} (chance "
          f"{1/n_cls:.3f})\n" + "=" * 70)
    for c, r in summary.items():
        if a.eval_protocol == "cross-subject":
            raw = r.get("band_power")
            extra = (
                f" band-power={raw['subject_balanced_accuracy_mean']:.3f}"
                if raw else ""
            )
            print(
                f"  {c:22s} subject-bal="
                f"{r['subject_balanced_accuracy_mean']:.3f}+-"
                f"{r['subject_balanced_accuracy_std']:.3f} "
                f"pooled-bal={r['balanced_accuracy']:.3f}{extra}"
            )
        elif "sweep" in r:                              # low-label curve
            print(f"  cohort {c} -- low-label sweep:")
            print(f"    {'n_calib':>8s}{'JEPA':>16s}{'band-power':>14s}{'delta':>9s}")
            for n_cal, rr in r["sweep"].items():
                print(f"    {n_cal:>8d}{rr['acc_mean']:>10.3f}+-{rr['acc_std']:.3f}"
                      f"{rr.get('raw_mean', float('nan')):>14.3f}"
                      f"{rr['acc_mean'] - rr.get('raw_mean', 0):>+9.3f}")
        else:
            extra = f"  raw={r['raw_mean']:.3f}" if "raw_mean" in r else ""
            print(f"  {c:22s} acc={r['acc_mean']:.3f} +- {r['acc_std']:.3f} "
                  f"(n={r['n']}){extra}")
    if a.eval_protocol == "cross-subject":
        print(
            "\nOne shared decoder was trained on labeled training subjects, "
            "selected on validation subjects, and evaluated on unseen test "
            "subjects with zero test-subject labels."
        )
    else:
        print(
            "\nEach cohort trained+validated+tested within itself; no cross-group "
            "transfer. Per-person calibration inside each cohort."
        )
    if a.save:
        result_path = (
            a.save[:-3] + "_results.json"
            if a.save.endswith(".pt")
            else a.save + "_results.json"
        )
        with open(result_path, "w") as f:
            json.dump(
                {
                    "eval_protocol": a.eval_protocol,
                    "dataset": a.dataset,
                    "seed": a.seed,
                    "probe_seed": a.probe_seed,
                    "bandpass_hz": list(a.bandpass),
                    "input_mode": a.input_mode,
                    "spectral_frontend": a.spectral_frontend,
                    "spectral_window_ms": a.spectral_window_ms,
                    "spectral_range_hz": list(a.spectral_range),
                    "spectral_aux": a.spectral_aux,
                    "spectral_aux_bands_hz": [
                        list(band) for band in _frequency_pairs(
                            a.spectral_aux_bands, "--spectral-aux-bands"
                        )
                    ],
                    "augmentation": _augment_summary(cfg),
                    "splits": split_summary,
                    "summary": summary,
                },
                f,
                indent=2,
            )
        print(f"results -> {result_path}")


if __name__ == "__main__":
    main()
