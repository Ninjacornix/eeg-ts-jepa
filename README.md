# tseegjepa

Multi-scale, **hardware-agnostic EEG foundation model** trained with a
**Joint-Embedding Predictive Architecture (JEPA)**. Latent-space prediction
only — no raw-signal reconstruction.

## What it does

- **Arbitrary montages / channel counts.** Each electrode is tokenized as a
  `(position/identity embedding, signal patch)` pair, so any device layout
  flows through one shared model. Unknown identities fall back safely to a
  shared bucket while continuous coordinates retain their location.
- **Multi-scale shared encoder.** Every block runs, *in parallel*, several
  temporal branches and a nearest-electrode graph branch. Tokenization can be
  true raw-only, fixed MI filter-bank, learned long-window spectrum, or a
  raw/spectral combination.
  Attention is factorized over channels and time instead of building a dense
  `(channels × patches)²` mask.
- **JEPA pretraining.** A context encoder sees a masked view
  (spatial-block × variable-duration temporal target blocks), a predictor
  forecasts the **latent embeddings** of the masked targets, and a complete
  **EMA target tower** (tokenizer + encoder, stop-gradient) produces targets.
- **Hierarchical variant** (`jepa_hier.py`). A temporal pyramid: a single mask
  at the finest level propagated upward, with a predictor + EMA target + VICReg
  per level, so abstraction is learned at multiple scales (micro-dynamics →
  rhythms → trial state). Same interface as the flat model.
- **Anti-collapse.** Embedding std + effective-rank monitored each step; VICReg
  variance + covariance terms on the encoder output keep variance up and dims
  decorrelated (covariance term is what stops rank collapse).
- **Cross-site generalization.** Diverse multi-site pretraining + optional
  DANN-style domain-invariance objective (gradient reversal).
- **Downstream.** Freeze the encoder → linear probe, or fine-tune. Evaluated
  primarily with a single shared decoder on labeled training subjects and
  zero-label transfer to unseen test subjects. Personalized calibration,
  leave-one-dataset-out, subgroup metrics, and OOD stress remain secondary
  evaluations. Spatial-anchor pooling retains scalp topology while producing
  one fixed feature size across different montages.

## Layout

```
src/tseegjepa/
  config.py          ModelConfig / MaskConfig / PretrainConfig
  tokenizer.py       per-electrode (identity+pos+signal+TF) tokenization
  encoder/           multi-scale parallel-branch encoder
    attention.py       branch attention masks (temporal scales + spatial)
    encoder.py         MultiScaleBlock / MultiScaleEncoder
  masking.py         spatial-block × variable-duration temporal target blocks
  pooling.py         montage-invariant spherical-anchor spatial pooling
  predictor.py       JEPA latent predictor (mask tokens at target positions)
  jepa.py            EEGJepa: tokenizer + context/EMA-target encoders + predictor + loss
  collapse.py        embedding-variance / effective-rank diagnostics
  domain.py          gradient-reversal domain-invariance head
  jepa_hier.py       HierarchicalEEGJepa: temporal pyramid, per-level prediction
  data/              montages (10-20), synthetic EEG, MOABB adapter + electrodes
    moabb_eeg.py       MOABB -> batch schema; demographics; crop/align
    electrodes.py      global standard_1020 registry (ids + scalp coords)
  train/             pretrain loop + linear-probe / fine-tune / raw baseline
    engine.py          shared AMP/LR/EMA/validation/early-stop training engine
    checkpoint.py      versioned resumable checkpoints + legacy loading
  eval/              leave-one-dataset-out + corruption stress tests
scripts/
  smoke_test.py      full synthetic end-to-end run (< 1 min, CPU)
  experiment.py      subject-disjoint cohort protocol + multi-dataset SSL (main)
  train_moabb.py     single-dataset MOABB pretrain + probe / LODO
  inspect_data.py    dataset EDA: balance, band-power, lateralization, ceilings
  visualize_data.py  figures: PSD, topomaps, band-power heatmap, embeddings
```

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .

# full end-to-end sanity run (tiny, CPU)
.venv/bin/python scripts/smoke_test.py
.venv/bin/python -m pytest

# real-ish pretrain on synthetic multi-site data
.venv/bin/python -m tseegjepa.train.pretrain --epochs 20 --sites 4 --save eegjepa.pt

# freeze + linear probe (or --finetune)
.venv/bin/python -m tseegjepa.train.linear_probe --ckpt eegjepa.pt

# leave-one-dataset-out + OOD corruption stress
.venv/bin/python -m tseegjepa.eval.loocv --sites 4 --epochs 6
```

## Real data via MOABB

```bash
uv pip install --python .venv/bin/python -e ".[moabb]"

# pretrain on 2 subjects, linear-probe the held-out 3rd (motor imagery, 4-class)
.venv/bin/python scripts/train_moabb.py --subjects 1 2 3 --epochs 15

# leave-one-subject-out SSL followed by within-subject calibration
.venv/bin/python scripts/train_moabb.py --subjects 1 2 3 4 --epochs 20 --lodo

# fine-tune instead of frozen probe
.venv/bin/python scripts/train_moabb.py --subjects 1 2 3 --finetune
```

`data/moabb_eeg.py` wraps any MOABB dataset/paradigm into the batch schema:
one trial = one sample, `domain` = subject (the leave-one-subject-out axis),
`subgroup` = session. Channels map to a global `standard_1020` registry
(`data/electrodes.py`, real scalp coordinates), and non-1020 channels (e.g. EOG)
are dropped, so a model pretrained on one dataset accepts any other montage.
Default `BNCI2014_001` = 22 EEG channels, 4 motor-imagery classes, 9 subjects;
pass `--dataset`/`--paradigm`/`--n-classes` for others.

## Subject-disjoint experiments (`scripts/experiment.py`)

The main protocol. People are split into disjoint **train / val / test** groups
(no subject in two splits). The encoder is pretrained self-supervised on train
people and monitored on val people. In the primary `cross-subject` protocol,
one shared decoder is trained using labels from train people, selected using val
people, and applied directly to test people without any test-person labels.
Mean per-subject balanced accuracy is the headline metric.
When `--save` is supplied, the encoder is written before downstream evaluation
and a companion `*_results.json` stores the final metrics.

```bash
# whole-dataset shared encoder, 70/15/15 subject split
python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
  --split 0.7 0.15 0.15 --split-stratify subdataset \
  --bandpass 0.5 45 --input-mode raw --spectral-frontend none \
  --eval-protocol cross-subject \
  --pool spatial --raw-baseline --device cuda --amp

# demographic cohorts: each cohort trained+validated+tested within itself
python scripts/experiment.py --dataset Dreyer2023 --cohort-by sex age ...

# multi-dataset SSL: pool extra datasets into pretraining (labels ignored),
# eval stays on --dataset; trials cropped+aligned to a common MI window
python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
  --pretrain-extra PhysionetMI:109 BNCI2014_001:9 Schirrmeister2017 \
  --crop-align end --eval-protocol cross-subject ...

# secondary personalized/few-label evaluation
python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
  --eval-protocol calibration --calib-trials 5 10 20 40 120 ...
```

Key flags: `--eval-protocol cross-subject|calibration`, `--cohort-by sex|age`
(demographic cohorts), `--hierarchical` (pyramid model), `--pretrain-extra`
(multi-dataset SSL pool), `--mask-frac` / `--dropout` (harder pretext / regular-
ization), `--crop-align end|center|start`, `--amp` (bf16), `--lr` / `--warmup-
epochs`, `--pool spatial|mean|chan`, `--device cuda|mps|cpu`. `spatial` is the
montage-independent default; `chan` requires a fixed montage. `--raw-baseline`
trains a matching shared band-power decoder in cross-subject mode. Demographics
(sex/age) are read from
MOABB metadata where available (e.g. Dreyer2023, BNCI2014_001); PhysioNet has
none.

### MI spectral-prior ablation

MOABB's motor-imagery paradigms default to an implicit 8–32 Hz filter. The
experiment script now always supplies an explicit filter, so `raw` is no longer
mistaken for broadband. Use `--bandpass 0.5 45` for broadband and
`--bandpass 5 35` for the MI-prior condition.

The controlled conditions are:

- `raw_broadband`: true raw-only tokens, 0.5–45 Hz.
- `raw_mi_band`: true raw-only tokens, 5–35 Hz.
- `raw_mu_beta_aux`: broadband raw plus 8–13/13–30 Hz target prediction.
- `raw_mu_beta_aug`: same as `raw_mu_beta_aux`, plus mild train-only SSL
  augmentations.
- `filterbank`: fixed 5–8/8–13/13–20/20–30/30–35 Hz input.
- `learned_spectral`: learned 5–35 Hz bins from two-second windows.
- `raw_plus_learned`: broadband time-domain plus learned spectral tokens.

Run one condition, or all six sequentially:

```bash
bash scripts/run_dreyer_mi.sh raw_broadband
bash scripts/run_dreyer_mi.sh raw_mu_beta_aug
bash scripts/run_dreyer_mi.sh all
```

`all` runs the six spectral-prior conditions; run `raw_mu_beta_aug` separately
to isolate augmentation. Every condition uses the same subject split, seeds,
optimization budget and shared decoder. `--mdfb-analysis` computes a post-hoc
MDFB-like band from the first two acquisition runs and compares it with
label-free decoder frequency saliency. Those test-subject labels are diagnostic
only and never tune the representation or decoder.

### Train-only SSL augmentation

The repo supports online augmentation during JEPA pretraining only; validation,
linear probes, raw baselines, and test sets remain clean. This is safer than
adding offline "synthetic labels": the extra views increase nuisance variation
without pretending to create new labeled subjects.

Useful mild settings for MI are crop/time jitter, small gain/noise perturbation,
low-probability channel dropout, and narrow frequency masking:

```bash
--augment \
--aug-crop-jitter-ms 250 \
--aug-time-jitter-ms 125 \
--aug-amplitude-jitter 0.10 \
--aug-gaussian-noise 0.02 \
--aug-channel-dropout 0.05 \
--aug-freq-mask-prob 0.25 \
--aug-freq-mask-width-hz 2.0 \
--aug-freq-mask-range 5 35
```

Treat this as an ablation. If the augmented run drops below the non-augmented
run, first reduce `--aug-freq-mask-prob`; broad or frequent frequency masking
can erase the subject-specific mu/beta information needed for MI.

For the recommended Dreyer2023 cross-subject run, use the preset runner instead
of a long command:

```bash
bash scripts/run_dreyer_mi.sh primary
```

Common overrides are environment variables:

```bash
BATCH_SIZE=16 bash scripts/run_dreyer_mi.sh primary
DEVICE=mps bash scripts/run_dreyer_mi.sh raw_mu_beta_aux
SAVE_DIR=runs LOG_DIR=runs bash scripts/run_dreyer_mi.sh all
DRY_RUN=1 bash scripts/run_dreyer_mi.sh primary
```

## Inspect & visualize data

```bash
python scripts/inspect_data.py  --dataset Dreyer2023 --subjects 1 2 3   # EDA + ceilings
python scripts/visualize_data.py --dataset PhysionetMI --subjects 1 --tsne --out figs
```

`inspect_data.py` reports class balance, per-channel band-power by class,
C3/C4 lateralization, demographics, and difficulty ceilings (log-bandpower
logistic regression + CSP+LDA). `visualize_data.py` saves raw traces, class-mean
PSD, band-power heatmaps, per-band scalp **topomaps**, lateralization bars, and
PCA/t-SNE trial embeddings.

## Using your own EEG

Replace `SyntheticEEGDataset` with a `Dataset` whose `__getitem__` returns:

```python
{
  "signal":   FloatTensor (C, T),     # resampled to ModelConfig.sample_rate
  "ch_ids":   LongTensor  (C,),       # data.montage.channel_ids(names)
  "ch_pos":   FloatTensor (C, 3),     # data.montage.channel_positions(names)
  "label":    int,                    # downstream only
  "subgroup": int,                    # for disaggregated metrics
  "domain":   int,                    # site/device id
}
```

Use `data.electrodes.channel_ids(names, strict=False)` for unknown names and
provide their measured coordinates when available. Collate with
`collate_variable_montage`, which pads ragged channel and time dimensions and
builds both validity masks.

## Notes

- The synthetic data is for plumbing/eval only; class = dominant frequency band,
  with per-site nuisances (gain, line noise, drift, sensor noise) to make
  cross-site and corruption evaluation meaningful.
- Targets are layer-normalized before the smooth-L1 loss; the EMA momentum and
  masking ratios are the main knobs against collapse — watch `collapse.py`
  output during pretraining.
- The tokenizer and encoder are both EMA-copied into the target tower.
- Pretraining scripts share one trainer and checkpoint format. Trainer
  checkpoints can restore optimizer, step, epoch, and random-number state.
