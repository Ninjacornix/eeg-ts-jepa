# tseegjepa

Multi-scale, **hardware-agnostic EEG foundation model** trained with a
**Joint-Embedding Predictive Architecture (JEPA)**. Latent-space prediction
only — no raw-signal reconstruction.

## What it does

- **Arbitrary montages / channel counts.** Each electrode is tokenized as a
  `(position/identity embedding, signal patch)` pair, so any device layout
  flows through one shared model. Identity = a stable per-electrode embedding;
  position = Fourier features of the 10-20 scalp coordinate.
- **Multi-scale shared encoder.** Every block runs, *in parallel*, several
  temporal branches (short 50–100 ms, medium 250–500 ms, long 1–4 s windows), a
  time-frequency branch (STFT magnitude features injected at tokenization), and
  a spatial branch (electrode-graph / channel attention), then **fuses** them
  into one latent space.
- **JEPA pretraining.** A context encoder sees a masked view
  (spatial-block × variable-duration temporal target blocks), a predictor
  forecasts the **latent embeddings** of the masked targets, and an **EMA target
  encoder (stop-gradient)** produces those targets from the unmasked view.
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
  with **leave-one-dataset-out**, **subgroup-disaggregated** metrics, and
  **OOD/corruption** stress tests.

## Layout

```
src/tseegjepa/
  config.py          ModelConfig / MaskConfig / PretrainConfig
  tokenizer.py       per-electrode (identity+pos+signal+TF) tokenization
  encoder/           multi-scale parallel-branch encoder
    attention.py       branch attention masks (temporal scales + spatial)
    encoder.py         MultiScaleBlock / MultiScaleEncoder
  masking.py         spatial-block × variable-duration temporal target blocks
  predictor.py       JEPA latent predictor (mask tokens at target positions)
  jepa.py            EEGJepa: tokenizer + context/EMA-target encoders + predictor + loss
  collapse.py        embedding-variance / effective-rank diagnostics
  domain.py          gradient-reversal domain-invariance head
  jepa_hier.py       HierarchicalEEGJepa: temporal pyramid, per-level prediction
  data/              montages (10-20), synthetic EEG, MOABB adapter + electrodes
    moabb_eeg.py       MOABB -> batch schema; demographics; crop/align
    electrodes.py      global standard_1020 registry (ids + scalp coords)
  train/             pretrain loop + linear-probe / fine-tune / raw baseline
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

# cross-subject generalization: leave-one-subject-out
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
people, monitored on val people (JEPA loss + collapse), and each test person is
**calibrated** (a personal probe fit on their own data) and scored on their own
held-out data — the realistic per-person deployment setting, not a cross-subject
zero-shot claim.

```bash
# whole-dataset shared encoder, 70/15/15 subject split
python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
  --split 0.7 0.15 0.15 --pool chan --raw-baseline --device cuda --amp

# demographic cohorts: each cohort trained+validated+tested within itself
python scripts/experiment.py --dataset Dreyer2023 --cohort-by sex age ...

# multi-dataset SSL: pool extra datasets into pretraining (labels ignored),
# eval stays on --dataset; trials cropped+aligned to a common MI window
python scripts/experiment.py --dataset Dreyer2023 --n-subjects 87 \
  --pretrain-extra PhysionetMI:109 BNCI2014_001:9 Schirrmeister2017 \
  --crop-align end --finetune ...
```

Key flags: `--cohort-by sex|age` (demographic cohorts), `--hierarchical`
(pyramid model), `--finetune` (unfreeze encoder per subject), `--pretrain-extra`
(multi-dataset SSL pool), `--mask-frac` / `--dropout` (harder pretext / regular-
ization), `--crop-align end|center|start`, `--amp` (bf16), `--lr` / `--warmup-
epochs`, `--device cuda|mps|cpu`. `--raw-baseline` prints a per-subject
log-bandpower ceiling next to each result. Demographics (sex/age) are read from
MOABB metadata where available (e.g. Dreyer2023, BNCI2014_001); PhysioNet has
none.

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

Add any missing electrode names + coordinates to `data/montage.py`. Collate with
`collate_variable_montage` (pads ragged channel counts + builds a validity mask).

## Notes

- The synthetic data is for plumbing/eval only; class = dominant frequency band,
  with per-site nuisances (gain, line noise, drift, sensor noise) to make
  cross-site and corruption evaluation meaningful.
- Targets are layer-normalized before the smooth-L1 loss; the EMA momentum and
  masking ratios are the main knobs against collapse — watch `collapse.py`
  output during pretraining.
- Tokenizer (patch embedding) is shared and trained via the context path; the
  encoder weights are EMA-copied into the target encoder each step.
