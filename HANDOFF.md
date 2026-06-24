# tseegjepa — Handoff

## Project goal

Build a multi-scale, hardware-agnostic EEG foundation model trained with a
Joint-Embedding Predictive Architecture (JEPA): predict masked latent
representations without reconstructing the raw waveform.

Requirements:

- Accept arbitrary EEG montages, channel counts, channel ordering, and recording
  lengths.
- Fuse multi-scale temporal, time-frequency, and spatial electrode-graph
  information.
- Pretrain with a masked online tower, predictor, and complete stop-gradient EMA
  target tower.
- Detect and reject representation collapse.
- Learn representations without labels, then train one shared MI decoder on
  labeled training subjects and transfer it to unseen subjects with zero target
  labels. Personalized calibration is secondary.
- Support subject/site-disjoint evaluation, demographic metrics, and corruption
  stress tests.

## Current architecture

### Tokenization

`src/tseegjepa/tokenizer.py`

- Per-electrode raw-patch, patch-FFT, or spectral-only embedding:
  `input_mode=raw|fft|both|spectral`.
- Explicit spectral frontend: none (true raw), fixed MI filter bank, or learned
  5–35 Hz bins from patch-centred two-second windows.
- Learned electrode identity plus continuous 3D scalp-position Fourier features.
- Unknown electrode names use a safe identity fallback instead of being dropped;
  measured coordinates can still distinguish them.
- Continuous Fourier temporal positions replace the old fixed 256-position table.
- Variable-duration recordings are padded with a sample-validity mask.

### Encoder and predictor

`src/tseegjepa/encoder/`, `src/tseegjepa/predictor.py`

- Parallel short/medium/long temporal branches.
- Spatial branch uses nearest-electrode neighborhoods from 3D coordinates.
- Attention is factorized into per-channel temporal attention and per-time
  spatial attention. It no longer constructs a dense
  `(channels × time)²` token-grid mask.
- Predictor uses the same factorized temporal/spatial organization.
- Complexity is substantially lower than the original dense-grid architecture,
  although temporal attention within each channel is still quadratic in the
  number of patches.

### JEPA objective

`src/tseegjepa/jepa.py`, `src/tseegjepa/jepa_hier.py`

- Complete EMA target tower: both tokenizer and encoder are momentum-updated.
- Smooth-L1 masked latent prediction.
- VICReg variance and covariance penalties operate on encoder representations.
- Optional masked-patch spectral band-power objective.
- Spectral targets are configurable and use long windows; the MI default is
  mu 8–13 Hz plus beta 13–30 Hz rather than broad delta-to-gamma targets.
- Optional domain-adversarial objective.
- Flat and hierarchical models now expose the same objectives and downstream
  fine-tuning interface.
- Hierarchical stages preserve the requested total depth, propagate gradients
  through the online pyramid, and return fixed-width downstream features even
  when recording duration changes.

### Train-only SSL augmentation

`src/tseegjepa/augment.py`, `src/tseegjepa/train/engine.py`

- Optional online augmentations are applied only to SSL training batches.
  Validation loss/collapse checks, linear probes, band-power baselines, MDFB
  diagnostics and test data stay clean.
- Supported knobs: crop jitter around the aligned SSL crop, within-trial time
  jitter, per-trial gain jitter, Gaussian sensor noise, channel dropout, and a
  narrow random frequency mask.
- This is not offline synthetic labeling. It creates more nuisance-varied
  unlabeled views without pretending to create new supervised subjects.
- Use mild settings for MI. Large frequency masks can remove the subject-
  specific mu/beta cue the model is supposed to encode.

### Masking and collapse control

`src/tseegjepa/masking.py`, `src/tseegjepa/collapse.py`

- Spatial masks select electrodes nearest to sampled scalp centers.
- `temporal_mask_frac` and `spatial_block_frac` now directly control actual
  coverage; the old temporal fraction was previously ignored.
- Variable-length padding is excluded from context and target masks.
- Validation checkpoint selection rejects collapsed representations instead of
  selecting solely by the lowest prediction loss.

### Montage-independent downstream representation

`src/tseegjepa/pooling.py`

- New default: `--pool spatial`.
- Per-channel representations are projected onto fixed spherical scalp anchors.
- Output width is independent of montage size and channel ordering while
  retaining coarse scalp topology and lateralization.
- `--pool chan` remains available for strict fixed-montage reproduction only.
- `--pool mean` is montage-independent but discards much of the spatial signal.

### Training and checkpoints

`src/tseegjepa/train/engine.py`, `src/tseegjepa/train/checkpoint.py`

- One shared pretraining engine now owns optimizer setup, LR schedule, EMA,
  AMP, worker seeding, collapse-aware validation, early stopping, and
  best-state restoration.
- Versioned checkpoint format stores model type/configuration and can store
  optimizer, epoch, step, and RNG state.
- Legacy checkpoints are loaded best-effort through compatibility remapping.
  Because the attention, tokenizer, and pooling architecture changed, retraining
  is recommended for publishable comparisons.
- `scripts/experiment.py --load` reuses an encoder and its original data split;
  it is evaluation/recalibration mode, not optimizer-resume mode.

### Downstream protocol

`src/tseegjepa/train/linear_probe.py`

- Primary `--eval-protocol cross-subject`: one subject-balanced linear decoder
  uses labels from training subjects, is selected by mean per-subject balanced
  accuracy on validation subjects, and is evaluated directly on unseen test
  subjects with zero test-subject labels.
- Train-only normalization/PCA; test labels are used only after prediction for
  metrics.
- Matching shared band-power baseline with the same train/validation/test people.
- Headline metric: mean and standard deviation of per-subject balanced accuracy.
- Secondary `--eval-protocol calibration`: personalized few-label/full-label
  probes on each test subject.
- Fine-tuning deep-copies the pretrained model for every subject/fold, preventing
  test-subject order leakage.
- Fine-tuning has its own validation rollback.
- Raw log-bandpower baseline remains available.

### Data

- `data/moabb_eeg.py`: MOABB loading, demographics, normalization, cropping and
  subject subsets. Unknown EEG sensor names are retained with fallback identity.
- Dreyer2023 retains the source `SUBDATASET` (A/B/C) and experimenter-sex
  metadata. Subject splits are stratified by A/B/C by default so every split
  represents each collection domain.
- `data/schema.py`: typed sample/batch contracts and runtime shape validation.

### Dreyer2023 collection domains

- A: 60 participants, 29 women, age 29.0 +/- 9.32 years; participant and
  experimenter gender study.
- B: 21 participants, 8 women, age 29.0 +/- 9.32 years; MDFB relationship study.
- C: 6 additional participants, 4 women (not 5), age 22.0 +/- 2.34 years,
  collected after the first two studies were published. They followed one of
  the prior experiment variants.
- A used OpenViBE 2.1.0 and B used 2.2.0. Within C, subjects 83 and 85 used
  the A-style experiment and OpenViBE 2.1.0; subjects 82, 84, 86, and 87 used
  the B-style experiment and OpenViBE 2.2.0. Experimenter composition also
  differs. Treat A/B/C as collection domains and report test metrics separately.
- With `--split 0.7 0.15 0.15`, stratification yields A/B/C counts of 42/15/4
  train, 9/3/1 validation, and 9/3/1 test. Dataset C has only one subject in
  validation and test, so its standalone result is exploratory. With seed 0,
  C82/C83/C85/C87 are train, C86 is validation, and C84 is test; therefore the
  C test result represents only one late B-style participant.
- Sources: [Dreyer et al. 2023 Scientific Data article](https://www.nature.com/articles/s41597-023-02445-z)
  and [MOABB Dreyer2023 dataset implementation](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/dreyer2023.py).

## Verification completed

- `python -m compileall -q src scripts tests`
- 19 unit/integration tests pass.
- Full `scripts/smoke_test.py` passes on CPU.
- Flat and hierarchical forward paths pass with mixed channel counts and mixed
  recording lengths.
- Spatial features are invariant to channel ordering and have fixed width across
  montages.
- Fine-tuning isolation, EMA tokenizer updates, collapse-aware selection,
  resumable checkpoints, and inputs beyond the old 256-patch limit are tested.
- GitHub Actions CI runs compilation and tests on Python 3.10 and 3.12.

The architecture refactor has been run for personalized calibration on
Dreyer2023, but the primary zero-shot cross-subject benchmark has not yet been
run. Existing 59.0% raw-input and collapsed `both` results are calibration
results and are not evidence of cross-subject decoding.

## Historical MI results

Task: two-class motor imagery with subject-disjoint train/validation/test
subjects and per-person calibration on held-out subjects.

| historical config | JEPA acc | band-power | note |
|---|---:|---:|---|
| frozen channel probe | 0.51–0.59 | 0.62 | below baseline |
| multi-dataset SSL | 0.52 | 0.62 | no observed scale gain |
| hierarchical | 0.51 | 0.62 | slightly worse |
| spectral auxiliary | 0.59 | 0.62 | approximately tied |
| fine-tune | 0.52 | 0.62 | overfit |
| FFT input | 0.58 | 0.62 | approximately tied |

Historical conclusion: the old architecture did not reliably beat classical
band-power on motor imagery. The architecture fixes remove evaluation and
representation limitations, but they do not guarantee that this scientific
conclusion will change. Rerun the full sweep before making a new claim.

Recent post-refactor personalized calibration runs:

- `raw`: 59.0% mean accuracy at 120 calibration trials versus 61.7% band-power.
- `both`: collapsed during pretraining; its downstream score is invalid and
  must not be compared as a healthy representation.

Neither result measures zero-shot cross-subject decoding.

Important correction: historical `raw`, `fft`, and `both` experiments used
MOABB's implicit 8–32 Hz motor-imagery filter. They were not broadband, even
when the model input was called `raw`. New experiments state `--bandpass`
explicitly.

## Recommended next runs

### 1. Verify locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[moabb,dev]"

.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test.py
```

### 2. Primary zero-shot cross-subject MI run

Start from scratch rather than loading an old encoder:

```bash
bash scripts/run_dreyer_mi.sh primary
```

With the single default cohort, this writes
`enc_cross_subject_aug_v5_all.pt` and
`enc_cross_subject_aug_v5_results.json`.

This is the broadband-raw plus mu/beta-auxiliary plus mild SSL-augmentation
candidate. It is one ablation condition, not evidence by itself that the
physiological prior or augmentation helps. Compare it against the non-augmented
`raw_mu_beta_aux` condition.

This protocol uses no labels from test subjects. Do not add `--calib-trials`;
that flag belongs to the secondary calibration protocol.

If CUDA still runs out of memory, reduce the batch size:

```bash
BATCH_SIZE=16 bash scripts/run_dreyer_mi.sh primary
```

Factorized attention should use less memory than the previous dense-grid model,
but the 384-dimensional model, multi-dataset batch construction, FFT/STFT
branches, and covariance regularizer remain substantial.

Useful runner overrides:

```bash
DEVICE=mps bash scripts/run_dreyer_mi.sh primary
WORKERS=4 bash scripts/run_dreyer_mi.sh primary
SAVE_DIR=runs LOG_DIR=runs bash scripts/run_dreyer_mi.sh primary
DRY_RUN=1 bash scripts/run_dreyer_mi.sh primary
```

### 3. Controlled spectral-prior ablation

Run conditions individually:

```bash
bash scripts/run_dreyer_mi.sh raw_broadband
bash scripts/run_dreyer_mi.sh raw_mi_band
bash scripts/run_dreyer_mi.sh raw_mu_beta_aux
bash scripts/run_dreyer_mi.sh raw_mu_beta_aug
bash scripts/run_dreyer_mi.sh filterbank
bash scripts/run_dreyer_mi.sh learned_spectral
bash scripts/run_dreyer_mi.sh raw_plus_learned
```

Or all six sequentially:

```bash
bash scripts/run_dreyer_mi.sh all
```

`all` runs the six spectral-prior conditions; run `raw_mu_beta_aug` separately
to isolate augmentation. Every condition uses the same A/B/C-stratified people,
seeds, architecture, optimization budget and shared decoder. Learned and fixed
spectral inputs use two-second windows; at 128 Hz this gives 0.5 Hz FFT
resolution.

The MDFB diagnostic uses labels from acquisition runs only after zero-shot
evaluation. It is an approximate reproduction of the MDFB heuristic and must
not be included in the zero-label decoding score.

This design follows the question raised by Benaroch et al. 2022 without
assuming its answer: their constrained MDFB algorithm was not significantly
better overall, so the fixed-prior and learned-prior conditions must be compared
empirically rather than treating 5–35 Hz knowledge as automatically superior.

### 4. Diagnostic only: evaluate the existing healthy raw encoder

The previous raw-input run was accidentally named `enc_fft_v2_all.pt`. Rename
it, then train the new shared decoder without repeating SSL:

```bash
mv enc_fft_v2_all.pt enc_raw_v2_all.pt

bash scripts/run_dreyer_mi.sh eval_old_raw
```

Do not use the collapsed `enc_both_v2_all.pt`.

This old checkpoint retains its original subject split, whose test set contains
A=8, B=5, C=0. It is useful only as a quick diagnostic and is not the primary
A/B/C-stratified benchmark. Do not force a new split onto it: its encoder has
already seen the old training subjects during SSL pretraining.

### 5. Re-evaluate a newly trained encoder without SSL pretraining

```bash
bash scripts/run_dreyer_mi.sh reeval
```

The stored train/validation/test split is reused. The shared decoder is retrained
deterministically from the stored encoder and training-subject labels.

### 6. Secondary few-label calibration

```bash
bash scripts/run_dreyer_mi.sh calibration
```

## Current repository state

- Base commit: `b690c0b`.
- The architecture refactor, tests, CI, and this handoff are currently
  uncommitted working-tree changes.
- Existing remote checkpoints such as `enc_all.pt` and `enc_fft_all.pt` were
  produced by the previous architecture. They may load best-effort, but use
  newly trained `*_v2` checkpoints for new comparisons.

## Infrastructure notes

- Python 3.10–3.12.
- Install real-data and development dependencies with `-e ".[moabb,dev]"`.
- Run long jobs inside `tmux` or with `nohup`; an interrupted SSH foreground
  process will otherwise stop.
- Keep checkpoints, logs, and generated figures outside Git.

## Remaining limitations

- Full real-data reruns are still required after the architecture refactor.
- The primary zero-shot cross-subject result is not available until the command
  above finishes.
- `--pool chan` remains fixed-montage only; use `--pool spatial` by default.
- Factorized attention removes full token-grid quadratic memory, but temporal
  attention is still quadratic within each channel. Very long continuous EEG
  should still use windows or larger patches.
- Spatial graph attention computes within each time point; it is not yet a
  custom sparse kernel.
- The spherical-anchor pooling resolution is fixed by
  `ModelConfig.pool_anchors` and is not currently exposed as a CLI flag.
