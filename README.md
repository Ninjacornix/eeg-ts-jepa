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
- **Collapse monitoring.** Embedding std + effective rank logged each interval;
  warns on low variance/rank.
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
  data/              montages (10-20) + synthetic multi-site EEG
  train/             pretrain loop + linear-probe / fine-tune
  eval/              leave-one-dataset-out + corruption stress tests
scripts/smoke_test.py   full end-to-end run (< 1 min, CPU)
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
