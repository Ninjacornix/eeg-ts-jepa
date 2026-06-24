from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import ConcatDataset
import pytest

from tseegjepa.config import MaskConfig, PretrainConfig
from tseegjepa.augment import augment_batch
from tseegjepa.data import MONTAGES, SyntheticEEGDataset, collate_variable_montage
from tseegjepa.jepa import EEGJepa
from tseegjepa.jepa_hier import HierarchicalEEGJepa, _pool_time
from tseegjepa.eval.mdfb import compare_bands, mdfb_like
from tseegjepa.masking import make_jepa_masks
from tseegjepa.tokenizer import EEGTokenizer
from tseegjepa.train.engine import PretrainTrainer
from tseegjepa.train.linear_probe import (
    _features,
    _loader,
    fit_cross_subject_probe,
    fit_cross_subject_raw_baseline,
    fit_linear_probe,
)


def tiny_cfg() -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.model.dim = 16
    cfg.model.depth = 2
    cfg.model.heads = 4
    cfg.model.patch_ms = 250
    cfg.model.temporal_windows = (1, 2)
    cfg.model.pool_anchors = 4
    cfg.pred_dim = 16
    cfg.pred_heads = 4
    cfg.pred_depth = 1
    cfg.batch_size = 4
    cfg.epochs = 1
    cfg.warmup_epochs = 0
    return cfg


def test_mask_ratios_are_effective():
    pos = torch.randn(2, 19, 3)
    valid = torch.ones(2, 19, dtype=torch.bool)
    rates = []
    for frac in (0.1, 0.5, 0.9):
        cfg = MaskConfig(temporal_mask_frac=frac)
        _, target = make_jepa_masks(
            pos, valid, 40, cfg, torch.Generator().manual_seed(3)
        )
        rates.append(float(target.float().mean()))
    assert rates[0] < rates[1] < rates[2]


def test_ssl_augmentation_preserves_batch_contract():
    cfg = tiny_cfg()
    cfg.augment.enabled = True
    cfg.augment.time_jitter_ms = 125
    cfg.augment.amplitude_jitter = 0.1
    cfg.augment.gaussian_noise = 0.02
    cfg.augment.channel_dropout = 0.2
    cfg.augment.freq_mask_prob = 1.0
    cfg.augment.freq_mask_width_hz = 2.0
    batch = collate_variable_montage([
        SyntheticEEGDataset(1, seconds=2, fixed_montage=MONTAGES["clinical_19"], seed=11)[0],
        SyntheticEEGDataset(1, seconds=1, fixed_montage=MONTAGES["consumer_8"], seed=12)[0],
    ])
    out = augment_batch(
        batch, cfg.augment, cfg.model.sample_rate,
        torch.Generator().manual_seed(3),
    )
    assert out["signal"].shape == batch["signal"].shape
    assert torch.equal(out["label"], batch["label"])
    assert out["ch_mask"].any(dim=1).all()
    assert not torch.allclose(out["signal"], batch["signal"])


def test_variable_time_and_montage_forward():
    cfg = tiny_cfg()
    items = [
        SyntheticEEGDataset(
            1, seconds=1.0, fixed_montage=MONTAGES["clinical_19"], seed=1
        )[0],
        SyntheticEEGDataset(
            1, seconds=1.5, fixed_montage=MONTAGES["consumer_8"], seed=2
        )[0],
    ]
    batch = collate_variable_montage(items)
    for cls in (EEGJepa, HierarchicalEEGJepa):
        model = cls(cfg)
        out = model(batch)
        assert torch.isfinite(out["loss"])
        features = model.encode(batch, pool="spatial")
        assert features.shape[0] == 2


def test_spatial_pool_dimension_is_montage_invariant():
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    dims = []
    for montage in (MONTAGES["clinical_19"], MONTAGES["consumer_8"]):
        batch = collate_variable_montage([
            SyntheticEEGDataset(1, seconds=1, fixed_montage=montage)[0]
        ])
        dims.append(model.encode(batch, pool="spatial").shape[1])
    assert dims == [cfg.model.pool_anchors * cfg.model.dim] * 2
    mixed = ConcatDataset([
        SyntheticEEGDataset(3, seconds=1, fixed_montage=MONTAGES["clinical_19"]),
        SyntheticEEGDataset(3, seconds=1, fixed_montage=MONTAGES["consumer_8"]),
    ])
    features, _, _ = _features(
        model, _loader(mixed, bs=2), torch.device("cpu"), pool="spatial"
    )
    assert features.shape == (6, cfg.model.pool_anchors * cfg.model.dim)


def test_spatial_features_are_channel_order_invariant():
    cfg = tiny_cfg()
    model = EEGJepa(cfg).eval()
    item = SyntheticEEGDataset(
        1, seconds=1, fixed_montage=MONTAGES["consumer_8"], seed=4
    )[0]
    order = torch.randperm(item["signal"].shape[0], generator=torch.Generator().manual_seed(2))
    shuffled = dict(item)
    for key in ("signal", "ch_ids", "ch_pos"):
        shuffled[key] = item[key][order]
    original_f = model.encode(collate_variable_montage([item]), pool="spatial")
    shuffled_f = model.encode(collate_variable_montage([shuffled]), pool="spatial")
    assert torch.allclose(original_f, shuffled_f, atol=1e-5, rtol=1e-4)


def test_hierarchical_feature_width_is_duration_invariant():
    cfg = tiny_cfg()
    model = HierarchicalEEGJepa(cfg)
    widths = []
    for seconds in (0.5, 2.0):
        batch = collate_variable_montage([
            SyntheticEEGDataset(1, seconds=seconds, fixed_montage=MONTAGES["consumer_8"])[0]
        ])
        widths.append(model.encode(batch, pool="spatial").shape[1])
    assert widths == [cfg.model.pool_anchors * cfg.model.dim * model.n_levels] * 2


def test_hierarchical_objective_parity():
    cfg = tiny_cfg()
    cfg.spectral_aux = 0.1
    cfg.use_domain_invariance = True
    cfg.n_domains = 2
    model = HierarchicalEEGJepa(cfg)
    batch = collate_variable_montage([
        SyntheticEEGDataset(1, seconds=1, site_id=0)[0],
        SyntheticEEGDataset(1, seconds=1, site_id=1)[0],
    ])
    out = model(batch)
    assert "loss_spec" in out and "loss_domain" in out


def test_finetune_does_not_mutate_shared_encoder():
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    before = copy.deepcopy(model.state_dict())
    train = SyntheticEEGDataset(16, seconds=1, n_classes=2, seed=7)
    test = SyntheticEEGDataset(8, seconds=1, n_classes=2, seed=8)
    fit_linear_probe(
        model, train, test, 2, torch.device("cpu"),
        epochs=1, finetune=True, pool="spatial",
    )
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)


def test_target_tokenizer_is_ema_updated():
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    online = next(model.tokenizer.parameters())
    target = next(model.target_tokenizer.parameters())
    old = target.detach().clone()
    with torch.no_grad():
        online.add_(1.0)
    model.update_target(0.5)
    assert torch.allclose(target, old + 0.5)


def test_tokenizer_has_no_fixed_time_or_identity_index_limit():
    cfg = tiny_cfg()
    tokenizer = EEGTokenizer(cfg.model)
    patches = cfg.model.max_time_patches + 5
    signal = torch.randn(1, 1, patches * cfg.model.patch_len)
    grid = tokenizer(
        signal,
        torch.tensor([[cfg.model.max_channels + 100]]),
        torch.tensor([[[0.0, 0.0, 1.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert grid.tokens.shape[1] == patches


def test_true_raw_and_long_window_spectral_frontends_are_distinct():
    cfg = tiny_cfg()
    cfg.model.sample_rate = 128
    cfg.model.patch_ms = 250
    cfg.model.input_mode = "raw"
    cfg.model.spectral_frontend = "none"
    raw = EEGTokenizer(cfg.model)
    assert not hasattr(raw, "spectral_proj")
    assert not hasattr(raw, "tf_proj")

    cfg.model.input_mode = "spectral"
    cfg.model.spectral_frontend = "filterbank"
    filterbank = EEGTokenizer(cfg.model)
    assert filterbank.spectral_proj.in_features == 5

    cfg.model.spectral_frontend = "learned"
    learned = EEGTokenizer(cfg.model)
    # Two seconds at 128 Hz gives 0.5-Hz resolution: 5..35 inclusive = 61 bins.
    assert learned.spectral_proj[1].in_features == 61


def test_mu_beta_auxiliary_uses_configured_long_window_targets():
    cfg = tiny_cfg()
    cfg.model.sample_rate = 128
    cfg.model.patch_ms = 250
    cfg.model.spectral_window_ms = 2000
    cfg.spectral_aux = 0.5
    cfg.spectral_aux_bands = ((8.0, 13.0), (13.0, 30.0))
    model = EEGJepa(cfg)
    batch = collate_variable_montage([
        SyntheticEEGDataset(1, seconds=2, seed=3)[0],
        SyntheticEEGDataset(1, seconds=2, seed=4)[0],
    ])
    out = model(batch)
    assert model.spectral_head.out_features == 2
    assert torch.isfinite(out["loss_spec"])


def test_mdfb_like_recovers_a_discriminative_mu_frequency():
    rng = np.random.default_rng(3)
    sample_rate, seconds, n = 128, 4, 40
    t = np.arange(sample_rate * seconds) / sample_rate
    names = ["C3", "FC3", "CP3", "C5", "C1",
             "C4", "FC4", "CP4", "C6", "C2"]
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(0, 0.2, size=(n, len(names), len(t)))
    for i, label in enumerate(y):
        amplitude = 0.5 if label == 0 else 2.0
        X[i, names.index("C3")] += amplitude * np.sin(2 * np.pi * 10 * t)
    result = mdfb_like(X, y, names, sample_rate)
    assert result["low_hz"] <= 10 <= result["high_hz"]
    comparison = compare_bands(
        result, {"low_hz": 9.5, "high_hz": 10.5, "peak_hz": 10.0}
    )
    assert comparison["learned_peak_in_mdfb"]


def test_hierarchical_masked_pool_is_normalized():
    x = torch.ones(1, 2 * 4, 3)
    mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.bool)
    pooled, _, pooled_mask = _pool_time(x, 2, 4, 2, mask)
    assert torch.allclose(pooled[pooled_mask.reshape(1, -1)], torch.ones(3, 3))


def test_collapse_aware_selection_rejects_collapsed_state():
    from tseegjepa.collapse import CollapseStats

    healthy = CollapseStats(0.1, 10, 0.5, False)
    collapsed = CollapseStats(0.0, 1, 0.01, True)
    assert PretrainTrainer.selection_score(0.5, healthy) < (
        PretrainTrainer.selection_score(0.01, collapsed)
    )
    assert PretrainTrainer.selection_score(0.01, collapsed) == float("inf")


def test_all_collapsed_validation_aborts(monkeypatch):
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    trainer = PretrainTrainer(model, cfg, torch.device("cpu"))
    train = SyntheticEEGDataset(4, seconds=1, seed=10)
    val = SyntheticEEGDataset(4, seconds=1, seed=11)

    from tseegjepa.collapse import CollapseStats

    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda ds: (0.01, CollapseStats(0.0, 1.0, 0.01, True)),
    )
    with pytest.raises(RuntimeError, match="no non-collapsed"):
        trainer.fit(train, val, verbose=False)


def test_zero_shot_cross_subject_probe_reports_subject_macro_metrics():
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    montage = MONTAGES["consumer_8"]
    train = ConcatDataset([
        SyntheticEEGDataset(
            12, seconds=1, site_id=0, n_classes=2, seed=1, fixed_montage=montage
        ),
        SyntheticEEGDataset(
            12, seconds=1, site_id=1, n_classes=2, seed=2, fixed_montage=montage
        ),
    ])
    val = SyntheticEEGDataset(
        10, seconds=1, site_id=2, n_classes=2, seed=3, fixed_montage=montage
    )
    test = ConcatDataset([
        SyntheticEEGDataset(
            10, seconds=1, site_id=3, n_classes=2, seed=4, fixed_montage=montage
        ),
        SyntheticEEGDataset(
            10, seconds=1, site_id=4, n_classes=2, seed=5, fixed_montage=montage
        ),
    ])
    result = fit_cross_subject_probe(
        model, train, val, test, 2, torch.device("cpu"),
        epochs=3, patience=2, pool="spatial", seed=0,
    )
    raw = fit_cross_subject_raw_baseline(
        train, val, test, 2, torch.device("cpu"),
        epochs=3, patience=2, seed=0,
    )
    assert result["n_train_subjects"] == 2
    assert result["n_test_subjects"] == 2
    assert len(result["per_subject"]) == 2
    assert 0.0 <= result["subject_balanced_accuracy_mean"] <= 1.0
    assert 0.0 <= raw["subject_balanced_accuracy_mean"] <= 1.0
