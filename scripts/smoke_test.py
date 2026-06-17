"""End-to-end smoke test: tiny pretrain -> collapse check -> probe -> LODO+OOD.

Runs in well under a minute on CPU. Exercises every component.
"""

import torch

from tseegjepa.config import PretrainConfig
from tseegjepa.data import SyntheticEEGDataset, collate_variable_montage, MONTAGES
from tseegjepa.jepa import EEGJepa
from tseegjepa.train.pretrain import pretrain, pick_device, move
from tseegjepa.train.linear_probe import fit_linear_probe
from tseegjepa.eval.loocv import leave_one_dataset_out
from tseegjepa.eval.corruptions import CorruptedDataset
from torch.utils.data import DataLoader


def tiny_cfg() -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.model.dim = 64
    cfg.model.depth = 2
    cfg.model.heads = 4
    cfg.model.patch_ms = 250.0      # coarse patches -> small token grid
    cfg.model.temporal_windows = (1, 2, 4)
    cfg.pred_dim = 48
    cfg.pred_depth = 2
    cfg.pred_heads = 4
    cfg.epochs = 3
    cfg.batch_size = 8
    cfg.warmup_epochs = 1
    cfg.collapse_log_every = 10
    return cfg


def test_shapes():
    cfg = tiny_cfg()
    ds = SyntheticEEGDataset(16, seconds=2.0, site_id=0, seed=0)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_variable_montage)
    model = EEGJepa(cfg)
    batch = next(iter(loader))
    out = model(batch)
    assert out["loss"].requires_grad and out["loss"].item() > 0
    cs = model.collapse_report(out)
    emb = model.encode(batch)
    assert emb.shape == (8, cfg.model.dim)
    print(f"[shapes] loss={out['loss'].item():.4f} {cs}  emb={tuple(emb.shape)}")


def test_variable_montage():
    """Different montages / channel counts must flow through the same model."""
    cfg = tiny_cfg()
    model = EEGJepa(cfg)
    for name, m in MONTAGES.items():
        ds = SyntheticEEGDataset(4, seconds=2.0, site_id=0, seed=1, fixed_montage=m)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_variable_montage)
        out = model(next(iter(loader)))
        print(f"[montage {name:14s} C={m.n_channels:2d}] loss={out['loss'].item():.4f}")


def test_pretrain_and_probe():
    cfg = tiny_cfg()
    device = pick_device("cpu")
    model = pretrain(cfg, device=device, n_sites=3, per_site=48, seconds=2.0, verbose=True)
    tr = SyntheticEEGDataset(128, seconds=2.0, site_id=9, seed=21)
    te = SyntheticEEGDataset(128, seconds=2.0, site_id=9, seed=22)
    res = fit_linear_probe(model, tr, te, n_classes=5, device=device)
    print("[probe]", {k: v for k, v in res.items() if k != "subgroup_accuracy"})
    # OOD spot check
    cte = CorruptedDataset(te, "gaussian", severity=1.0)
    ood = fit_linear_probe(model, tr, cte, n_classes=5, device=device)
    print(f"[ood gaussian] acc={ood['accuracy']:.3f} drop={res['accuracy']-ood['accuracy']:+.3f}")


def test_domain_invariance():
    cfg = tiny_cfg()
    cfg.use_domain_invariance = True
    cfg.epochs = 2
    model = pretrain(cfg, device=pick_device("cpu"), n_sites=3, per_site=32,
                     seconds=2.0, verbose=False)
    print("[domain-inv] trained ok")


def test_lodo():
    cfg = tiny_cfg()
    res = leave_one_dataset_out(
        n_sites=3, cfg=cfg, device=pick_device("cpu"),
        pretrain_epochs=2, per_site=48, probe_n=96,
        corruptions=["gaussian", "channel_dropout"], verbose=True,
    )
    print("[lodo summary]", res["_summary"])


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60, "\n shapes"); test_shapes()
    print("=" * 60, "\n variable montage"); test_variable_montage()
    print("=" * 60, "\n pretrain + probe + ood"); test_pretrain_and_probe()
    print("=" * 60, "\n domain invariance"); test_domain_invariance()
    print("=" * 60, "\n leave-one-dataset-out"); test_lodo()
    print("=" * 60, "\n ALL SMOKE TESTS PASSED")
