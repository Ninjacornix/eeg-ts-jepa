from __future__ import annotations

from dataclasses import asdict

import torch

from tseegjepa.config import PretrainConfig
from tseegjepa.jepa import EEGJepa
from tseegjepa.spectral import LEGACY_SPEC_BANDS
from tseegjepa.train.checkpoint import config_from_dict
from tseegjepa.train.engine import PretrainTrainer


def test_trainer_checkpoint_restores_optimizer_progress(tmp_path):
    cfg = PretrainConfig()
    cfg.model.dim = 16
    cfg.model.heads = 4
    cfg.model.depth = 1
    cfg.pred_dim = 16
    cfg.pred_heads = 4
    model = EEGJepa(cfg)
    trainer = PretrainTrainer(model, cfg, torch.device("cpu"))
    trainer.step = 17
    trainer.epoch = 3
    path = tmp_path / "resume.pt"
    trainer.save(path, purpose="test")

    restored, metadata, missing, unexpected = PretrainTrainer.from_checkpoint(
        path, torch.device("cpu")
    )
    assert restored.step == 17
    assert restored.epoch == 3
    assert metadata["purpose"] == "test"
    assert not missing and not unexpected


def test_legacy_config_restores_hidden_stft_and_five_band_auxiliary():
    data = asdict(PretrainConfig())
    data["model"].pop("spectral_frontend")
    data["model"]["use_tf_branch"] = True
    data.pop("spectral_aux_bands")
    restored = config_from_dict(data)
    assert restored.model.spectral_frontend == "legacy_stft"
    assert restored.spectral_aux_bands == LEGACY_SPEC_BANDS
