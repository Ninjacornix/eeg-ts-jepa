"""Versioned, resumable checkpoints with legacy-model compatibility."""

from __future__ import annotations

import random
import copy
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from ..config import AugmentConfig, MaskConfig, ModelConfig, PretrainConfig
from ..jepa import EEGJepa
from ..jepa_hier import HierarchicalEEGJepa
from ..spectral import LEGACY_SPEC_BANDS

FORMAT_VERSION = 3


def config_from_dict(data: dict) -> PretrainConfig:
    data = dict(data)
    model_data = dict(data.pop("model", {}))
    if "spectral_frontend" not in model_data:
        model_data["spectral_frontend"] = (
            "legacy_stft" if model_data.get("use_tf_branch", True) else "none"
        )
    if "spectral_aux_bands" not in data:
        data["spectral_aux_bands"] = LEGACY_SPEC_BANDS
    model = ModelConfig(**model_data)
    mask = MaskConfig(**data.pop("mask", {}))
    augment = AugmentConfig(**data.pop("augment", {}))
    return PretrainConfig(model=model, mask=mask, augment=augment, **data)


def _upgrade_dataclass(value, defaults):
    for field in fields(defaults):
        if not hasattr(value, field.name):
            setattr(value, field.name, copy.deepcopy(getattr(defaults, field.name)))
    return value


def upgrade_config(cfg: PretrainConfig) -> PretrainConfig:
    """Fill fields absent from pickled v1 dataclasses."""
    had_aux_bands = hasattr(cfg, "spectral_aux_bands")
    had_frontend = hasattr(cfg.model, "spectral_frontend")
    cfg = _upgrade_dataclass(cfg, PretrainConfig())
    cfg.model = _upgrade_dataclass(cfg.model, ModelConfig())
    cfg.mask = _upgrade_dataclass(cfg.mask, MaskConfig())
    if not hasattr(cfg, "augment"):
        cfg.augment = AugmentConfig()
    else:
        cfg.augment = _upgrade_dataclass(cfg.augment, AugmentConfig())
    if not had_aux_bands:
        cfg.spectral_aux_bands = LEGACY_SPEC_BANDS
    if not had_frontend:
        cfg.model.spectral_frontend = (
            "legacy_stft" if cfg.model.use_tf_branch else "none"
        )
    return cfg


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model,
    cfg: PretrainConfig,
    optimizer=None,
    step: int = 0,
    epoch: int = 0,
    **metadata,
) -> None:
    hierarchical = isinstance(model, HierarchicalEEGJepa)
    blob = {
        "format_version": FORMAT_VERSION,
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "model_type": "hierarchical" if hierarchical else "flat",
        "model_args": {
            "n_levels": getattr(model, "n_levels", None),
            "pool_factor": getattr(model, "pool_factor", None),
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "rng_state": capture_rng_state(),
        "metadata": metadata,
    }
    torch.save(blob, path)


def _remap_legacy_branches(state: dict, cfg: PretrainConfig) -> dict:
    """Map the old dense branch ModuleList onto factorized branch modules."""
    out = {}
    n_temporal = len(cfg.model.temporal_windows)
    for key, value in state.items():
        new_key = key
        if ".branches." in key:
            prefix, rest = key.split(".branches.", 1)
            index, suffix = rest.split(".", 1)
            idx = int(index)
            branch = f".temporal.{idx}." if idx < n_temporal else ".spatial."
            new_key = prefix + branch + suffix
        out[new_key] = value
    return out


def load_model_state(model, state: dict) -> tuple[list[str], list[str]]:
    state = _remap_legacy_branches(state, model.cfg)
    # Old checkpoints shared the online tokenizer with the target encoder.
    if not any(k.startswith("target_tokenizer.") for k in state):
        for key, value in list(state.items()):
            if key.startswith("tokenizer."):
                state["target_tokenizer." + key[len("tokenizer."):]] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    optimizer=None,
    restore_rng: bool = False,
):
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = (
        config_from_dict(blob["config"])
        if "config" in blob
        else upgrade_config(blob["cfg"])  # v1
    )
    model_type = blob.get(
        "model_type",
        "hierarchical" if blob.get("hierarchical", False) else "flat",
    )
    args = blob.get("model_args", {})
    if model_type == "hierarchical":
        model = HierarchicalEEGJepa(
            cfg,
            n_levels=args.get("n_levels") or blob.get("levels", 3),
            pool_factor=args.get("pool_factor") or blob.get("pool_factor", 2),
        )
    else:
        model = EEGJepa(cfg)
    model = model.to(device)
    missing, unexpected = load_model_state(model, blob["state_dict"])
    if optimizer is not None and blob.get("optimizer"):
        optimizer.load_state_dict(blob["optimizer"])
    if restore_rng:
        restore_rng_state(blob.get("rng_state"))
    metadata = dict(blob.get("metadata", {}))
    for key in ("splits", "seed", "probe_seed", "cohort"):
        if key in blob:
            metadata.setdefault(key, blob[key])
    return model, cfg, blob, metadata, missing, unexpected
