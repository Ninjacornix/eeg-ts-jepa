"""Configuration dataclasses for the EEG-JEPA model and training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # --- signal / patching ---
    sample_rate: int = 200          # Hz; all input resampled to this canonical rate
    patch_ms: float = 50.0          # base (short-scale) patch duration in ms
    # multi-scale temporal windows (in *patches*) for the parallel temporal branches
    # short ~50-100ms, medium ~250-500ms, long ~1-4s given patch_ms=50 -> patch=50ms
    temporal_windows: tuple[int, ...] = (2, 8, 32)   # 100ms, 400ms, 1600ms windows
    # --- time-frequency branch ---
    n_fft: int = 16                 # STFT over a patch; <= patch length in samples
    tf_bins: int = 8                # number of (mel-ish) frequency bins kept
    use_tf_branch: bool = True
    # --- spatial branch ---
    use_spatial_branch: bool = True
    pos_fourier_bands: int = 8      # Fourier features for 3D electrode coordinates
    # --- transformer ---
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_channels: int = 256         # size of learned electrode-identity table
    max_time_patches: int = 256     # size of learned temporal-position table

    @property
    def patch_len(self) -> int:
        return max(1, int(round(self.sample_rate * self.patch_ms / 1000.0)))


@dataclass
class MaskConfig:
    # spatial-block masking: drop contiguous blocks of nearby electrodes
    spatial_block_frac: float = 0.5     # fraction of channels covered by spatial mask
    n_spatial_blocks: int = 2
    # variable-duration temporal masking: mask spans of time patches
    temporal_mask_frac: float = 0.5     # fraction of time patches masked (target)
    min_span: int = 2                   # min span length in patches
    max_span: int = 12                  # max span length in patches
    # number of distinct target blocks the predictor must forecast
    n_target_blocks: int = 4


@dataclass
class PretrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    # predictor
    pred_dim: int = 128
    pred_depth: int = 3
    pred_heads: int = 4
    # EMA schedule for the target encoder
    ema_base: float = 0.996
    ema_final: float = 1.0
    # optim
    lr: float = 1.5e-3
    weight_decay: float = 0.04
    epochs: int = 20
    batch_size: int = 16
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    # domain-invariance (optional)
    use_domain_invariance: bool = False
    domain_lambda: float = 0.1          # gradient-reversal strength
    n_domains: int = 1
    # collapse monitoring
    collapse_log_every: int = 50
    # anti-collapse: VICReg-style regularizers on predicted embeddings.
    # var: pushes per-dim std toward var_target (fights variance collapse).
    # cov: decorrelates dims, off-diagonal covariance -> 0 (fights RANK collapse,
    #      the dims-become-correlated failure that variance alone misses).
    var_reg: float = 1.0
    var_target: float = 1.0
    cov_reg: float = 1.0
    seed: int = 0
