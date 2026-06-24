"""Configuration dataclasses for the EEG-JEPA model and training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # --- signal / patching ---
    sample_rate: int = 200          # Hz; all input resampled to this canonical rate
    patch_ms: float = 50.0          # base (short-scale) patch duration in ms
    # per-patch input representation fed to the encoder:
    #   "raw"  = time-domain samples (default)
    #   "fft"  = rFFT log-magnitude of the patch (frequency domain -> matches MI)
    #   "both" = raw + patch-FFT embeddings summed
    #   "spectral" = only the selected long-window spectral frontend
    input_mode: str = "raw"
    # multi-scale temporal windows (in *patches*) for the parallel temporal branches
    # short ~50-100ms, medium ~250-500ms, long ~1-4s given patch_ms=50 -> patch=50ms
    temporal_windows: tuple[int, ...] = (2, 8, 32)   # 100ms, 400ms, 1600ms windows
    # --- spectral frontend ---
    # none: true raw/patch-FFT input with no hidden spectral branch
    # filterbank: fixed MI bands over long, patch-centred windows
    # learned: log-power FFT bins over the same long windows, learned projection
    # legacy_stft: old short-window branch, retained only for checkpoint loading
    spectral_frontend: str = "none"
    spectral_window_ms: float = 2000.0
    spectral_fmin: float = 5.0
    spectral_fmax: float = 35.0
    filterbank_bands: tuple[tuple[float, float], ...] = (
        (5.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 35.0),
    )
    # Legacy short-window settings.
    n_fft: int = 16                 # STFT over a patch; <= patch length in samples
    tf_bins: int = 8                # number of (mel-ish) frequency bins kept
    use_tf_branch: bool = False
    # --- spatial branch ---
    use_spatial_branch: bool = True
    spatial_k: int = 8              # nearest-electrode graph degree
    pos_fourier_bands: int = 8      # Fourier features for 3D electrode coordinates
    time_fourier_bands: int = 8     # unbounded continuous temporal positions
    pool_anchors: int = 8           # fixed spherical anchors for montage-invariant pooling
    # --- transformer ---
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_channels: int = 256         # identity hash buckets; final bucket = unknown
    max_time_patches: int = 256     # deprecated; retained for checkpoint compatibility

    @property
    def patch_len(self) -> int:
        return max(1, int(round(self.sample_rate * self.patch_ms / 1000.0)))

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.patch_ms <= 0:
            raise ValueError("sample_rate and patch_ms must be positive")
        if self.input_mode not in {"raw", "fft", "both", "spectral"}:
            raise ValueError(f"unknown input_mode {self.input_mode!r}")
        if self.spectral_frontend not in {
            "none", "filterbank", "learned", "legacy_stft"
        }:
            raise ValueError(
                f"unknown spectral_frontend {self.spectral_frontend!r}"
            )
        if self.input_mode == "spectral" and self.spectral_frontend == "none":
            raise ValueError("input_mode='spectral' requires a spectral frontend")
        if self.spectral_window_ms <= 0:
            raise ValueError("spectral_window_ms must be positive")
        nyquist = self.sample_rate / 2
        if not 0 <= self.spectral_fmin < self.spectral_fmax <= nyquist:
            raise ValueError("spectral range must lie inside the Nyquist limit")
        for lo, hi in self.filterbank_bands:
            if not 0 <= lo < hi <= nyquist:
                raise ValueError(f"invalid filter-bank band {(lo, hi)}")
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("dim must be positive and divisible by heads")
        if self.depth <= 0 or not self.temporal_windows:
            raise ValueError("depth and temporal_windows must be non-empty")
        if self.max_channels < 2:
            raise ValueError("max_channels must reserve at least one unknown bucket")
        if self.spatial_k <= 0 or self.pool_anchors <= 0:
            raise ValueError("spatial_k and pool_anchors must be positive")


@dataclass
class MaskConfig:
    # spatial-block masking: drop contiguous blocks of nearby electrodes
    spatial_block_frac: float = 0.5     # fraction of channels covered by spatial mask
    n_spatial_blocks: int = 2
    # variable-duration temporal masking: mask spans of time patches
    temporal_mask_frac: float = 0.5     # fraction of time patches masked (target)
    min_span: int = 2                   # min span length in patches
    max_span: int = 12                  # max span length in patches
    # number of distinct temporal target centers
    n_target_blocks: int = 4

    def validate(self) -> None:
        for name, value in (
            ("spatial_block_frac", self.spatial_block_frac),
            ("temporal_mask_frac", self.temporal_mask_frac),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.n_spatial_blocks <= 0 or self.n_target_blocks <= 0:
            raise ValueError("mask block counts must be positive")
        if self.min_span <= 0 or self.max_span < self.min_span:
            raise ValueError("invalid temporal span range")


@dataclass
class AugmentConfig:
    """Train-only SSL augmentations.

    These create extra unlabeled views for representation learning. They should
    stay mild for MI: too much temporal/frequency corruption can remove the
    mu/beta lateralization that the decoder must later use.
    """

    enabled: bool = False
    crop_jitter_ms: float = 0.0      # randomize dataset crop start around align point
    time_jitter_ms: float = 0.0      # circular shift within the valid trial window
    amplitude_jitter: float = 0.0    # log-normal per-trial gain std
    gaussian_noise: float = 0.0      # noise std as fraction of per-trial std
    channel_dropout: float = 0.0     # probability of dropping a valid electrode
    min_channels: int = 4
    freq_mask_prob: float = 0.0      # probability of one narrow FFT band-stop mask
    freq_mask_width_hz: float = 2.0
    freq_mask_fmin: float = 5.0
    freq_mask_fmax: float = 35.0

    def validate(self, sample_rate: int) -> None:
        for name in ("crop_jitter_ms", "time_jitter_ms", "amplitude_jitter",
                     "gaussian_noise", "freq_mask_width_hz"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("channel_dropout", "freq_mask_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_channels < 1:
            raise ValueError("min_channels must be positive")
        nyquist = sample_rate / 2
        if not 0 <= self.freq_mask_fmin < self.freq_mask_fmax <= nyquist:
            raise ValueError("frequency-mask range must lie inside Nyquist")


@dataclass
class PretrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
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
    # anti-collapse: VICReg-style regularizers on context-encoder embeddings.
    # var: pushes per-dim std toward var_target (fights variance collapse).
    # cov: decorrelates dims, off-diagonal covariance -> 0 (fights RANK collapse,
    #      the dims-become-correlated failure that variance alone misses).
    var_reg: float = 1.0
    var_target: float = 1.0
    cov_reg: float = 1.0
    # spectral auxiliary: also predict log band-power of masked patches, forcing
    # the encoder to represent mu/beta power (the motor-imagery signal). 0 = off.
    spectral_aux: float = 0.0
    spectral_aux_bands: tuple[tuple[float, float], ...] = (
        (8.0, 13.0),
        (13.0, 30.0),
    )
    seed: int = 0

    def validate(self) -> None:
        self.model.validate()
        self.mask.validate()
        self.augment.validate(self.model.sample_rate)
        if self.pred_dim <= 0 or self.pred_heads <= 0 or self.pred_dim % self.pred_heads:
            raise ValueError("pred_dim must be positive and divisible by pred_heads")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 <= self.ema_base <= self.ema_final <= 1.0:
            raise ValueError("EMA momentum must satisfy 0 <= base <= final <= 1")
        for lo, hi in self.spectral_aux_bands:
            if not 0 <= lo < hi <= self.model.sample_rate / 2:
                raise ValueError(f"invalid spectral auxiliary band {(lo, hi)}")
