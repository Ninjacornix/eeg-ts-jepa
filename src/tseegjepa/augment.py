"""Mild train-only EEG augmentations for SSL pretraining."""

from __future__ import annotations

import torch

from .config import AugmentConfig


def _sample_mask(batch: dict, signal: torch.Tensor) -> torch.Tensor:
    mask = batch.get("sample_mask")
    if mask is None:
        return torch.ones(
            signal.shape[0], signal.shape[-1],
            dtype=torch.bool, device=signal.device,
        )
    return mask.to(device=signal.device, dtype=torch.bool)


def _valid_mask(batch: dict, signal: torch.Tensor) -> torch.Tensor:
    ch_mask = batch["ch_mask"].to(device=signal.device, dtype=torch.bool)
    return ch_mask.unsqueeze(-1) & _sample_mask(batch, signal).unsqueeze(1)


def _rand(shape, signal: torch.Tensor, generator: torch.Generator | None):
    return torch.rand(shape, device=signal.device, generator=generator)


def _randn(shape, signal: torch.Tensor, generator: torch.Generator | None):
    return torch.randn(shape, device=signal.device, generator=generator)


def _randint(
    low: int,
    high: int,
    shape,
    signal: torch.Tensor,
    generator: torch.Generator | None,
):
    return torch.randint(low, high, shape, device=signal.device, generator=generator)


def _time_jitter(
    signal: torch.Tensor,
    sample_mask: torch.Tensor,
    max_shift: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if max_shift <= 0:
        return signal
    out = signal.clone()
    shifts = _randint(
        -max_shift, max_shift + 1, (signal.shape[0],), signal, generator
    )
    for i, shift in enumerate(shifts.tolist()):
        n_valid = int(sample_mask[i].sum().item())
        if n_valid > 1 and shift:
            out[i, :, :n_valid] = torch.roll(
                signal[i, :, :n_valid], shift, dims=-1
            )
    return out


def _channel_dropout(
    ch_mask: torch.Tensor,
    drop_prob: float,
    min_channels: int,
    signal: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if drop_prob <= 0:
        return ch_mask
    out = ch_mask.clone()
    B, _ = out.shape
    for i in range(B):
        valid = out[i].nonzero(as_tuple=False).flatten()
        if valid.numel() <= min_channels:
            continue
        drop = _rand((valid.numel(),), signal, generator) < drop_prob
        remaining = valid.numel() - int(drop.sum().item())
        if remaining < min_channels:
            dropped = drop.nonzero(as_tuple=False).flatten()
            restore_n = min_channels - remaining
            restore = dropped[
                torch.randperm(dropped.numel(), generator=generator)[:restore_n]
            ]
            drop[restore] = False
        out[i, valid[drop]] = False
    return out


def _frequency_mask(
    signal: torch.Tensor,
    sample_mask: torch.Tensor,
    cfg: AugmentConfig,
    sample_rate: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if cfg.freq_mask_prob <= 0 or cfg.freq_mask_width_hz <= 0:
        return signal
    out = signal.clone()
    B = signal.shape[0]
    apply = _rand((B,), signal, generator) < cfg.freq_mask_prob
    lo, hi = cfg.freq_mask_fmin, cfg.freq_mask_fmax
    width = min(cfg.freq_mask_width_hz, hi - lo)
    if width <= 0:
        return signal
    for i in range(B):
        if not bool(apply[i]):
            continue
        n_valid = int(sample_mask[i].sum().item())
        if n_valid < 4:
            continue
        center_min = lo + width / 2
        center_max = hi - width / 2
        u = float(_rand((), signal, generator).item())
        center = center_min + u * max(0.0, center_max - center_min)
        freqs = torch.fft.rfftfreq(
            n_valid, d=1.0 / sample_rate, device=signal.device
        )
        band = (freqs >= center - width / 2) & (freqs <= center + width / 2)
        if not band.any():
            band[(freqs - center).abs().argmin()] = True
        spec = torch.fft.rfft(out[i, :, :n_valid].float(), dim=-1)
        spec[:, band] = 0
        out[i, :, :n_valid] = torch.fft.irfft(
            spec, n=n_valid, dim=-1
        ).to(out.dtype)
    return out


def augment_batch(
    batch: dict,
    cfg: AugmentConfig,
    sample_rate: int,
    generator: torch.Generator | None = None,
) -> dict:
    """Return a train-only augmented batch.

    Validation, probing, and test data should call the model directly without this
    function. The label tensor is left unchanged because these are SSL views, not
    new supervised examples.
    """
    if not cfg.enabled:
        return batch
    out = dict(batch)
    signal = batch["signal"].clone()
    sample_mask = _sample_mask(batch, signal)

    max_shift = int(round(sample_rate * cfg.time_jitter_ms / 1000.0))
    signal = _time_jitter(signal, sample_mask, max_shift, generator)

    if cfg.amplitude_jitter > 0:
        scale = torch.exp(
            _randn((signal.shape[0], 1, 1), signal, generator)
            * cfg.amplitude_jitter
        )
        signal = signal * scale

    if cfg.gaussian_noise > 0:
        valid = _valid_mask(batch, signal)
        denom = valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        mean = (signal * valid).sum(dim=(1, 2), keepdim=True) / denom.view(-1, 1, 1)
        var = (((signal - mean) * valid) ** 2).sum(dim=(1, 2), keepdim=True)
        std = torch.sqrt(var / denom.view(-1, 1, 1) + 1e-6)
        noise = _randn(signal.shape, signal, generator) * std
        signal = signal + noise * cfg.gaussian_noise * valid

    signal = _frequency_mask(signal, sample_mask, cfg, sample_rate, generator)

    ch_mask = _channel_dropout(
        batch["ch_mask"].clone(), cfg.channel_dropout, cfg.min_channels,
        signal, generator,
    )
    valid = ch_mask.to(signal.device).unsqueeze(-1) & sample_mask.unsqueeze(1)
    out["signal"] = signal.masked_fill(~valid, 0)
    out["ch_mask"] = ch_mask
    return out
