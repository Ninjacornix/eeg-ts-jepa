"""Shared spectral utilities for physiologically informed EEG objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


LEGACY_SPEC_BANDS = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
)
MI_AUX_BANDS = ((8.0, 13.0), (13.0, 30.0))
MI_FILTERBANK_BANDS = (
    (5.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
    (30.0, 35.0),
)


def centered_windows(
    signal: torch.Tensor,
    patch_len: int,
    n_patches: int,
    window_len: int,
) -> torch.Tensor:
    """Return one long, patch-centred window for every temporal token.

    Args:
        signal: ``(B, C, samples)``.
        patch_len: token stride in samples.
        n_patches: number of valid patch positions in the padded batch.
        window_len: spectral context length in samples.
    """
    window_len = max(int(window_len), int(patch_len))
    extra = window_len - patch_len
    left = extra // 2
    right = extra - left
    padded = F.pad(signal, (left, right))
    windows = padded.unfold(-1, window_len, patch_len)
    if windows.shape[-2] < n_patches:
        raise ValueError("spectral window extraction produced too few patches")
    return windows[..., :n_patches, :]


def log_power_spectrum(
    windows: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windowed log-power spectrum and corresponding frequencies."""
    length = windows.shape[-1]
    taper = torch.hann_window(
        length, device=windows.device, dtype=windows.dtype
    )
    spectrum = torch.fft.rfft((windows * taper).float(), dim=-1)
    power = spectrum.abs().square()
    freqs = torch.fft.rfftfreq(length, 1.0 / sample_rate).to(windows.device)
    return torch.log1p(power), freqs


def select_frequency_range(
    log_power: torch.Tensor,
    freqs: torch.Tensor,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        raise ValueError(
            f"spectral range {fmin:g}-{fmax:g} Hz has no FFT bins"
        )
    return log_power[..., mask]


def band_power_features(
    windows: torch.Tensor,
    sample_rate: int,
    bands,
) -> torch.Tensor:
    """Log mean power for each requested frequency band."""
    log_power, freqs = log_power_spectrum(windows, sample_rate)
    power = torch.expm1(log_power)
    features = []
    for lo, hi in bands:
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if mask.any():
            features.append(torch.log(power[..., mask].mean(-1) + 1e-6))
        else:
            features.append(torch.zeros_like(power[..., 0]))
    return torch.stack(features, dim=-1)


def frequency_bin_count(
    sample_rate: int,
    window_len: int,
    fmin: float,
    fmax: float,
) -> int:
    freqs = torch.fft.rfftfreq(window_len, 1.0 / sample_rate)
    return int(((freqs >= fmin) & (freqs <= fmax)).sum())
