"""Per-electrode tokenization.

Each token is a (position/identity embedding, signal patch) pair:

    token = signal_proj(patch)                     # raw time-domain content
          + tf_proj(stft(patch))                   # time-frequency content
          + electrode_identity_embed[ch_id]        # *which* electrode (device-agnostic)
          + pos_mlp(fourier(xyz))                  # *where* on the scalp
          + time_pos_embed[t]                      # temporal index of the patch

The result is a flat token grid of shape (B, C * Tp, dim) plus index tensors
describing each token's channel and time position, so downstream branches can
build channel-wise / time-wise attention masks for arbitrary montages.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .spectral import (
    band_power_features,
    centered_windows,
    frequency_bin_count,
    log_power_spectrum,
    select_frequency_range,
)


def fourier_features(xyz: torch.Tensor, bands: int) -> torch.Tensor:
    """(..., 3) -> (..., 3 * 2 * bands) sinusoidal positional features."""
    freqs = 2.0 ** torch.arange(bands, device=xyz.device, dtype=xyz.dtype)
    ang = xyz.unsqueeze(-1) * freqs * math.pi          # (...,3,bands)
    feats = torch.cat([ang.sin(), ang.cos()], dim=-1)  # (...,3,2*bands)
    return feats.flatten(-2)


class TokenGrid:
    """Container describing a tokenized batch (returned by EEGTokenizer)."""

    __slots__ = ("tokens", "ch_index", "time_index", "token_mask", "shape")

    def __init__(self, tokens, ch_index, time_index, token_mask, shape):
        self.tokens = tokens          # (B, N, D)
        self.ch_index = ch_index      # (B, N) channel slot index
        self.time_index = time_index  # (B, N) time-patch index
        self.token_mask = token_mask  # (B, N) True = valid token
        self.shape = shape            # (B, C, Tp)


class EEGTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        P = cfg.patch_len
        self.input_mode = cfg.input_mode
        if self.input_mode in ("raw", "both"):
            self.signal_proj = nn.Linear(P, cfg.dim)        # time-domain patch
        if self.input_mode in ("fft", "both"):
            self.fft_proj = nn.Linear(P // 2 + 1, cfg.dim)  # rFFT log-magnitude patch

        self.spectral_frontend = cfg.spectral_frontend
        self.spectral_window_len = max(
            P, int(round(cfg.sample_rate * cfg.spectral_window_ms / 1000.0))
        )
        if self.spectral_frontend == "legacy_stft":
            n_freq = cfg.n_fft // 2 + 1
            self.register_buffer("_tf_window", torch.hann_window(cfg.n_fft))
            self.tf_pool = nn.Linear(n_freq, cfg.tf_bins)
            self.tf_proj = nn.Linear(cfg.tf_bins, cfg.dim)
        elif self.spectral_frontend == "filterbank":
            self.spectral_proj = nn.Linear(len(cfg.filterbank_bands), cfg.dim)
        elif self.spectral_frontend == "learned":
            n_freq = frequency_bin_count(
                cfg.sample_rate,
                self.spectral_window_len,
                cfg.spectral_fmin,
                cfg.spectral_fmax,
            )
            self.spectral_proj = nn.Sequential(
                nn.LayerNorm(n_freq),
                nn.Linear(n_freq, cfg.dim),
            )

        self.ident_embed = nn.Embedding(cfg.max_channels, cfg.dim)
        pf_dim = 3 * 2 * cfg.pos_fourier_bands
        self.pos_mlp = nn.Sequential(
            nn.Linear(pf_dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim)
        )
        tf_dim = 2 * cfg.time_fourier_bands
        self.time_mlp = nn.Sequential(
            nn.Linear(tf_dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def _identity(self, ch_ids: torch.Tensor) -> torch.Tensor:
        # IDs outside the configured registry share the unknown bucket; their
        # continuous position still distinguishes them.
        ids = ch_ids.clamp(min=0, max=self.cfg.max_channels - 1)
        return self.ident_embed(ids)

    def _time_embeddings(
        self, n_time: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if n_time < 1:
            return torch.empty(0, self.cfg.dim, device=device, dtype=dtype)
        t = torch.arange(n_time, device=device, dtype=dtype)
        t = t / max(1, n_time - 1)
        freqs = 2.0 ** torch.arange(
            self.cfg.time_fourier_bands, device=device, dtype=dtype
        )
        ang = t[:, None] * freqs[None, :] * math.pi
        return self.time_mlp(torch.cat((ang.sin(), ang.cos()), dim=-1))

    def _patchify(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, Tp, P), trailing samples dropped."""
        B, C, T = signal.shape
        P = self.cfg.patch_len
        Tp = T // P
        if Tp < 1:
            raise ValueError(
                f"signal length {T} is shorter than one patch ({P} samples)"
            )
        signal = signal[..., : Tp * P]
        return signal.reshape(B, C, Tp, P)

    def _stft_feats(self, patches: torch.Tensor) -> torch.Tensor:
        """(B,C,Tp,P) -> (B,C,Tp,tf_bins) magnitude features."""
        B, C, Tp, P = patches.shape
        nfft = self.cfg.n_fft
        x = patches.reshape(-1, P)
        # force length to exactly n_fft (truncate+window, or zero-pad) so rfft is
        # called WITHOUT n= -> avoids the spurious "output resized" rfft warning.
        if P >= nfft:
            x = x[:, :nfft] * self._tf_window.unsqueeze(0)
        else:
            x = F.pad(x, (0, nfft - P))
        spec = torch.fft.rfft(x.contiguous(), dim=-1)      # n_freq = n_fft//2 + 1
        mag = torch.log1p(spec.abs())                      # (BCTp, n_freq)
        binned = self.tf_pool(mag)                         # (BCTp, tf_bins)
        return binned.reshape(B, C, Tp, -1)

    def spectral_windows(
        self, signal: torch.Tensor, n_patches: int | None = None
    ) -> torch.Tensor:
        if n_patches is None:
            n_patches = signal.shape[-1] // self.cfg.patch_len
        return centered_windows(
            signal,
            self.cfg.patch_len,
            n_patches,
            self.spectral_window_len,
        )

    def _long_spectral_feats(
        self, signal: torch.Tensor, n_patches: int
    ) -> torch.Tensor:
        windows = self.spectral_windows(signal, n_patches)
        if self.spectral_frontend == "filterbank":
            return band_power_features(
                windows, self.cfg.sample_rate, self.cfg.filterbank_bands
            )
        log_power, freqs = log_power_spectrum(windows, self.cfg.sample_rate)
        return select_frequency_range(
            log_power,
            freqs,
            self.cfg.spectral_fmin,
            self.cfg.spectral_fmax,
        )

    def position_embeddings(
        self,
        ch_ids: torch.Tensor,   # (B, C)
        ch_pos: torch.Tensor,   # (B, C, 3)
        n_time: int,
    ) -> torch.Tensor:
        """(B, C*Tp, D) identity+spatial+temporal embeddings, no signal content.

        Used by the predictor so mask tokens know *where* (electrode + time) they
        must forecast, independent of any observed signal.
        """
        cfg = self.cfg
        B, C = ch_ids.shape
        ident = self._identity(ch_ids)
        pos = self.pos_mlp(fourier_features(ch_pos, cfg.pos_fourier_bands))
        sp = (ident + pos).unsqueeze(2)                       # (B,C,1,D)
        tp = self._time_embeddings(
            n_time, ch_ids.device, ch_pos.dtype
        ).view(1, 1, n_time, cfg.dim)
        emb = (sp + tp).expand(B, C, n_time, cfg.dim)
        return emb.reshape(B, C * n_time, cfg.dim)

    def forward(
        self,
        signal: torch.Tensor,      # (B, C, T)
        ch_ids: torch.Tensor,      # (B, C) electrode identity indices
        ch_pos: torch.Tensor,      # (B, C, 3) unit-sphere coords
        ch_mask: torch.Tensor,     # (B, C) True = valid channel
        sample_mask: torch.Tensor | None = None,  # (B,T) True = valid sample
    ) -> TokenGrid:
        cfg = self.cfg
        patches = self._patchify(signal)                   # (B,C,Tp,P)
        B, C, Tp, P = patches.shape

        # per-patch input embedding: time-domain, frequency-domain, or both
        tok = None
        if self.input_mode in ("raw", "both"):
            tok = self.signal_proj(patches)
        if self.input_mode in ("fft", "both"):
            # (B,C,Tp,P//2+1)
            mag = torch.log1p(
                torch.fft.rfft(patches.float(), dim=-1).abs()
            )
            fft_tok = self.fft_proj(mag.to(patches.dtype))
            tok = fft_tok if tok is None else tok + fft_tok
        if self.spectral_frontend == "legacy_stft":
            tok = tok + self.tf_proj(self._stft_feats(patches))
        elif self.spectral_frontend in {"filterbank", "learned"}:
            spectral = self._long_spectral_feats(signal, Tp).to(patches.dtype)
            spectral_tok = self.spectral_proj(spectral)
            tok = spectral_tok if tok is None else tok + spectral_tok
        if tok is None:
            raise RuntimeError("tokenizer has no active signal input")

        # identity + spatial position (broadcast over time)
        ident = self._identity(ch_ids)                     # (B,C,D)
        pos = self.pos_mlp(fourier_features(ch_pos, cfg.pos_fourier_bands))  # (B,C,D)
        tok = tok + (ident + pos).unsqueeze(2)

        # temporal position (broadcast over channels)
        tok = tok + self._time_embeddings(
            Tp, signal.device, tok.dtype
        ).view(1, 1, Tp, cfg.dim)

        tok = self.norm(tok)
        tokens = tok.reshape(B, C * Tp, cfg.dim)

        ch_index = torch.arange(C, device=signal.device).view(1, C, 1).expand(B, C, Tp)
        time_index = torch.arange(Tp, device=signal.device).view(1, 1, Tp).expand(B, C, Tp)
        if sample_mask is None:
            patch_mask = torch.ones(B, Tp, dtype=torch.bool, device=signal.device)
        else:
            if (sample_mask.sum(-1) < P).any():
                raise ValueError("every sample must contain at least one full patch")
            usable = sample_mask[:, : Tp * P]
            patch_mask = usable.reshape(B, Tp, P).all(-1)
        token_mask = ch_mask.unsqueeze(2) & patch_mask.unsqueeze(1)

        return TokenGrid(
            tokens=tokens,
            ch_index=ch_index.reshape(B, -1),
            time_index=time_index.reshape(B, -1),
            token_mask=token_mask.reshape(B, -1),
            shape=(B, C, Tp),
        )
