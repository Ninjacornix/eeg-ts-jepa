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
        self.cfg = cfg
        P = cfg.patch_len
        self.signal_proj = nn.Linear(P, cfg.dim)

        self.use_tf = cfg.use_tf_branch
        if self.use_tf:
            n_freq = cfg.n_fft // 2 + 1
            self.register_buffer("_tf_window", torch.hann_window(cfg.n_fft))
            self.tf_pool = nn.Linear(n_freq, cfg.tf_bins)
            self.tf_proj = nn.Linear(cfg.tf_bins, cfg.dim)

        self.ident_embed = nn.Embedding(cfg.max_channels, cfg.dim)
        self.time_embed = nn.Embedding(cfg.max_time_patches, cfg.dim)
        pf_dim = 3 * 2 * cfg.pos_fourier_bands
        self.pos_mlp = nn.Sequential(
            nn.Linear(pf_dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def _patchify(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, Tp, P), trailing samples dropped."""
        B, C, T = signal.shape
        P = self.cfg.patch_len
        Tp = T // P
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
        ident = self.ident_embed(ch_ids)
        pos = self.pos_mlp(fourier_features(ch_pos, cfg.pos_fourier_bands))
        sp = (ident + pos).unsqueeze(2)                       # (B,C,1,D)
        tpos = torch.arange(n_time, device=ch_ids.device)
        tp = self.time_embed(tpos).view(1, 1, n_time, cfg.dim)
        emb = (sp + tp).expand(B, C, n_time, cfg.dim)
        return emb.reshape(B, C * n_time, cfg.dim)

    def forward(
        self,
        signal: torch.Tensor,      # (B, C, T)
        ch_ids: torch.Tensor,      # (B, C) electrode identity indices
        ch_pos: torch.Tensor,      # (B, C, 3) unit-sphere coords
        ch_mask: torch.Tensor,     # (B, C) True = valid channel
    ) -> TokenGrid:
        cfg = self.cfg
        patches = self._patchify(signal)                   # (B,C,Tp,P)
        B, C, Tp, P = patches.shape

        tok = self.signal_proj(patches)                    # (B,C,Tp,D)
        if self.use_tf:
            tok = tok + self.tf_proj(self._stft_feats(patches))

        # identity + spatial position (broadcast over time)
        ident = self.ident_embed(ch_ids)                   # (B,C,D)
        pos = self.pos_mlp(fourier_features(ch_pos, cfg.pos_fourier_bands))  # (B,C,D)
        tok = tok + (ident + pos).unsqueeze(2)

        # temporal position (broadcast over channels)
        tpos = torch.arange(Tp, device=signal.device)
        tok = tok + self.time_embed(tpos).view(1, 1, Tp, cfg.dim)

        tok = self.norm(tok)
        tokens = tok.reshape(B, C * Tp, cfg.dim)

        ch_index = torch.arange(C, device=signal.device).view(1, C, 1).expand(B, C, Tp)
        time_index = torch.arange(Tp, device=signal.device).view(1, 1, Tp).expand(B, C, Tp)
        token_mask = ch_mask.unsqueeze(2).expand(B, C, Tp)

        return TokenGrid(
            tokens=tokens,
            ch_index=ch_index.reshape(B, -1),
            time_index=time_index.reshape(B, -1),
            token_mask=token_mask.reshape(B, -1),
            shape=(B, C, Tp),
        )
