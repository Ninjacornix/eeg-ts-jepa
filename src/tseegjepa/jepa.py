"""EEG-JEPA: the full self-supervised model.

Data flow per step (I-JEPA-style, latent-space only -- no raw reconstruction):

  tokenize -> sample masks
  target tokenizer+encoder (EMA, stop-grad): encode UNMASKED view -> target latents
  context encoder: encode MASKED view (targets hidden)
  predictor: forecast target latents from context + positions
  loss = smooth-L1(predicted, target latents)   [+ optional domain-invariance]
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PretrainConfig
from .collapse import collapse_stats
from .data.schema import validate_eeg_batch
from .domain import DomainClassifier
from .encoder import MultiScaleEncoder
from .masking import make_jepa_masks
from .pooling import SpatialAnchorPool
from .predictor import Predictor
from .spectral import LEGACY_SPEC_BANDS, band_power_features
from .tokenizer import EEGTokenizer

# Backward-compatible public name; new models use cfg.spectral_aux_bands.
SPEC_BANDS = LEGACY_SPEC_BANDS


@torch.no_grad()
def band_power_target(
    windows: torch.Tensor,
    sample_rate: int,
    bands=SPEC_BANDS,
) -> torch.Tensor:
    """Long-window log band power aligned to every temporal token."""
    return band_power_features(windows, sample_rate, bands)


class EEGJepa(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        mcfg = cfg.model
        self.tokenizer = EEGTokenizer(mcfg)
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.context_encoder = MultiScaleEncoder(mcfg)
        # Complete EMA target tower: tokenizer + encoder.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in list(self.target_tokenizer.parameters()) + list(self.target_encoder.parameters()):
            p.requires_grad_(False)
        self.spatial_pool = SpatialAnchorPool(mcfg.pool_anchors)
        self.predictor = Predictor(
            enc_dim=mcfg.dim, pred_dim=cfg.pred_dim,
            depth=cfg.pred_depth, heads=cfg.pred_heads,
        )
        self.domain_head = (
            DomainClassifier(mcfg.dim, cfg.n_domains, cfg.domain_lambda)
            if cfg.use_domain_invariance else None
        )
        # spectral auxiliary head: predictor latent -> log band-power per band
        self.spectral_head = (
            nn.Linear(mcfg.dim, len(cfg.spectral_aux_bands))
            if cfg.spectral_aux > 0 else None
        )

    # --- EMA -------------------------------------------------------------
    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in (
            (self.target_tokenizer, self.tokenizer),
            (self.target_encoder, self.context_encoder),
        ):
            for tp, cp in zip(target.parameters(), online.parameters()):
                tp.mul_(momentum).add_(cp.detach(), alpha=1.0 - momentum)
            for tb, cb in zip(target.buffers(), online.buffers()):
                tb.copy_(cb)

    # --- pretraining step ------------------------------------------------
    def forward(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        validate_eeg_batch(batch)
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
        sample_mask = batch.get("sample_mask")

        grid = self.tokenizer(signal, ch_ids, ch_pos, ch_mask, sample_mask)
        B, C, Tp = grid.shape
        patch_mask = grid.token_mask.view(B, C, Tp).any(1)
        ctx_mask, tgt_mask = make_jepa_masks(
            ch_pos, ch_mask, Tp, self.cfg.mask, generator=generator,
            time_mask=patch_mask,
        )

        # --- targets: EMA encoder on the full (unmasked) valid view ---
        with torch.no_grad():
            tgt_grid = self.target_tokenizer(
                signal, ch_ids, ch_pos, ch_mask, sample_mask
            )
            tgt_repr = self.target_encoder(
                tgt_grid.tokens, tgt_grid.ch_index, tgt_grid.time_index,
                tgt_grid.token_mask, ch_pos,
            )
            tgt_repr = F.layer_norm(tgt_repr, (tgt_repr.shape[-1],))  # stabilize targets

        # --- context: hide target tokens (zero embedding + attention mask) ---
        ctx_tokens = grid.tokens * ctx_mask.unsqueeze(-1)
        ctx_repr = self.context_encoder(
            ctx_tokens, grid.ch_index, grid.time_index, ctx_mask, ch_pos
        )

        # --- predict target latents ---
        pos_emb = self.tokenizer.position_embeddings(ch_ids, ch_pos, Tp)
        pred = self.predictor(ctx_repr, pos_emb, ctx_mask, tgt_mask, (C, Tp))

        # pred/tgt_repr: (B, N, D); tgt_mask: (B, N) -> select (n_tgt, D)
        pred_t = pred[tgt_mask]                       # (n_tgt, D) online side
        loss = F.smooth_l1_loss(pred_t, tgt_repr[tgt_mask])
        out = {"loss_pred": loss, "loss": loss}

        # --- spectral auxiliary: predict log band-power of masked patches ---
        if self.spectral_head is not None and tgt_mask.any():
            windows = self.tokenizer.spectral_windows(signal, Tp)
            spec_tgt = band_power_target(
                windows,
                self.cfg.model.sample_rate,
                self.cfg.spectral_aux_bands,
            )
            spec_tgt = spec_tgt.reshape(B, C * Tp, -1)            # (B,N,n_bands)
            spec_tgt = F.layer_norm(spec_tgt, (spec_tgt.shape[-1],))
            spec_pred = self.spectral_head(pred)                  # (B,N,n_bands)
            spec_loss = F.mse_loss(spec_pred[tgt_mask], spec_tgt[tgt_mask])
            out["loss_spec"] = spec_loss
            out["loss"] = out["loss"] + self.cfg.spectral_aux * spec_loss

        # anti-collapse (VICReg): regularize the CONTEXT-ENCODER embeddings (the
        # tensor used downstream + tracked by the collapse monitor) -- not the
        # predictor output, which can stay diverse while the encoder collapses.
        enc_emb = ctx_repr[ctx_mask]                  # (n_ctx, D)
        if enc_emb.shape[0] > 1:
            xc = enc_emb - enc_emb.mean(0, keepdim=True)
            n, D = xc.shape
            if self.cfg.var_reg > 0:
                std = torch.sqrt(xc.var(dim=0) + 1e-4)
                var_loss = F.relu(self.cfg.var_target - std).mean()
                out["loss_var"] = var_loss
                out["loss"] = out["loss"] + self.cfg.var_reg * var_loss
            if self.cfg.cov_reg > 0:
                cov = (xc.t() @ xc) / (n - 1)                 # (D, D)
                off = cov - torch.diag(torch.diagonal(cov))   # zero the diagonal
                cov_loss = off.pow(2).sum() / D               # decorrelation penalty
                out["loss_cov"] = cov_loss
                out["loss"] = out["loss"] + self.cfg.cov_reg * cov_loss

        # --- optional domain-invariance ---
        if self.domain_head is not None:
            w = ctx_mask.float().unsqueeze(-1)
            pooled = (ctx_repr * w).sum(1) / w.sum(1).clamp_min(1.0)
            dlogits = self.domain_head(pooled)
            dloss = F.cross_entropy(dlogits, batch["domain"])
            out["loss_domain"] = dloss
            out["loss"] = out["loss"] + dloss

        # diagnostics
        out["_target_embeddings"] = tgt_repr[tgt_mask].detach()
        out["n_targets"] = int(tgt_mask.sum())
        out["n_context"] = int(ctx_mask.sum())
        return out

    def encode_features(self, batch: dict, pool: str = "mean") -> torch.Tensor:
        """Encoder sample embedding (grad flows -> usable for fine-tuning).

        pool="mean": masked mean over all tokens -> (B, D). Montage-agnostic, but
            averages over channels (drops spatial/lateralization patterns).
        pool="spatial": time-pool per channel, then map electrodes onto fixed
            spherical anchors -> (B, anchors*D). Preserves scalp structure and
            works across arbitrary channel counts and channel ordering.
        pool="chan": time-pool per channel, flatten -> (B, C*D). Keeps per-channel
            structure (e.g. C3/C4 ERD lateralization) -> better for motor imagery.
            Requires a FIXED montage across the eval set.
        """
        validate_eeg_batch(batch)
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
        grid = self.tokenizer(
            signal, ch_ids, ch_pos, ch_mask, batch.get("sample_mask")
        )
        B, C, Tp = grid.shape
        repr_ = self.context_encoder(
            grid.tokens, grid.ch_index, grid.time_index, grid.token_mask, ch_pos
        )
        if pool == "mean":
            w = grid.token_mask.float().unsqueeze(-1)
            return (repr_ * w).sum(1) / w.sum(1).clamp_min(1.0)
        if pool in {"chan", "spatial"}:
            r = repr_.view(B, C, Tp, -1)
            m = grid.token_mask.view(B, C, Tp, 1).to(r.dtype)
            chan = (r * m).sum(2) / m.sum(2).clamp_min(1.0)
            if pool == "spatial":
                return self.spatial_pool(chan, ch_pos, ch_mask)
            return chan.reshape(B, C * chan.shape[-1])
        raise ValueError(f"unknown pool {pool!r}")

    @torch.no_grad()
    def encode(self, batch: dict, pool: str = "mean") -> torch.Tensor:
        """Frozen-encoder embedding (no grad) for the linear probe."""
        return self.encode_features(batch, pool)

    def collapse_report(self, out: dict):
        return collapse_stats(out["_target_embeddings"])

    def feature_parameters(self):
        """Parameters used by downstream encoding/fine-tuning."""
        return list(self.tokenizer.parameters()) + list(self.context_encoder.parameters())
