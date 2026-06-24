"""Hierarchical EEG-JEPA.

A temporal pyramid of JEPA levels. The encoder is a stack of stages; between
stages tokens are mean-pooled along time, so each level sees a coarser, more
abstract view:

  level 0 : finest patches (micro-dynamics)
  level 1 : pooled x2 (rhythms / ERD)
  level 2 : pooled x4 (trial / state)

A SINGLE mask is sampled at the finest level and propagated UPWARD by pooling
(a coarse token is context/target only if ALL its fine children are). Both the
online (context) tower and the EMA target tower run the full pyramid; at EVERY
level a predictor forecasts the EMA target's latent at masked positions. Loss is
the sum of per-level latent-prediction (+VICReg) terms -> abstraction is learned
at multiple scales, not just one.

Exposes the same interface as EEGJepa (forward/update_target/encode/
collapse_report) so the training scripts can swap it in with a flag.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collapse import collapse_stats
from .config import PretrainConfig
from .data.schema import validate_eeg_batch
from .domain import DomainClassifier
from .encoder import MultiScaleEncoder
from .jepa import band_power_target
from .masking import make_jepa_masks
from .pooling import SpatialAnchorPool
from .predictor import Predictor
from .tokenizer import EEGTokenizer


def _pool_time(tokens, C, Tp, factor, token_mask=None):
    """(B, C*Tp, D) -> (B, C*Tp', D) mean-pooling adjacent time patches per channel."""
    B, N, D = tokens.shape
    Tp2 = Tp // factor
    x = tokens.view(B, C, Tp, D)[:, :, : Tp2 * factor]
    x = x.view(B, C, Tp2, factor, D)
    if token_mask is None:
        pooled = x.mean(3)
        pooled_mask = torch.ones(B, C, Tp2, dtype=torch.bool, device=tokens.device)
    else:
        m = token_mask[:, :, : Tp2 * factor].view(B, C, Tp2, factor)
        w = m.to(x.dtype).unsqueeze(-1)
        pooled = (x * w).sum(3) / w.sum(3).clamp_min(1.0)
        pooled_mask = m.any(3)
    return pooled.reshape(B, C * Tp2, D), Tp2, pooled_mask


def _level_indices(C, Tp, ch_mask, device):
    """Index/mask tensors for a (C, Tp) grid flattened channel-major."""
    B = ch_mask.shape[0]
    ch_index = torch.arange(C, device=device).view(1, C, 1).expand(B, C, Tp)
    time_index = torch.arange(Tp, device=device).view(1, 1, Tp).expand(B, C, Tp)
    token_mask = ch_mask.unsqueeze(2).expand(B, C, Tp)
    return (ch_index.reshape(B, -1), time_index.reshape(B, -1),
            token_mask.reshape(B, -1))


def _pool_mask(fine, factor):
    """(B, C, Tp0) bool -> (B, C, Tp0/factor) bool, True only if ALL children True."""
    B, C, Tp = fine.shape
    Tp2 = Tp // factor
    f = fine[:, :, : Tp2 * factor].view(B, C, Tp2, factor)
    return f.all(-1)


class HierarchicalEEGJepa(nn.Module):
    def __init__(self, cfg: PretrainConfig, n_levels: int = 3, pool_factor: int = 2):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.n_levels = n_levels
        self.pool_factor = pool_factor
        mcfg = cfg.model
        self.tokenizer = EEGTokenizer(mcfg)
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        for p in self.target_tokenizer.parameters():
            p.requires_grad_(False)
        self.spatial_pool = SpatialAnchorPool(mcfg.pool_anchors)

        # Split depth across levels while preserving the requested total depth.
        base, rem = divmod(max(mcfg.depth, n_levels), n_levels)
        depths = [base + int(i < rem) for i in range(n_levels)]
        stages = []
        for depth in depths:
            stage_cfg = copy.deepcopy(mcfg)
            stage_cfg.depth = depth
            stages.append(MultiScaleEncoder(stage_cfg))
        self.online = nn.ModuleList(stages)
        self.target = nn.ModuleList([copy.deepcopy(e) for e in self.online])
        for e in self.target:
            for p in e.parameters():
                p.requires_grad_(False)
        self.predictors = nn.ModuleList([
            Predictor(mcfg.dim, cfg.pred_dim, cfg.pred_depth, cfg.pred_heads)
            for _ in range(n_levels)
        ])
        self.domain_head = (
            DomainClassifier(mcfg.dim, cfg.n_domains, cfg.domain_lambda)
            if cfg.use_domain_invariance else None
        )
        self.spectral_head = (
            nn.Linear(mcfg.dim, len(cfg.spectral_aux_bands))
            if cfg.spectral_aux > 0 else None
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for tp, op in zip(self.target_tokenizer.parameters(), self.tokenizer.parameters()):
            tp.mul_(momentum).add_(op.detach(), alpha=1.0 - momentum)
        for tb, ob in zip(self.target_tokenizer.buffers(), self.tokenizer.buffers()):
            tb.copy_(ob)
        for te, oe in zip(self.target, self.online):
            for tp, op in zip(te.parameters(), oe.parameters()):
                tp.mul_(momentum).add_(op.detach(), alpha=1.0 - momentum)
            for tb, ob in zip(te.buffers(), oe.buffers()):
                tb.copy_(ob)

    def forward(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        validate_eeg_batch(batch)
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
        dev = signal.device
        sample_mask = batch.get("sample_mask")
        grid = self.tokenizer(signal, ch_ids, ch_pos, ch_mask, sample_mask)
        with torch.no_grad():
            target_grid = self.target_tokenizer(
                signal, ch_ids, ch_pos, ch_mask, sample_mask
            )
        B, C, Tp0 = grid.shape
        valid0 = grid.token_mask.view(B, C, Tp0)
        patch_mask = valid0.any(1)

        # finest mask, then propagate upward
        ctx0, tgt0 = make_jepa_masks(
            ch_pos, ch_mask, Tp0, self.cfg.mask, generator, patch_mask
        )
        ctx0 = ctx0.view(B, C, Tp0)
        tgt0 = tgt0.view(B, C, Tp0)

        ctx_tokens = grid.tokens * ctx0.reshape(B, C * Tp0, 1)   # hide targets (online)
        full_tokens = target_grid.tokens
        valid_level = valid0

        # cap levels so the coarsest still has >=2 time patches to mask
        n_eff = 1
        while n_eff < self.n_levels and Tp0 // (self.pool_factor ** n_eff) >= 2:
            n_eff += 1

        total = torch.zeros((), device=dev)
        pred_total = torch.zeros((), device=dev)
        out = {}
        Tp = Tp0
        finest_tgt_emb = None
        finest_ctx_repr = finest_pred = None
        for l in range(n_eff):
            F_l = self.pool_factor ** l                 # cumulative pooling so far
            ctx_m = _pool_mask(ctx0, F_l).reshape(B, -1)
            tgt_m = _pool_mask(tgt0, F_l).reshape(B, -1)
            valid_m = valid_level.reshape(B, -1)
            ch_index, time_index, _ = _level_indices(C, Tp, ch_mask, dev)

            # target (EMA, stop-grad) on full view
            with torch.no_grad():
                tgt_repr = self.target[l](
                    full_tokens, ch_index, time_index, valid_m, ch_pos
                )
                tgt_repr = F.layer_norm(tgt_repr, (tgt_repr.shape[-1],))
            # online context
            ctx_repr = self.online[l](
                ctx_tokens, ch_index, time_index, ctx_m, ch_pos
            )

            pos_emb = self.tokenizer.position_embeddings(ch_ids, ch_pos, Tp)
            pred = self.predictors[l](
                ctx_repr, pos_emb, ctx_m, tgt_m, (C, Tp)
            )

            if tgt_m.any():
                # VICReg on the ENCODER output (context tokens) -- regularizing the
                # predictor output does NOT prevent encoder collapse.
                pred_l = F.smooth_l1_loss(pred[tgt_m], tgt_repr[tgt_m])
                loss_l = pred_l + self._vicreg(ctx_repr[ctx_m])
            else:
                pred_l = torch.zeros((), device=dev)
                loss_l = torch.zeros((), device=dev)
            total = total + loss_l
            pred_total = pred_total + pred_l
            out[f"loss_l{l}"] = loss_l.detach()
            if l == 0:
                finest_ctx_repr = ctx_repr
                finest_pred = pred
                finest_tgt_emb = (tgt_repr[tgt_m] if tgt_m.any()
                                  else tgt_repr.reshape(-1, tgt_repr.shape[-1])).detach()

            # Coarsen tokens for the next stage. Gradients from upper levels flow
            # through the online pyramid, making this a genuinely hierarchical tower.
            if l < n_eff - 1:
                ctx_tokens, _, _ = _pool_time(
                    ctx_repr, C, Tp, self.pool_factor,
                    ctx_m.view(B, C, Tp),
                )
                with torch.no_grad():
                    full_tokens, Tp, valid_level = _pool_time(
                        tgt_repr, C, Tp, self.pool_factor, valid_level
                    )

        out["loss_pred"] = pred_total / n_eff
        out["loss"] = total / n_eff
        if self.spectral_head is not None and tgt0.any():
            windows = self.tokenizer.spectral_windows(signal, Tp0)
            spec_tgt = band_power_target(
                windows,
                self.cfg.model.sample_rate,
                self.cfg.spectral_aux_bands,
            )
            spec_tgt = F.layer_norm(
                spec_tgt.reshape(B, C * Tp0, -1),
                (len(self.cfg.spectral_aux_bands),),
            )
            spec_loss = F.mse_loss(
                self.spectral_head(finest_pred)[tgt0.reshape(B, -1)],
                spec_tgt[tgt0.reshape(B, -1)],
            )
            out["loss_spec"] = spec_loss
            out["loss"] = out["loss"] + self.cfg.spectral_aux * spec_loss
        if self.domain_head is not None:
            ctx_flat = ctx0.reshape(B, -1)
            w = ctx_flat.to(finest_ctx_repr.dtype).unsqueeze(-1)
            pooled = (finest_ctx_repr * w).sum(1) / w.sum(1).clamp_min(1.0)
            domain_loss = F.cross_entropy(
                self.domain_head(pooled), batch["domain"]
            )
            out["loss_domain"] = domain_loss
            out["loss"] = out["loss"] + domain_loss
        out["n_levels_used"] = n_eff
        out["_target_embeddings"] = finest_tgt_emb
        out["n_targets"] = int(tgt0.sum())
        out["n_context"] = int(ctx0.sum())
        return out

    def _vicreg(self, x):
        if x.shape[0] < 2:
            return torch.zeros((), device=x.device)
        xc = x - x.mean(0, keepdim=True)
        n, D = xc.shape
        loss = torch.zeros((), device=x.device)
        if self.cfg.var_reg > 0:
            std = torch.sqrt(xc.var(0) + 1e-4)
            loss = loss + self.cfg.var_reg * F.relu(self.cfg.var_target - std).mean()
        if self.cfg.cov_reg > 0:
            cov = (xc.t() @ xc) / (n - 1)
            off = cov - torch.diag(torch.diagonal(cov))
            loss = loss + self.cfg.cov_reg * off.pow(2).sum() / D
        return loss

    def encode_features(self, batch: dict, pool: str = "mean") -> torch.Tensor:
        """Concatenate per-level pooled reps -> richer multi-scale feature.

        pool='mean': (B, D*n_levels). pool='chan': per-channel at every level,
        concatenated -> (B, C*D*n_levels) (keeps spatial structure).
        """
        validate_eeg_batch(batch)
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
        dev = signal.device
        grid = self.tokenizer(
            signal, ch_ids, ch_pos, ch_mask, batch.get("sample_mask")
        )
        B, C, Tp = grid.shape
        n_eff = 1
        while n_eff < self.n_levels and Tp // (self.pool_factor ** n_eff) >= 1:
            n_eff += 1
        x = grid.tokens
        valid_level = grid.token_mask.view(B, C, Tp)
        feats = []
        for l in range(n_eff):
            valid_m = valid_level.reshape(B, -1)
            ch_index, time_index, _ = _level_indices(C, Tp, ch_mask, dev)
            rep = self.online[l](
                x, ch_index, time_index, valid_m, ch_pos
            )
            r = rep.view(B, C, Tp, -1)
            m = valid_level.unsqueeze(-1).to(r.dtype)
            if pool in {"chan", "spatial"}:
                chan = (r * m).sum(2) / m.sum(2).clamp_min(1.0)
                if pool == "spatial":
                    feats.append(self.spatial_pool(chan, ch_pos, ch_mask))
                else:
                    feats.append(chan.reshape(B, -1))
            else:
                pooled = (r * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0)
                feats.append(pooled)
            if l < n_eff - 1:
                x, Tp, valid_level = _pool_time(
                    rep, C, Tp, self.pool_factor, valid_level
                )
        per_level_dim = (
            self.cfg.model.dim
            if pool == "mean"
            else C * self.cfg.model.dim
            if pool == "chan"
            else self.cfg.model.pool_anchors * self.cfg.model.dim
        )
        while len(feats) < self.n_levels:
            feats.append(torch.zeros(
                B, per_level_dim, device=signal.device, dtype=feats[0].dtype
            ))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def encode(self, batch: dict, pool: str = "mean") -> torch.Tensor:
        return self.encode_features(batch, pool)

    def collapse_report(self, out: dict):
        return collapse_stats(out["_target_embeddings"])

    def feature_parameters(self):
        return list(self.tokenizer.parameters()) + list(self.online.parameters())
