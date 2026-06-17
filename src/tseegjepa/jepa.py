"""EEG-JEPA: the full self-supervised model.

Data flow per step (I-JEPA-style, latent-space only -- no raw reconstruction):

  tokenize -> sample masks
  target encoder (EMA, stop-grad): encode UNMASKED view -> target latents
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
from .domain import DomainClassifier
from .encoder import MultiScaleEncoder
from .masking import make_jepa_masks
from .predictor import Predictor
from .tokenizer import EEGTokenizer


class EEGJepa(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model
        self.tokenizer = EEGTokenizer(mcfg)
        self.context_encoder = MultiScaleEncoder(mcfg)
        # EMA target encoder: structural copy, frozen, updated by momentum only.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(
            enc_dim=mcfg.dim, pred_dim=cfg.pred_dim,
            depth=cfg.pred_depth, heads=cfg.pred_heads,
        )
        self.domain_head = (
            DomainClassifier(mcfg.dim, cfg.n_domains, cfg.domain_lambda)
            if cfg.use_domain_invariance else None
        )

    # --- EMA -------------------------------------------------------------
    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.mul_(momentum).add_(cp.detach(), alpha=1.0 - momentum)
        for tb, cb in zip(self.target_encoder.buffers(),
                          self.context_encoder.buffers()):
            tb.copy_(cb)

    # --- pretraining step ------------------------------------------------
    def forward(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]

        grid = self.tokenizer(signal, ch_ids, ch_pos, ch_mask)
        B, C, Tp = grid.shape
        ctx_mask, tgt_mask = make_jepa_masks(
            ch_pos, ch_mask, Tp, self.cfg.mask, generator=generator
        )

        # --- targets: EMA encoder on the full (unmasked) valid view ---
        with torch.no_grad():
            tgt_repr = self.target_encoder(
                grid.tokens, grid.ch_index, grid.time_index, grid.token_mask
            )
            tgt_repr = F.layer_norm(tgt_repr, (tgt_repr.shape[-1],))  # stabilize targets

        # --- context: hide target tokens (zero embedding + attention mask) ---
        ctx_tokens = grid.tokens * ctx_mask.unsqueeze(-1)
        ctx_repr = self.context_encoder(
            ctx_tokens, grid.ch_index, grid.time_index, ctx_mask
        )

        # --- predict target latents ---
        pos_emb = self.tokenizer.position_embeddings(ch_ids, ch_pos, Tp)
        pred = self.predictor(ctx_repr, pos_emb, ctx_mask, tgt_mask)

        # pred/tgt_repr: (B, N, D); tgt_mask: (B, N) -> select (n_tgt, D)
        pred_t = pred[tgt_mask]                       # (n_tgt, D) online side
        loss = F.smooth_l1_loss(pred_t, tgt_repr[tgt_mask])
        out = {"loss_pred": loss, "loss": loss}

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

    @torch.no_grad()
    def encode(self, batch: dict, pool: str = "mean") -> torch.Tensor:
        """Frozen-encoder sample embedding for downstream use.

        pool="mean": masked mean over all tokens -> (B, D). Montage-agnostic, but
            averages over channels (drops spatial/lateralization patterns).
        pool="chan": time-pool per channel, flatten -> (B, C*D). Keeps per-channel
            structure (e.g. C3/C4 ERD lateralization) -> better for tasks like
            motor imagery. Requires a FIXED montage across the eval set.
        """
        signal = batch["signal"]
        ch_ids, ch_pos, ch_mask = batch["ch_ids"], batch["ch_pos"], batch["ch_mask"]
        grid = self.tokenizer(signal, ch_ids, ch_pos, ch_mask)
        B, C, Tp = grid.shape
        repr_ = self.context_encoder(
            grid.tokens, grid.ch_index, grid.time_index, grid.token_mask
        )
        if pool == "mean":
            w = grid.token_mask.float().unsqueeze(-1)
            return (repr_ * w).sum(1) / w.sum(1).clamp_min(1.0)
        if pool == "chan":
            r = repr_.view(B, C, Tp, -1)
            m = grid.token_mask.view(B, C, Tp, 1).float()
            chan = (r * m).sum(2) / m.sum(2).clamp_min(1.0)   # (B, C, D)
            return chan.reshape(B, C * chan.shape[-1])
        raise ValueError(f"unknown pool {pool!r}")

    def collapse_report(self, out: dict):
        return collapse_stats(out["_target_embeddings"])
