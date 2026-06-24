"""Shared pretraining engine used by synthetic, MOABB, and evaluations."""

from __future__ import annotations

import contextlib
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..augment import augment_batch
from ..collapse import CollapseStats, collapse_stats
from ..data import collate_variable_montage
from .utils import ema_at, lr_at, move


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_pred: float | None
    collapse: CollapseStats | None
    selection_score: float | None


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class PretrainTrainer:
    def __init__(
        self,
        model,
        cfg,
        device: torch.device,
        amp: bool = False,
        workers: int = 0,
    ):
        cfg.validate()
        self.model = model
        self.cfg = cfg
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.workers = workers
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.step = 0
        self.epoch = 0

    def _amp_context(self):
        if self.amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def loader(self, ds, shuffle: bool, epoch_seed: int = 0) -> DataLoader:
        gen = torch.Generator().manual_seed(self.cfg.seed + epoch_seed)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(ds) >= self.cfg.batch_size,
            collate_fn=collate_variable_montage,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            # Loaders are rebuilt with an epoch-specific seed for exact resume.
            persistent_workers=False,
            worker_init_fn=_seed_worker if self.workers else None,
            generator=gen,
        )

    @torch.no_grad()
    def evaluate(self, ds) -> tuple[float, CollapseStats | None]:
        loader = self.loader(ds, shuffle=False, epoch_seed=100_000)
        self.model.eval()
        total, count, embeddings = 0.0, 0, []
        # Reset masks each evaluation so checkpoints are compared on the same views.
        mask_gen = torch.Generator().manual_seed(self.cfg.seed + 200_000)
        for batch in loader:
            with self._amp_context():
                out = self.model(move(batch, self.device), generator=mask_gen)
            total += float(out["loss_pred"].detach())
            count += 1
            if sum(x.shape[0] for x in embeddings) < 4096:
                embeddings.append(out["_target_embeddings"].detach().float().cpu())
        stats = collapse_stats(torch.cat(embeddings)) if embeddings else None
        return total / max(1, count), stats

    @staticmethod
    def selection_score(pred_loss: float, stats: CollapseStats | None) -> float:
        if stats is None or stats.collapsed:
            return float("inf")
        # Prediction loss alone rewards constant, trivially predictable targets.
        collapse_penalty = (
            max(0.0, 0.02 - stats.embed_std) * 100.0
            + max(0.0, 0.15 - stats.rank_ratio) * 20.0
        )
        return pred_loss + collapse_penalty

    def fit(
        self,
        train_ds,
        val_ds=None,
        patience: int = 0,
        verbose: bool = True,
    ) -> list[EpochResult]:
        sizing_loader = self.loader(train_ds, shuffle=False)
        if len(sizing_loader) == 0:
            raise ValueError("pretraining dataset produced no batches")
        steps_per_epoch = len(sizing_loader)
        total_steps = self.cfg.epochs * steps_per_epoch
        warmup = self.cfg.warmup_epochs * steps_per_epoch
        best_score, best_state, best_optimizer = float("inf"), None, None
        best_step, best_epoch, no_improve = 0, 0, 0
        history = []

        for epoch in range(self.epoch, self.cfg.epochs):
            train_loader = self.loader(train_ds, shuffle=True, epoch_seed=epoch)
            mask_gen = torch.Generator().manual_seed(
                self.cfg.seed + 10_000 * epoch
            )
            aug_gen = torch.Generator().manual_seed(
                self.cfg.seed + 30_000 * epoch
            )
            self.model.train()
            running = 0.0
            for batch in train_loader:
                if self.cfg.augment.enabled:
                    batch = augment_batch(
                        batch, self.cfg.augment, self.cfg.model.sample_rate,
                        generator=aug_gen,
                    )
                lr = lr_at(self.step, total_steps, warmup, self.cfg.lr)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                with self._amp_context():
                    out = self.model(move(batch, self.device), generator=mask_gen)
                self.optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
                self.optimizer.step()
                self.model.update_target(
                    ema_at(
                        self.step, total_steps,
                        self.cfg.ema_base, self.cfg.ema_final,
                    )
                )
                running += float(out["loss"].detach())
                self.step += 1

            val_pred, stats, score = None, None, None
            improved = False
            if val_ds is not None:
                val_pred, stats = self.evaluate(val_ds)
                score = self.selection_score(val_pred, stats)
                if score < best_score:
                    best_score, no_improve, improved = score, 0, True
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    best_optimizer = copy.deepcopy(self.optimizer.state_dict())
                    best_step, best_epoch = self.step, epoch + 1
                else:
                    no_improve += 1
            result = EpochResult(
                epoch, running / len(train_loader), val_pred, stats, score
            )
            history.append(result)
            self.epoch = epoch + 1
            if verbose:
                val = (
                    f" val_pred={val_pred:.4f} score={score:.4f} | {stats}"
                    if val_pred is not None else ""
                )
                mark = " <-- best valid representation" if improved else ""
                print(
                    f"  ep{epoch:02d} train_loss={result.train_loss:.4f}"
                    f"{val}{mark}"
                )
            if patience and val_ds is not None and no_improve >= patience:
                if verbose:
                    print(f"  early stop after {patience} non-improving epochs")
                break

        if val_ds is not None and best_state is None:
            raise RuntimeError(
                "pretraining produced no non-collapsed validation checkpoint; "
                "downstream evaluation and checkpoint saving have been aborted"
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.optimizer.load_state_dict(best_optimizer)
            self.step, self.epoch = best_step, best_epoch
            if verbose:
                print(
                    f"  restored best non-collapsed checkpoint @ ep"
                    f"{best_epoch - 1:02d} step={best_step} score={best_score:.4f}"
                )
        return history

    def save(self, path, epoch: int | None = None, **metadata) -> None:
        from .checkpoint import save_checkpoint

        save_checkpoint(
            path, self.model, self.cfg, optimizer=self.optimizer,
            step=self.step, epoch=self.epoch if epoch is None else epoch, **metadata,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path,
        device: torch.device,
        amp: bool = False,
        workers: int = 0,
    ):
        from .checkpoint import load_checkpoint, restore_rng_state

        model, cfg, blob, metadata, missing, unexpected = load_checkpoint(
            path, device
        )
        trainer = cls(model, cfg, device, amp=amp, workers=workers)
        if blob.get("optimizer"):
            trainer.optimizer.load_state_dict(blob["optimizer"])
        trainer.step = int(blob.get("step", 0))
        trainer.epoch = int(blob.get("epoch", 0))
        restore_rng_state(blob.get("rng_state"))
        return trainer, metadata, missing, unexpected
