"""Optional domain-invariance objective for cross-site/device generalization.

A gradient-reversal domain classifier (DANN-style) tries to predict the source
site/device from the pooled context embedding; reversing its gradient pushes the
encoder toward site-invariant features.  Disabled by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class DomainClassifier(nn.Module):
    def __init__(self, dim: int, n_domains: int, lambd: float = 0.1):
        super().__init__()
        self.lambd = lambd
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_domains)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(grad_reverse(pooled, self.lambd))
