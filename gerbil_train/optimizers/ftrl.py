"""FTRL-Proximal optimizer for online/streaming learning.

Implements the FTRL-Proximal algorithm from McMahan et al. (2013):

    w_{t+1} = 0                                          if |z_t| ≤ λ1
    w_{t+1} = -( (β + √n_t) / α + λ₂ )⁻¹ · (z_t - sign(z_t)·λ1)  otherwise

where:
    g_t   = gradient at step t
    σ_t   = (√(n_t + g_t²) - √n_t) / α
    z_t   = z_{t-1} + g_t - σ_t · w_t
    n_t   = n_{t-1} + g_t²
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class FTRL(Optimizer):
    """FTRL-Proximal optimizer with per-coordinate learning rates and L1 sparsity.

    :param params: Model parameters to optimize
    :param alpha: Per-coordinate learning rate (higher = slower)
    :param beta: Smoothing parameter
    :param lambda1: L1 regularization (higher = sparser)
    :param lambda2: L2 regularization
    """

    def __init__(
        self,
        params,
        alpha: float = 0.1,
        beta: float = 1.0,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if beta <= 0:
            raise ValueError(f"Invalid beta: {beta}")
        if lambda1 < 0:
            raise ValueError(f"Invalid lambda1: {lambda1}")
        if lambda2 < 0:
            raise ValueError(f"Invalid lambda2: {lambda2}")

        defaults = dict(alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2)
        super().__init__(params, defaults)

        # Per-parameter state: z (accumulated gradient - regularization) and n (squared gradient sum)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["z"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["n"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["alpha"]
            beta = group["beta"]
            lambda1 = group["lambda1"]
            lambda2 = group["lambda2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                z = state["z"]
                n = state["n"]

                # n_t = n_{t-1} + g_t²
                # sigma = (√(n + g²) - √n) / alpha
                n_new = n + grad * grad
                sigma = (n_new.sqrt() - n.sqrt()) / alpha

                # z_t = z_{t-1} + g_t - sigma · w_t
                z.add_(grad - sigma * p)

                # Update n
                n.copy_(n_new)

                # Apply proximal operator: clamp to zero if |z| <= lambda1
                # w = -( (beta + √n) / alpha + lambda2 )⁻¹ · (z - sign(z)·lambda1)
                p.copy_(
                    -((beta + n_new.sqrt()) / alpha + lambda2).reciprocal()
                    * (z.sign() * lambda1).sub(z).sign()
                    * (z.abs() - lambda1).clamp(min=0)
                )

        return loss
