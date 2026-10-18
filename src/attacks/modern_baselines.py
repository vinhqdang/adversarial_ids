"""
Recent (2022-2024) constraint-aware baselines, re-implemented against this repository's
12-feature problem-space schema so every method faces identical invariants.

1. CAPGDTransferAttack  - Constrained Adaptive PGD (Simonetto et al., NeurIPS 2024). Gradients
   come from a differentiable surrogate MLP; after every step the candidate is repaired onto the
   flow manifold with ``project_network_invariants``. The tree-ensemble target is only queried
   (black-box), once per iteration, and the attack stops at the first evasive candidate.
2. NIDSGANAttack        - Learned perturbation generator in the style of NIDSGAN
   (Zolbayar et al., 2022): a network trained against the surrogate emits a bounded delta that is
   repaired onto the manifold; the target is queried once per noise draw.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import AttackResult, BaseNIDSAttack
from ..datasets.schema import (NetworkFeatureSchema, project_network_invariants,
                               verify_physical_feasibility)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass(frozen=True)
class LogStandardizer:
    """z = (log1p(x) - mu) / sd, an invertible map for non-negative heavy-tailed flow features."""
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> 'LogStandardizer':
        L = np.log1p(np.maximum(X, 0.0))
        return cls(L.mean(axis=0), L.std(axis=0) + 1e-6)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return (np.log1p(np.maximum(X, 0.0)) - self.mu) / self.sd

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return np.expm1(Z * self.sd + self.mu)


def train_surrogate(X: np.ndarray, y: np.ndarray, norm: LogStandardizer,
                    epochs: int = 6, max_rows: int = 1_000_000, seed: int = 42) -> nn.Module:
    """Fit a small MLP surrogate (the attacker's own model, trained on public flow data)."""
    rng = np.random.RandomState(seed)
    if len(X) > max_rows:
        sel = rng.choice(len(X), max_rows, replace=False)
        X, y = X[sel], y[sel]
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128),
                        nn.ReLU(), nn.Linear(128, 1)).to(DEVICE)
    Z = torch.tensor(norm.forward(X), dtype=torch.float32, device=DEVICE)
    t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Z), device=DEVICE)
        for i in range(0, len(Z), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss_fn(net(Z[b]).squeeze(1), t[b]).backward()
            opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _mutable_mask(schema: NetworkFeatureSchema) -> np.ndarray:
    mask = np.ones(len(schema.feature_names))
    mask[list(schema.immutable_indices)] = 0.0
    return mask


class _SurrogateAttack(BaseNIDSAttack):
    def __init__(self, model, schema: NetworkFeatureSchema, surrogate: nn.Module,
                 norm: LogStandardizer):
        super().__init__(model, schema)
        self.surrogate = surrogate
        self.norm = norm
        self.mask = _mutable_mask(schema)

    def _repair(self, x: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        return project_network_invariants(x, self.schema, x_orig)

    def _result(self, x_orig: np.ndarray, x_adv: np.ndarray, success: bool) -> AttackResult:
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_adv - x_orig)))
        return AttackResult(
            x_original=x_orig, x_adversarial=x_adv, success=success,
            physically_valid=verify_physical_feasibility(x_orig, x_adv, self.schema),
            queries_used=self.query_count, weighted_cost=cost, feature_shifts=x_adv - x_orig)


class CAPGDTransferAttack(_SurrogateAttack):
    """Constrained APGD (momentum, step-size halving) on a surrogate with per-step repair."""

    def __init__(self, model, schema, surrogate, norm, eps: float = 1.0,
                 max_iter: int = 20, momentum: float = 0.75):
        super().__init__(model, schema, surrogate, norm)
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()
        z0 = self.norm.forward(x_orig[None])[0]
        mask = torch.tensor(self.mask, dtype=torch.float32, device=DEVICE)
        step = 2.0 * self.eps
        z, prev_z = z0.copy(), z0.copy()
        best_loss, best_x, since_improve = np.inf, x_orig.copy(), 0

        for _ in range(self.max_iter):
            zt = torch.tensor(z, dtype=torch.float32, device=DEVICE, requires_grad=True)
            logit = self.surrogate(zt[None]).squeeze()
            logit.backward()
            grad = zt.grad * mask
            z_new = z - step * np.sign(grad.cpu().numpy())
            z_new = np.clip(z_new, z0 - self.eps, z0 + self.eps)
            z_mom = z_new + self.momentum * (z_new - z) + (1 - self.momentum) * (z - prev_z)
            z_mom = np.clip(z_mom, z0 - self.eps, z0 + self.eps)
            prev_z = z
            x_cand = self._repair(self.norm.inverse(z_mom[None])[0], x_orig)
            z = self.norm.forward(x_cand[None])[0]

            if self.predict_malicious_prob(x_cand) < 0.5:
                return self._result(x_orig, x_cand, True)
            with torch.no_grad():
                loss = float(self.surrogate(torch.tensor(z, dtype=torch.float32,
                                                         device=DEVICE)[None]).item())
            if loss < best_loss:
                best_loss, best_x, since_improve = loss, x_cand, 0
            else:
                since_improve += 1
                if since_improve >= 2:
                    step, since_improve = step / 2.0, 0
        return self._result(x_orig, best_x, False)


class NIDSGANAttack(_SurrogateAttack):
    """Amortised generator G(z, noise) -> bounded delta, trained against the surrogate."""

    def __init__(self, model, schema, surrogate, norm, generator: nn.Module,
                 eps: float = 1.0, n_draws: int = 5):
        super().__init__(model, schema, surrogate, norm)
        self.generator = generator
        self.eps = eps
        self.n_draws = n_draws

    @staticmethod
    def train_generator(X_mal: np.ndarray, surrogate: nn.Module, norm: LogStandardizer,
                        schema: NetworkFeatureSchema, eps: float = 1.0, epochs: int = 8,
                        l1_weight: float = 0.05, seed: int = 42) -> nn.Module:
        torch.manual_seed(seed)
        d = X_mal.shape[1]
        gen = nn.Sequential(nn.Linear(d + 8, 128), nn.ReLU(), nn.Linear(128, 128),
                            nn.ReLU(), nn.Linear(128, d), nn.Tanh()).to(DEVICE)
        mask = torch.tensor(_mutable_mask(schema), dtype=torch.float32, device=DEVICE)
        Z = torch.tensor(norm.forward(X_mal), dtype=torch.float32, device=DEVICE)
        opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(Z), device=DEVICE)
            for i in range(0, len(Z), 2048):
                zb = Z[perm[i:i + 2048]]
                noise = torch.randn(len(zb), 8, device=DEVICE)
                delta = eps * mask * gen(torch.cat([zb, noise], 1))
                loss = (surrogate(zb + delta).squeeze(1).mean()
                        + l1_weight * delta.abs().sum(1).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
        gen.eval()
        for p in gen.parameters():
            p.requires_grad_(False)
        return gen

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()
        z0 = torch.tensor(self.norm.forward(x_orig[None]), dtype=torch.float32, device=DEVICE)
        mask = torch.tensor(self.mask, dtype=torch.float32, device=DEVICE)
        x_last = x_orig
        for _ in range(self.n_draws):
            noise = torch.randn(1, 8, device=DEVICE)
            delta = self.eps * mask * self.generator(torch.cat([z0, noise], 1))
            z_adv = (z0 + delta).cpu().numpy()
            x_last = self._repair(self.norm.inverse(z_adv)[0], x_orig)
            if self.predict_malicious_prob(x_last) < 0.5:
                return self._result(x_orig, x_last, True)
        return self._result(x_orig, x_last, False)
