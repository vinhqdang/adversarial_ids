"""The v2 hardening recipe (library-agnostic): certified monotone gate + hardened free classifier.

Components
* ``train_gate``  - capacity-boosted tree ensemble that (i) never sees the window, rate and mean-length
  columns and (ii) is constrained to be non-decreasing in the four columns the schema lets an attacker
  only increase (forward packets, forward bytes, duration, transport-header bytes). Every
  schema-feasible evasion moves those columns up, immutable columns stay fixed, so the gate score can
  never fall: a flow the gate flags stays flagged under ANY feasible transformation (certificate).
* ``train_orbit``/``train_adversarial`` - the free classifier trained on physically transformed copies of
  both classes (invariance by augmentation) and, for ``train_adversarial``, on flows crafted by
  FFE and the white-box attack against the current model each round, with their true labels.
* ``fuse`` - OR of the free classifier and the gate shifted to fire at validation-chosen score ``tau``.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.defenses.attackers import AttackContext, craft_flows
from src.defenses.augment import iterative_adversarial_training
from src.defenses.models import FusedModel, OffsetMargin, RobustTreeModel

GATE_HIDDEN = (7, 8, 9, 10)                    # window, rates, mean length: movable or derived
GATE_MONOTONE = {2: 1, 4: 1, 6: 1, 11: 1}      # attacker can only increase these
MPS_HIDDEN = (6, 8, 9)                         # old MPS: duration and the two rates
GATE_CAPACITY: Dict[str, dict] = {
    'xgboost': {'n_estimators': 300, 'max_depth': 8},
    'lightgbm': {'n_estimators': 300, 'max_depth': 8, 'num_leaves': 200},
    'catboost': {'iterations': 300, 'depth': 8},
    'random_forest': {'n_estimators': 100, 'max_depth': 16},
}
ORBIT_ROWS = 100_000
CRAFT_ROWS = 3000
ROUNDS = 2
LGAT_ROWS = 5000


def train_gate(library: str, X: np.ndarray, y: np.ndarray) -> RobustTreeModel:
    return RobustTreeModel(library, hidden=GATE_HIDDEN, monotone=GATE_MONOTONE,
                           params=GATE_CAPACITY[library]).train(X, y)


def _craft_fn(ctx: AttackContext, n_jobs: int) -> Callable:
    plan = (('ffe_m5', 'ffe_m20'), ('ffe_m20', 'whitebox'))   # attack families rotate with the round

    def craft(model, X_mal: np.ndarray, rnd: int) -> np.ndarray:
        first, second = plan[rnd % len(plan)]
        h = len(X_mal) // 2
        return np.vstack([craft_flows(first, model, ctx, X_mal[:h], n_jobs),
                          craft_flows(second, model, ctx, X_mal[h:], n_jobs)])
    return craft


def train_orbit(library: str, X: np.ndarray, y: np.ndarray, seed: int = 42,
                aug_weight: float = 1.0) -> RobustTreeModel:
    """Attack-agnostic invariance augmentation only (no attack is run); ``aug_weight`` softens it."""
    model, _ = iterative_adversarial_training(
        lambda a, b, w: RobustTreeModel(library).train(a, b, w), X, y, craft_fn=None,
        rounds=0, n_orbit=ORBIT_ROWS, aug_weight=aug_weight, seed=seed)
    return model


def train_adversarial(library: str, X: np.ndarray, y: np.ndarray, base: RobustTreeModel,
                      ctx: AttackContext, n_jobs: int = 12, seed: int = 42) -> RobustTreeModel:
    """Iterative adversarial training (FFE M=5/20 and white-box, true labels) plus orbit augmentation."""
    model, _ = iterative_adversarial_training(
        lambda a, b, w: RobustTreeModel(library).train(a, b, w), X, y, _craft_fn(ctx, n_jobs),
        rounds=ROUNDS, n_craft=CRAFT_ROWS, n_orbit=ORBIT_ROWS, seed=seed, base_model=base)
    return model


def train_lgat(library: str, X: np.ndarray, y: np.ndarray, base: RobustTreeModel, ctx: AttackContext,
               hidden: Tuple[int, ...] = (), n_jobs: int = 12, seed: int = 42) -> RobustTreeModel:
    """Old LGAT: ``LGAT_ROWS`` FFE M=5 flows crafted against ``base`` and added with label 1."""
    rng = np.random.RandomState(seed)
    mal = np.where(y == 1)[0]
    chosen = rng.choice(mal, min(LGAT_ROWS, int(0.5 * len(mal))), replace=False)
    aug = craft_flows('ffe_m5', base, ctx, X[chosen], n_jobs)
    return RobustTreeModel(library, hidden=hidden).train(
        np.vstack([X, aug]), np.concatenate([y, np.ones(len(aug), dtype=int)]))


def fuse(free, gate: RobustTreeModel, tau: Optional[float]):
    """OR-fusion of ``free`` with ``gate`` firing at score ``tau`` (None = no gate)."""
    if tau is None:
        return free
    return FusedModel(free, OffsetMargin(gate, float(np.log(tau / (1.0 - tau)))))
