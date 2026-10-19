"""Pluggable attack registry and (parallel) cohort evaluation for the defense experiments.

An attack is registered as a factory ``factory(model, ctx) -> attack`` where ``attack`` has the
interface of ``FlowFieldEvasionAttack`` (``attack_sample(x) -> AttackResult``). A stronger attack
(e.g. FFE-v2) is dropped in with ``register_attack('ffe_v2', factory)``; ``run_defenses_v2.py
--plugin package.module`` imports a module that performs the registration. Factories must be
picklable or closure-free enough for cloudpickle, because CPU attacks run in worker processes.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.attacks.flow_field_evasion import FlowFieldEvasionAttack
from src.datasets.schema import NetworkFeatureSchema, get_cicids_schema
from src.defenses.whitebox_generic import GenericWhiteBoxAttack

SURROGATE_ROWS = 300_000


@dataclass
class AttackContext:
    schema: NetworkFeatureSchema
    stds: np.ndarray
    X_query: Optional[np.ndarray] = None       # public flows the CAPGD attacker labels with the target
    surrogates: Dict[int, tuple] = field(default_factory=dict)

    def surrogate_for(self, model):
        """Surrogate MLP trained on the defended model's own soft outputs (label-querying attacker)."""
        import src.attacks.modern_baselines as mb
        mb.torch.set_num_threads(2)
        mb.DEVICE = 'cpu'
        from src.attacks.modern_baselines import LogStandardizer, train_surrogate
        key = id(model)
        if key not in self.surrogates:
            labels = model.predict_proba(self.X_query)[:, 1]
            norm = LogStandardizer.fit(self.X_query)
            self.surrogates[key] = (train_surrogate(self.X_query, labels, norm, epochs=6), norm)
        return self.surrogates[key]


AttackFactory = Callable[[object, AttackContext], object]
ATTACKS: Dict[str, AttackFactory] = {}
SEQUENTIAL = {'capgd'}   # GPU surrogate: run in the main process


def register_attack(name: str, factory: AttackFactory, sequential: bool = False) -> None:
    ATTACKS[name] = factory
    if sequential:
        SEQUENTIAL.add(name)


def _ffe(budget: int) -> AttackFactory:
    return lambda model, ctx: FlowFieldEvasionAttack(model, ctx.schema, step_budget=budget, feature_stds=ctx.stds)


def _capgd(model, ctx: AttackContext):
    import src.attacks.modern_baselines as mb
    mb.torch.set_num_threads(2)
    mb.DEVICE = 'cpu'            # tiny MLP: CPU is faster than a time-sliced shared GPU and deterministic
    from src.attacks.modern_baselines import CAPGDTransferAttack
    surrogate, norm = ctx.surrogate_for(model)
    return CAPGDTransferAttack(model, ctx.schema, surrogate, norm)


register_attack('ffe_m5', _ffe(5))
register_attack('ffe_m20', _ffe(20))
register_attack('capgd', _capgd, sequential=True)
register_attack('whitebox', lambda model, ctx: GenericWhiteBoxAttack(model, ctx.schema))


def _run_chunk(factory: AttackFactory, model, ctx: AttackContext, X: np.ndarray) -> List[Tuple[bool, bool, int, np.ndarray]]:
    attack = factory(model, ctx)
    out = []
    for x in X:
        r = attack.attack_sample(x)
        out.append((bool(r.success), bool(r.physically_valid), int(r.queries_used), r.x_adversarial))
    return out


def run_attack(name: str, model, ctx: AttackContext, X: np.ndarray, n_jobs: int = 12) -> List[Tuple[bool, bool, int, np.ndarray]]:
    """Per-flow (success, feasible, queries, adversarial flow) for ``X`` (callers pass flows the model detects)."""
    if len(X) == 0:
        return []
    factory = ATTACKS[name]
    if hasattr(model, 'set_predict_threads'):
        model.set_predict_threads(1)
    if name in SEQUENTIAL or n_jobs <= 1:
        return _run_chunk(factory, model, ctx, X)
    from joblib import Parallel, delayed
    chunks = np.array_split(X, min(n_jobs * 2, len(X)))
    lite = AttackContext(ctx.schema, ctx.stds, None, {})
    res = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_run_chunk)(factory, model, lite, c) for c in chunks)
    return [r for part in res for r in part]


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval (in %) for a binomial proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def make_context(X_train: np.ndarray, seed: int = 42) -> AttackContext:
    rng = np.random.RandomState(seed)
    rows = rng.choice(len(X_train), min(SURROGATE_ROWS, len(X_train)), replace=False)
    return AttackContext(get_cicids_schema(), X_train.std(axis=0) + 1e-6, X_train[rows])


def craft_flows(name: str, model, ctx: AttackContext, X: np.ndarray, n_jobs: int = 12) -> np.ndarray:
    """Adversarial flows (the attack's final candidate, successful or not) for adversarial training."""
    res = run_attack(name, model, ctx, X, n_jobs)
    return np.array([r[3] for r in res]).reshape(len(res), -1)
