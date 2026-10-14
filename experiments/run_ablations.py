"""
Reproducible ablation studies for Flow-Field Evasion (paper Section 8, Ablations 1-4).

Evaluates FFE on a balanced cohort (20 detected flows per attack family) against the same
XGBoost target used by experiments/run_benchmark.py while sweeping:
  1. the step budget M,
  2. the enabled generator subset,
  3. the operational cost regime c,
  4. the TSBS relative snapping offset epsilon_rel.

Results are written to results/ablation_results.json.
"""

import json
import os
import sys
import time
from typing import Dict, FrozenSet, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.schema import get_cicids_schema
from src.datasets.generator import generate_cicids_traffic, generate_attack_family_cohorts
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack, ALL_GENERATORS

N_PER_FAMILY = 20
STEP_BUDGETS = (1, 2, 3, 5, 10, 20, 50)
EPSILONS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
GENERATOR_SUBSETS = {
    'Temporal Dilation Only': frozenset({'time'}),
    'Payload Padding Only': frozenset({'pad'}),
    'Chaff Injection Only': frozenset({'chaff'}),
    'Window Morphing Only': frozenset({'win'}),
    'Time + Padding': frozenset({'time', 'pad'}),
    'Time + Chaff': frozenset({'time', 'chaff'}),
    'Padding + Chaff': frozenset({'pad', 'chaff'}),
    'Complete Lie Cone': ALL_GENERATORS,
}
COST_REGIMES = {
    'A (Default / Balanced)': {'time': 1e-4, 'pad': 1e-2, 'chaff': 1.0, 'win': 1e-3},
    'B (Latency-Sensitive)': {'time': 10.0, 'pad': 1e-3, 'chaff': 0.1, 'win': 1e-3},
    'C (Unconstrained Delay)': {'time': 1e-5, 'pad': 10.0, 'chaff': 10.0, 'win': 1e-4},
    'D (Uniform / Flat)': {'time': 1.0, 'pad': 1.0, 'chaff': 1.0, 'win': 1.0},
}


def build_target():
    """Trains the benchmark XGBoost target and returns (model, schema, feature_stds)."""
    schema = get_cicids_schema()
    X, y, _ = generate_cicids_traffic(n_samples=12000, attack_ratio=0.25, random_state=42)
    X_train, y_train = X[:8000], y[:8000]
    model = NIDSEnsembleTrainer('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42,
    })
    model.train(X_train, y_train)
    return model, schema, np.std(X_train, axis=0) + 1e-6


def build_cohort(model) -> np.ndarray:
    """Balanced cohort of correctly detected malicious flows, N_PER_FAMILY per family."""
    cohorts = generate_attack_family_cohorts(n_per_family=N_PER_FAMILY, random_state=42)
    flows = np.vstack(list(cohorts.values()))
    return flows[model.predict_proba(flows)[:, 1] >= 0.5]


def evaluate(model, schema, stds, cohort: np.ndarray, *, step_budget: int = 20,
             enabled: FrozenSet[str] = ALL_GENERATORS, costs: Optional[Dict[str, float]] = None,
             epsilon_rel: float = 1e-4) -> Dict[str, object]:
    """Runs FFE over the cohort and aggregates success, feasibility, cost, queries and usage."""
    attack = FlowFieldEvasionAttack(model, schema, step_budget=step_budget, epsilon_rel=epsilon_rel,
                                    feature_stds=stds, enabled_generators=enabled, control_costs=costs)
    usage: Dict[str, int] = {}
    ok = valid = real = 0
    queries: List[int] = []
    costs_l1: List[float] = []
    delays_ms: List[float] = []
    start = time.time()
    for flow in cohort:
        res = attack.attack_sample(flow)
        ok += res.success
        valid += res.physically_valid
        real += res.success and res.physically_valid
        queries.append(res.queries_used)
        costs_l1.append(res.weighted_cost)
        delays_ms.append((res.x_adversarial[6] - flow[6]) / 1000.0)
        for name, count in attack.last_generator_usage.items():
            usage[name] = usage.get(name, 0) + count
    n = len(cohort)
    total_use = sum(usage.values())
    dominant = max(usage, key=usage.get) if usage else None
    return {
        'n': n,
        'asr': 100.0 * ok / n,
        'pfr': 100.0 * valid / n,
        'real_asr': 100.0 * real / n,
        'avg_queries': float(np.mean(queries)),
        'avg_cost_l1': float(np.mean(costs_l1)),
        'mean_delay_ms': float(np.mean(delays_ms)),
        'total_time_s': time.time() - start,
        'dominant_generator': dominant,
        'dominant_share': (100.0 * usage[dominant] / total_use) if dominant else 0.0,
    }


def run_ablations() -> Dict[str, object]:
    model, schema, stds = build_target()
    cohort = build_cohort(model)
    print(f'Ablation cohort: {len(cohort)} detected malicious flows')
    common = dict(model=model, schema=schema, stds=stds, cohort=cohort)
    results = {
        'cohort_size': int(len(cohort)),
        'budget': {str(m): evaluate(step_budget=m, **common) for m in STEP_BUDGETS},
        'generators': {k: evaluate(enabled=v, **common) for k, v in GENERATOR_SUBSETS.items()},
        'costs': {k: evaluate(costs=v, **common) for k, v in COST_REGIMES.items()},
        'epsilon': {str(e): evaluate(epsilon_rel=e, **common) for e in EPSILONS},
    }
    for section in ('budget', 'generators', 'costs', 'epsilon'):
        print(f'\n[{section}]')
        for key, r in results[section].items():
            print(f"  {key:28s} ASR={r['asr']:6.1f}% PFR={r['pfr']:6.1f}% Q={r['avg_queries']:6.1f} "
                  f"cost={r['avg_cost_l1']:.3f} delay={r['mean_delay_ms']:9.1f}ms "
                  f"dom={r['dominant_generator']}({r['dominant_share']:.0f}%) t={r['total_time_s']:.2f}s")
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'ablation_results.json')
    with open(out, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'\n[Results Saved] {os.path.abspath(out)}')
    return results


if __name__ == '__main__':
    run_ablations()
