"""
Full-scale evasion benchmark.

For each dataset: load the *entire* deduplicated capture, split it 70/10/20
(stratified by label and attack family), train XGBoost (CUDA) and LightGBM on the
whole training split, evaluate on the whole test split, then attack a large
cohort of correctly detected malicious test flows with FFE and the baselines.
Per-flow outcomes are stored so that paired tests and confidence intervals can be
computed from the saved JSON.

Usage:
    python experiments/run_full_benchmark.py --datasets cse2018 cic2017 unsw \
        --n-fast 1000 --n-slow 200
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common import detected_by_all
from src.datasets.full_data import FEATURES, build_full_dataset, stratified_split
from src.datasets.schema import get_cicids_schema, project_network_invariants_batch
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.cost import operational_cost
from src.attacks.unconstrained import UnconstrainedFeatureSpaceAttack
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack
from src.attacks.modern_baselines import (
    DEVICE, CAPGDTransferAttack, LogStandardizer, NIDSGANAttack, train_surrogate,
)
from src.attacks.baselines import (
    ZOONIDSAttack, HopSkipJumpNIDSAttack, TrafficManipulatorPSOAttack, MOEVAAttack,
)

import torch  # noqa: E402  (must come after xgboost/lightgbm are loaded)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
SEED = 42

MODEL_CONFIGS = {
    'xgboost': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
                'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED},
    'lightgbm': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
                 'random_state': SEED},
}


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval (in %) for a binomial proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * max(0.0, centre - half), 100 * min(1.0, centre + half)]


def build_attacks(model, schema, stds, surrogate_bundle) -> Dict[str, tuple]:
    """name -> (attack, is_slow)."""
    surrogate, norm, generator = surrogate_bundle
    return {
        'Unconstrained Feature-Space': (UnconstrainedFeatureSpaceAttack(model, schema, max_iter=15), False),
        'ZOO-NIDS': (ZOONIDSAttack(model, schema, max_iter=50), False),
        'HopSkipJump-NIDS': (HopSkipJumpNIDSAttack(model, schema, max_iter=30), False),
        'CAA / MOEVA': (MOEVAAttack(model, schema, pop_size=25, n_generations=20), True),
        'Traffic Manipulator PSO': (TrafficManipulatorPSOAttack(model, schema, n_particles=15, max_iter=25), True),
        'CAPGD (surrogate transfer)': (CAPGDTransferAttack(model, schema, surrogate, norm), False),
        'NIDSGAN-style generator': (NIDSGANAttack(model, schema, surrogate, norm, generator), False),
        'Flow-Field Evasion (FFE)': (FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds), False),
        'FFE without TSBS': (FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds, enable_tsbs=False), False),
    }


def run_attack(attack, cohort: np.ndarray, schema, stds: np.ndarray) -> Dict:
    """Attack every flow and score all methods with the same two cost measures.

    ``op_cost`` is the paper's operational cost (delay, padding, chaff, window; see
    ``src.attacks.cost``), ``dl1`` the feature-normalised L1 distance mean_j |dx_j| / sigma_j and
    ``schema_cost`` the schema-weighted L1; all are computed here from the original and final
    flow, so every method is measured identically. Costs are those of the final candidate.
    """
    outcomes = []
    t0 = time.time()
    for flow in cohort:
        res = attack.attack_sample(flow)
        delta = np.abs(res.x_adversarial - res.x_original)
        outcomes.append({'success': bool(res.success), 'valid': bool(res.physically_valid),
                         'queries': int(res.queries_used),
                         'op_cost': operational_cost(res.x_original, res.x_adversarial),
                         'schema_cost': float(np.sum(schema.perturbation_costs * delta)),
                         'dl1': float(np.mean(delta / stds))})
    elapsed = time.time() - t0
    n = len(outcomes)
    real = sum(o['success'] and o['valid'] for o in outcomes)
    succ = sum(o['success'] for o in outcomes)
    valid = sum(o['valid'] for o in outcomes)
    real_o = [o for o in outcomes if o['success'] and o['valid']]
    return {
        'n': n, 'asr': 100 * succ / n, 'pfr': 100 * valid / n, 'real_asr': 100 * real / n,
        'real_asr_ci95': wilson(real, n),
        'avg_queries': float(np.mean([o['queries'] for o in outcomes])),
        'mean_op_cost_all': float(np.mean([o['op_cost'] for o in outcomes])),
        'median_op_cost_real': float(np.median([o['op_cost'] for o in real_o])) if real_o else None,
        'mean_dl1_all': float(np.mean([o['dl1'] for o in outcomes])),
        'median_dl1_real': float(np.median([o['dl1'] for o in real_o])) if real_o else None,
        'ms_per_flow': 1000 * elapsed / n, 'outcomes': outcomes,
    }


def pick_cohort(models, X_test, y_test, fam_test, n: int, rng) -> Dict:
    """Random malicious test flows that every given model detects (one cohort for all models)."""
    mal = np.where(y_test == 1)[0]
    detected = mal[detected_by_all(models, X_test[mal])]
    take = rng.choice(len(detected), size=min(n, len(detected)), replace=False)
    idx = detected[take]
    return {'idx': idx, 'X': X_test[idx], 'family': fam_test[idx].tolist(),
            'n_malicious_test': int(len(mal)), 'n_detected': int(len(detected))}


def run_dataset(name: str, n_fast: int, n_slow: int) -> Dict:
    schema = get_cicids_schema()
    df = build_full_dataset(name)
    splits = stratified_split(df, seed=SEED)
    data = {}
    for k, part in splits.items():
        data[k] = (project_network_invariants_batch(part[FEATURES].values.astype(float)),
                   part['label'].values.astype(int), part['attack_family'].values)
    sizes = {k: int(len(v[1])) for k, v in data.items()}
    print(f'\n=== {name}: {sizes} | attacks in test: {int(data["test"][1].sum()):,}')
    X_tr, y_tr, _ = data['train']
    X_te, y_te, fam_te = data['test']
    stds = X_tr.std(axis=0) + 1e-6

    out = {'dataset': name, 'dedup': df.attrs.get('dedup_stats'), 'split_sizes': sizes, 'models': {}}
    trained = {}
    for m_type, cfg in MODEL_CONFIGS.items():
        trainer = NIDSEnsembleTrainer(m_type, cfg)
        t0 = time.time()
        trainer.train(X_tr, y_tr)
        if m_type == 'xgboost':
            trainer.model.get_booster().set_param({'device': 'cpu'})  # one device for metrics and attacks
        metrics = trainer.evaluate(X_te, y_te)
        metrics['train_seconds'] = time.time() - t0
        out['models'][m_type] = {'clean': metrics}
        trained[m_type] = trainer
        print(f'  {m_type}: acc={metrics["accuracy"]:.4f} f1={metrics["f1"]:.4f} '
              f'auc={metrics["roc_auc"]:.4f} ({metrics["train_seconds"]:.1f}s)')

    norm = LogStandardizer.fit(X_tr)
    t0 = time.time()
    surrogate = train_surrogate(X_tr, y_tr, norm)
    mal_idx = np.random.RandomState(SEED).permutation(np.where(y_tr == 1)[0])[:500_000]
    generator = NIDSGANAttack.train_generator(X_tr[mal_idx], surrogate, norm, schema)
    out['surrogate_train_seconds'] = time.time() - t0
    probe = X_te[np.random.RandomState(SEED).choice(len(X_te), min(50_000, len(X_te)), replace=False)]
    with torch.no_grad():
        s_dec = (surrogate(torch.tensor(norm.forward(probe), dtype=torch.float32,
                                        device=DEVICE)).squeeze(1) > 0).cpu().numpy()
    for m_type, trainer in trained.items():
        t_dec = trainer.predict_proba(probe)[:, 1] >= 0.5
        out['models'][m_type]['surrogate_agreement'] = float((s_dec == t_dec).mean())
    cohort = pick_cohort(list(trained.values()), X_te, y_te, fam_te, n_fast, np.random.RandomState(SEED))
    for m_type, trainer in trained.items():
        out['models'][m_type]['cohort'] = {k: cohort[k] for k in ('n_malicious_test', 'n_detected', 'family')}
        attacks = build_attacks(trainer, schema, stds, (surrogate, norm, generator))
        # The multi-method table is produced for XGBoost; LightGBM runs FFE only.
        names = list(attacks) if m_type == 'xgboost' else ['Flow-Field Evasion (FFE)']
        out['models'][m_type]['attacks'] = {}
        for atk_name in names:
            atk, is_slow = attacks[atk_name]
            sub = cohort['X'][:n_slow] if is_slow else cohort['X']
            np.random.seed(SEED)  # stochastic baselines are reproducible
            res = run_attack(atk, sub, schema, stds)
            out['models'][m_type]['attacks'][atk_name] = res
            print(f'  [{m_type}] {atk_name:28s} N={res["n"]:4d} ASR={res["asr"]:5.1f} '
                  f'PFR={res["pfr"]:5.1f} RealASR={res["real_asr"]:5.1f} '
                  f'q={res["avg_queries"]:6.1f} {res["ms_per_flow"]:8.1f} ms/flow')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--n-fast', type=int, default=1000)
    ap.add_argument('--n-slow', type=int, default=200)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.datasets:
        result = run_dataset(name, args.n_fast, args.n_slow)
        path = os.path.join(RESULTS_DIR, f'full_{name}.json')
        with open(path, 'w') as fh:
            json.dump(result, fh, indent=1)
        print(f'  saved {path}')


if __name__ == '__main__':
    main()
