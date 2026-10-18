"""
Multi-seed robustness of the headline comparison.

For every seed the whole pipeline is redone: stratified split, XGBoost training, cohort of detected
malicious test flows, surrogate and generator training. FFE, CAPGD (surrogate transfer) and the
NIDSGAN-style generator are then attacked on the same cohort. This captures retraining variance,
which the single-seed benchmark cannot.

Usage: python experiments/run_seeds.py --datasets unsw cic2017 cse2018 --seeds 1 2 3 4 5 --n 300
Output: results/seeds_<dataset>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common import XGB_CFG, detected_by_all, load_arrays
from run_full_benchmark import run_attack
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack
from src.attacks.modern_baselines import (DEVICE, CAPGDTransferAttack, LogStandardizer,
                                          NIDSGANAttack, train_surrogate)
from src.datasets.schema import get_cicids_schema
from src.models.tree_models import NIDSEnsembleTrainer

import torch  # noqa: E402  (must come after xgboost/lightgbm are loaded)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def run_seed(name: str, seed: int, n: int) -> dict:
    schema = get_cicids_schema()
    data = load_arrays(name, seed=seed)
    X_tr, y_tr, _ = data['train']
    X_te, y_te, _ = data['test']
    stds = X_tr.std(axis=0) + 1e-6

    cfg = dict(XGB_CFG, random_state=seed)
    model = NIDSEnsembleTrainer('xgboost', cfg)
    model.train(X_tr, y_tr)
    model.model.get_booster().set_param({'device': 'cpu'})
    clean = model.evaluate(X_te, y_te)

    norm = LogStandardizer.fit(X_tr)
    surrogate = train_surrogate(X_tr, y_tr, norm, seed=seed)
    mal = np.random.RandomState(seed).permutation(np.where(y_tr == 1)[0])[:500_000]
    generator = NIDSGANAttack.train_generator(X_tr[mal], surrogate, norm, schema, seed=seed)

    probe = X_te[np.random.RandomState(seed).choice(len(X_te), min(50_000, len(X_te)), replace=False)]
    with torch.no_grad():
        s_dec = (surrogate(torch.tensor(norm.forward(probe), dtype=torch.float32,
                                        device=DEVICE)).squeeze(1) > 0).cpu().numpy()
    agreement = float((s_dec == (model.predict_proba(probe)[:, 1] >= 0.5)).mean())

    rng = np.random.RandomState(seed)
    mal_te = np.where(y_te == 1)[0]
    det = mal_te[detected_by_all(model, X_te[mal_te])]
    cohort = X_te[rng.choice(det, size=min(n, len(det)), replace=False)]

    attacks = {
        'FFE': FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds),
        'CAPGD': CAPGDTransferAttack(model, schema, surrogate, norm),
        'NIDSGAN-style': NIDSGANAttack(model, schema, surrogate, norm, generator),
    }
    out = {'seed': seed, 'accuracy': clean['accuracy'], 'f1': clean['f1'],
           'surrogate_agreement': agreement, 'n_cohort': int(len(cohort)), 'attacks': {}}
    for atk_name, atk in attacks.items():
        np.random.seed(seed)
        res = run_attack(atk, cohort, schema, stds)
        out['attacks'][atk_name] = {k: res[k] for k in ('real_asr', 'asr', 'avg_queries', 'mean_op_cost_all')}
        print(f'  [{name} seed {seed}] {atk_name:14s} real_asr={res["real_asr"]:5.1f} '
              f'q={res["avg_queries"]:5.1f}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['unsw', 'cic2017', 'cse2018'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.datasets:
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            runs.append(run_seed(name, seed, args.n))
            print(f'  seed {seed} done in {time.time() - t0:.0f}s')
        with open(os.path.join(RESULTS_DIR, f'seeds_{name}.json'), 'w') as fh:
            json.dump({'dataset': name, 'runs': runs}, fh, indent=1)


if __name__ == '__main__':
    main()
