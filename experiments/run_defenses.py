"""
Defensive countermeasures against Flow-Field Evasion (paper Section 9).

* MPS (malleability-penalised splitting, alpha -> infinity limit): the tree ensemble never sees the
  duration and the two rate features, which an attacker moves most cheaply with temporal dilation.
  The attacker still works on, and perturbs, the full 12-feature flow.
* LGAT (Lie-guided adversarial training): the training set is augmented with FFE flows (M = 5)
  generated against the undefended model for a random sample of malicious training flows.
* MPS + LGAT.

Each model is attacked with FFE at M = 5 and M = 20 and with the surrogate-transfer CAPGD baseline,
on the malicious test flows that the undefended model detects (balanced by attack family) and
that the evaluated model itself still detects. Real ASR counts evasions that are schema-feasible.
Results go to results/defense_results.json.
"""

import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.tree_models import NIDSEnsembleTrainer  # xgboost/lightgbm before torch
from common import SEED, XGB_CFG, ColumnMaskedModel, detected_by_all, detected_cohort, load_arrays
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack, PhysicalFlowGenerators
from src.attacks.modern_baselines import CAPGDTransferAttack, LogStandardizer, train_surrogate
from src.datasets.schema import get_cicids_schema

DATASET = 'cse2018'          # overridden by the command line
N_PER_FAMILY = 50
MAX_AUGMENT = 5000            # Lie-guided samples generated per LGAT model (cost cap)
MASKED = (6, 8, 9)            # duration and the two rates: the features temporal dilation changes
TRAIN_BUDGET = 5              # FFE budget used to craft the augmentation
EVAL_BUDGETS = (5, 20)


def compute_feature_malleability_index(X: np.ndarray, schema) -> np.ndarray:
    """Feature Physical Malleability Index M_mal(j) = E[max_k delta_s_k |v_k(x)_j| / sigma_j]."""
    stds = np.std(X, axis=0) + 1e-6
    delta_s = [1e4, 100.0, 1.0, 1000.0]
    idx = np.random.RandomState(SEED).choice(len(X), min(len(X), 1000), replace=False)
    m_mal = np.zeros(X.shape[1])
    for i in idx:
        x = X[i]
        gens = [PhysicalFlowGenerators.v_time(x), PhysicalFlowGenerators.v_pad(x),
                PhysicalFlowGenerators.v_chaff(x), PhysicalFlowGenerators.v_win(x, 64240.0)]
        m_mal += np.max([delta_s[k] * np.abs(gens[k]) / stds for k in range(4)], axis=0)
    m_mal /= len(idx)
    m_mal[list(schema.immutable_indices)] = 0.0
    return m_mal


def attack_rates(attack, cohort: np.ndarray) -> Dict[str, float]:
    ok = real = 0
    for flow in cohort:
        res = attack.attack_sample(flow)
        ok += res.success
        real += res.success and res.physically_valid
    n = max(1, len(cohort))
    return {'n': int(len(cohort)), 'asr': round(100.0 * ok / n, 1), 'real_asr': round(100.0 * real / n, 1)}


def evaluate_model(model, X_test, y_test, cohort, schema, stds, capgd_parts) -> Dict[str, object]:
    clean = model.evaluate(X_test, y_test)
    own = cohort[detected_by_all(model, cohort)]       # flows this model still detects
    out = {'clean_accuracy': round(100 * clean['accuracy'], 2), 'clean_f1': round(100 * clean['f1'], 2),
           'clean_attack_recall': round(100 * clean['recall'], 2),
           'clean_roc_auc': round(100 * clean['roc_auc'], 2), 'cohort_detected': int(len(own))}
    for m in EVAL_BUDGETS:
        out[f'ffe_m{m}'] = attack_rates(FlowFieldEvasionAttack(model, schema, step_budget=m, feature_stds=stds), own)
    surrogate, norm = capgd_parts
    out['capgd'] = attack_rates(CAPGDTransferAttack(model, schema, surrogate, norm), own)
    return out


def train_xgb(X, y) -> NIDSEnsembleTrainer:
    t = NIDSEnsembleTrainer('xgboost', XGB_CFG)
    t.train(X, y)
    t.model.get_booster().set_param({'device': 'cpu'})   # same device for metrics and attacks
    return t


def run_defenses_experiment(dataset: str = DATASET) -> Dict[str, object]:
    global DATASET
    DATASET = dataset
    schema = get_cicids_schema()
    print(f'Loading full {DATASET} dataset...')
    data = load_arrays(DATASET)
    X_train, y_train, _ = data['train']
    X_test, y_test, fam_test = data['test']
    stds = np.std(X_train, axis=0) + 1e-6

    print('\n[1/4] Undefended XGBoost')
    base = train_xgb(X_train, y_train)
    cohort, _ = detected_cohort(base, X_test, y_test, fam_test, N_PER_FAMILY)
    print(f'  cohort: {len(cohort)} detected malicious flows')
    norm = LogStandardizer.fit(X_train)
    capgd_parts = (train_surrogate(X_train, y_train, norm), norm)
    m_mal = compute_feature_malleability_index(X_train, schema)
    print('  M_mal:', np.round(m_mal, 3))

    results = {'Undefended': evaluate_model(base, X_test, y_test, cohort, schema, stds, capgd_parts)}
    print('  ', results['Undefended'])

    print('\n[2/4] MPS (duration and rate features hidden from the model)')
    mps = ColumnMaskedModel(NIDSEnsembleTrainer('xgboost', XGB_CFG), MASKED)
    mps.train(X_train, y_train)
    mps.trainer.model.get_booster().set_param({'device': 'cpu'})
    results['MPS'] = evaluate_model(mps, X_test, y_test, cohort, schema, stds, capgd_parts)
    print('  ', results['MPS'])

    print('\n[3/4] LGAT (Lie-guided adversarial training)')
    mal_idx = np.where(y_train == 1)[0]
    n_aug = min(int(0.5 * len(mal_idx)), MAX_AUGMENT)
    chosen = np.random.RandomState(SEED).choice(mal_idx, n_aug, replace=False)
    crafter = FlowFieldEvasionAttack(base, schema, step_budget=TRAIN_BUDGET, feature_stds=stds)
    print(f'  crafting {n_aug} Lie-guided training flows...')
    aug = np.array([crafter.attack_sample(X_train[i]).x_adversarial for i in chosen])
    X_aug, y_aug = np.vstack([X_train, aug]), np.concatenate([y_train, np.ones(n_aug, dtype=int)])
    lgat = train_xgb(X_aug, y_aug)
    results['LGAT'] = evaluate_model(lgat, X_test, y_test, cohort, schema, stds, capgd_parts)
    print('  ', results['LGAT'])

    print('\n[4/4] MPS + LGAT')
    dual = ColumnMaskedModel(NIDSEnsembleTrainer('xgboost', XGB_CFG), MASKED)
    dual.train(X_aug, y_aug)
    dual.trainer.model.get_booster().set_param({'device': 'cpu'})
    results['MPS + LGAT'] = evaluate_model(dual, X_test, y_test, cohort, schema, stds, capgd_parts)
    print('  ', results['MPS + LGAT'])

    out = {'dataset': DATASET, 'cohort_size': int(len(cohort)), 'lgat_augment_samples': int(n_aug),
           'masked_features': list(MASKED), 'malleability_index': [float(v) for v in m_mal],
           'attack_prevalence_test': float(y_test.mean()), 'models': results}
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'defense_results.json' if DATASET == 'cse2018' else f'defense_{DATASET}.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\n[Saved] {os.path.abspath(out_path)}')
    return out


if __name__ == '__main__':
    import sys
    run_defenses_experiment(sys.argv[1] if len(sys.argv) > 1 else DATASET)
