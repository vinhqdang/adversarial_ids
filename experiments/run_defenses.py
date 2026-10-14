"""
Reproducible Defensive Countermeasures & Model Hardening (Paper Section 9).

Implements:
1. Malleability-Penalized Tree Splitting (MPS): penalizes splits on highly malleable flow rate/duration features.
2. Lie-Guided Adversarial Training (LGAT): augments training data with physical Lie flow trajectories.
3. Dual-Hardened Model (MPS + LGAT).

Evaluates clean test accuracy, ROC-AUC, robust recall (1 - ASR), and FFE ASR.
Results are saved to results/defense_results.json.
"""

import json
import os
import sys
import time
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.schema import get_cicids_schema, project_network_invariants
from src.datasets.generator import generate_cicids_traffic, generate_attack_family_cohorts
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack, PhysicalFlowGenerators


def compute_feature_malleability_index(X: np.ndarray, schema) -> np.ndarray:
    """
    Computes Feature Physical Malleability Index M_mal(j) across training data (Eq. 26).
    M_mal(j) = E [ max_k ( delta_s_k * |v_k(x)_j| / sigma_j ) ]
    """
    stds = np.std(X, axis=0) + 1e-6
    delta_s = [1e4, 100.0, 1.0, 1000.0]  # characteristic single-step perturbation scales
    
    n_samples = min(len(X), 1000)
    indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
    
    m_mal = np.zeros(12)
    for idx in indices:
        x = X[idx]
        v1 = PhysicalFlowGenerators.v_time(x)
        v2 = PhysicalFlowGenerators.v_pad(x)
        v3 = PhysicalFlowGenerators.v_chaff(x)
        v4 = PhysicalFlowGenerators.v_win(x, 64240.0)
        
        generators = [v1, v2, v3, v4]
        variations = np.zeros((4, 12))
        for k in range(4):
            variations[k] = (delta_s[k] * np.abs(generators[k])) / stds
        
        m_mal += np.max(variations, axis=0)
        
    m_mal /= n_samples
    # Immutables have strictly zero malleability
    m_mal[schema.immutable_indices] = 0.0
    return m_mal


def evaluate_defense_model(trainer, X_test, y_test, cohort, schema, stds, step_budget: int = 5) -> Dict[str, float]:
    """Evaluates clean classification metrics and adversarial robustness under FFE."""
    clean_eval = trainer.evaluate(X_test, y_test)
    
    # Run FFE on the cohort with evaluation budget M_train = 5
    attack = FlowFieldEvasionAttack(trainer, schema, step_budget=step_budget, feature_stds=stds)
    successes = 0
    for flow in cohort:
        res = attack.attack_sample(flow)
        if res.success:
            successes += 1
            
    n_cohort = len(cohort)
    ffe_asr = 100.0 * successes / n_cohort
    robust_recall = 100.0 - ffe_asr
    
    return {
        'clean_accuracy': round(clean_eval['accuracy'] * 100.0, 2),
        'clean_roc_auc': round(clean_eval['roc_auc'] * 100.0, 2),
        'robust_recall': round(robust_recall, 1),
        'ffe_asr': round(ffe_asr, 1)
    }


def run_defenses_experiment() -> Dict[str, object]:
    schema = get_cicids_schema()
    print("Generating training dataset (12,000 flows)...")
    X, y, _ = generate_cicids_traffic(n_samples=12000, attack_ratio=0.25, random_state=42)
    X_train, y_train = X[:8000], y[:8000]
    X_test, y_test = X[8000:], y[8000:]
    stds = np.std(X_train, axis=0) + 1e-6

    # 1. Baseline XGBoost (Undefended)
    print("\n[1/4] Training Baseline XGBoost (Undefended)...")
    base_trainer = NIDSEnsembleTrainer('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
    })
    base_trainer.train(X_train, y_train)

    # Build cohort of 100 flows detected by baseline
    cohorts = generate_attack_family_cohorts(n_per_family=20, random_state=42)
    candidate_flows = np.vstack(list(cohorts.values()))
    base_probs = base_trainer.predict_proba(candidate_flows)[:, 1]
    cohort = candidate_flows[base_probs >= 0.5]
    print(f"Cohort size: {len(cohort)} malicious flows")

    base_results = evaluate_defense_model(base_trainer, X_test, y_test, cohort, schema, stds, step_budget=5)
    print("  Baseline:", base_results)

    # 2. MPS-Hardened XGBoost (Malleability Penalty)
    print("\n[2/4] Training MPS-Hardened XGBoost (Malleability Penalty)...")
    m_mal = compute_feature_malleability_index(X_train, schema)
    print("  Computed M_mal:", np.round(m_mal, 3))
    
    # Penalize splits on duration (6) and rates (8, 9) by zeroing malleable split signals
    scale_vector = np.ones(12)
    scale_vector[[6, 8, 9]] = 0.0
    X_train_mps = X_train * scale_vector
    X_test_mps = X_test * scale_vector
    cohort_mps = cohort * scale_vector
    stds_mps = stds * scale_vector

    mps_trainer = NIDSEnsembleTrainer('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
    })
    mps_trainer.train(X_train_mps, y_train)
    mps_results = evaluate_defense_model(mps_trainer, X_test_mps, y_test, cohort_mps, schema, stds_mps, step_budget=5)
    print("  MPS-Hardened:", mps_results)

    # 3. LGAT-Hardened XGBoost (Lie-Guided Adversarial Training)
    print("\n[3/4] Training LGAT-Hardened XGBoost (Lie-Guided AT)...")
    mal_train_indices = np.where(y_train == 1)[0]
    n_augment = int(0.5 * len(mal_train_indices))
    selected_indices = np.random.RandomState(42).choice(mal_train_indices, n_augment, replace=False)
    
    attack_fast = FlowFieldEvasionAttack(base_trainer, schema, step_budget=5, feature_stds=stds)
    aug_flows = []
    print(f"  Generating {n_augment} Lie-guided physical adversarial training samples...")
    for idx in selected_indices:
        res = attack_fast.attack_sample(X_train[idx])
        aug_flows.append(res.x_adversarial)
        
    X_train_lgat = np.vstack([X_train, np.array(aug_flows)])
    y_train_lgat = np.concatenate([y_train, np.ones(n_augment)])
    
    lgat_trainer = NIDSEnsembleTrainer('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
    })
    lgat_trainer.train(X_train_lgat, y_train_lgat)
    lgat_results = evaluate_defense_model(lgat_trainer, X_test, y_test, cohort, schema, stds, step_budget=5)
    print("  LGAT-Hardened:", lgat_results)

    # 4. Dual-Hardened (MPS + LGAT)
    print("\n[4/4] Training Dual-Hardened XGBoost (MPS + LGAT)...")
    X_train_dual = X_train_lgat * scale_vector
    dual_trainer = NIDSEnsembleTrainer('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
    })
    dual_trainer.train(X_train_dual, y_train_lgat)
    dual_results = evaluate_defense_model(dual_trainer, X_test_mps, y_test, cohort_mps, schema, stds_mps, step_budget=5)
    print("  Dual-Hardened:", dual_results)

    results = {
        'cohort_size': len(cohort),
        'models': {
            'Baseline XGBoost (Undefended)': base_results,
            'MPS-Hardened XGBoost': mps_results,
            'LGAT-Hardened XGBoost': lgat_results,
            'Dual-Hardened (MPS + LGAT)': dual_results
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'defense_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Defense Results] {os.path.abspath(out_path)}")
    return results


if __name__ == '__main__':
    run_defenses_experiment()
