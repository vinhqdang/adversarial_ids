"""
Cross-Architecture Adversarial Transferability Experiment (Paper Section 8, Ablation 5).

Evaluates the black-box transferability of adversarial flows crafted via FFE across:
- XGBoost (CUDA)
- LightGBM
- CatBoost
- Random Forest

Results are saved to results/transferability_results.json and rendered in Table 7 and Figure 4(d).
"""

import json
import os
import sys
import time
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.schema import get_cicids_schema
from common import detected_cohort, load_arrays
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack


MODEL_CONFIGS = {
    'XGBoost (CUDA)': ('xgboost', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
    }),
    'LightGBM': ('lightgbm', {
        'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
        'random_state': 42, 'verbosity': -1
    }),
    'CatBoost': ('catboost', {
        'iterations': 60, 'depth': 6, 'learning_rate': 0.1,
        'random_seed': 42, 'verbose': False
    }),
    'Random Forest': ('random_forest', {
        'n_estimators': 100, 'max_depth': 12, 'random_state': 42
    })
}


def run_transferability_experiment(dataset: str = 'cse2018', n_per_family: int = 50) -> Dict[str, object]:
    schema = get_cicids_schema()
    print(f"Loading full {dataset} dataset...")
    data = load_arrays(dataset)
    X_train, y_train, _ = data['train']
    X_test, y_test, fam_test = data['test']
    stds = np.std(X_train, axis=0) + 1e-6

    # 1. Train all 4 models
    models: Dict[str, NIDSEnsembleTrainer] = {}
    clean_metrics: Dict[str, Dict[str, float]] = {}
    for name, (m_type, params) in MODEL_CONFIGS.items():
        print(f"Training {name}...")
        t0 = time.time()
        trainer = NIDSEnsembleTrainer(m_type, params)
        trainer.train(X_train, y_train)
        elapsed = time.time() - t0
        metrics = trainer.evaluate(X_test, y_test)
        metrics['train_time_s'] = elapsed
        clean_metrics[name] = metrics
        models[name] = trainer
        print(f"  {name}: Acc={metrics['accuracy']*100:.2f}%, AUC={metrics['roc_auc']*100:.2f}%, Time={elapsed:.2f}s")

    models['XGBoost (CUDA)'].model.get_booster().set_param({'device': 'cpu'})

    # 2. Build test cohort (detected by at least XGBoost)
    # Flows detected by the primary source model (XGBoost), balanced over families
    cohort, _ = detected_cohort(models['XGBoost (CUDA)'], X_test, y_test, fam_test, n_per_family)
    print(f"\nCohort size for transferability: {len(cohort)} detected flows")

    model_names = list(MODEL_CONFIGS.keys())
    matrix = {src: {} for src in model_names}
    detailed_results = {}

    for src_name in model_names:
        src_model = models[src_name]
        print(f"\nCrafting FFE adversarial flows against source: {src_name}...")
        attack = FlowFieldEvasionAttack(src_model, schema, step_budget=20, feature_stds=stds)
        adv_flows = []
        src_successes = 0
        
        t0 = time.time()
        for flow in cohort:
            res = attack.attack_sample(flow)
            adv_flows.append(res.x_adversarial)
            if res.success:
                src_successes += 1
        elapsed = time.time() - t0
        adv_matrix = np.array(adv_flows)
        
        detailed_results[src_name] = {
            'craft_time_s': elapsed,
            'source_asr': 100.0 * src_successes / len(cohort)
        }

        # Evaluate against all target models
        for tgt_name in model_names:
            tgt_model = models[tgt_name]
            tgt_probs = tgt_model.predict_proba(adv_matrix)[:, 1]
            tgt_evaded = int(np.sum(tgt_probs < 0.5))
            asr = 100.0 * tgt_evaded / len(cohort)
            matrix[src_name][tgt_name] = round(asr, 1)
            print(f"  -> Target {tgt_name:18s}: ASR = {asr:5.1f}% ({tgt_evaded}/{len(cohort)})")

    output_data = {
        'dataset': dataset,
        'model_names': model_names,
        'cohort_size': len(cohort),
        'clean_metrics': clean_metrics,
        'transferability_matrix': matrix,
        'detailed_results': detailed_results
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'transferability_results.json')
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Saved Transferability Results] {os.path.abspath(out_path)}")
    return output_data


if __name__ == '__main__':
    run_transferability_experiment()
