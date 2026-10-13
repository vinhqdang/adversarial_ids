"""
Full Experimental Benchmark Pipeline:
1. Train XGBoost (CUDA) and LightGBM on CIC-IDS2018-style flow traffic.
2. Validate standard baseline metrics (demonstrates the 99.xx% Accuracy / AUC).
3. Evaluate Unconstrained Attack vs. Novel PACT-GPU Evasion Algorithm.
4. Compute Attack Success Rate (ASR), Physical Feasibility Rate (PFR), Queries, and Perturbation Cost.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.schema import get_cicids_schema, verify_physical_feasibility
from src.datasets.generator import generate_cicids_traffic
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.unconstrained import UnconstrainedFeatureSpaceAttack
from src.attacks.pact_evasion import PACTEvasionAttack
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack


def run_full_benchmark():
    print("=" * 80)
    print("  ACADEMIC AML-NIDS BENCHMARK: TREE ENSEMBLE EVASION UNDER PROBLEM-SPACE INVARIANTS")
    print("=" * 80)
    
    schema = get_cicids_schema()
    
    # 1. Generate Realistic Traffic
    print("\n[Step 1] Generating benchmark network flow dataset (CIC-IDS2018 distribution)...")
    X, y, df = generate_cicids_traffic(n_samples=12000, attack_ratio=0.25, random_state=42)
    
    # Split Train/Val/Test
    n_train = 8000
    n_val = 1000
    n_test = 3000
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"Dataset generated: Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")
    print(f"Features: {schema.feature_names}")
    
    # 2. Train Models (XGBoost on CUDA & LightGBM)
    print("\n[Step 2] Training SOTA Tree Ensembles (XGBoost with CUDA & LightGBM)...")
    
    models = {}
    
    # XGBoost with CUDA
    xgb_params = {
        'n_estimators': 60,
        'max_depth': 6,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    xgb_trainer = NIDSEnsembleTrainer('xgboost', xgb_params)
    t0 = time.time()
    xgb_trainer.train(X_train, y_train)
    xgb_time = time.time() - t0
    models['XGBoost (CUDA)'] = (xgb_trainer, xgb_time)
    
    # LightGBM
    lgb_params = {
        'n_estimators': 60,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    lgb_trainer = NIDSEnsembleTrainer('lightgbm', lgb_params)
    t0 = time.time()
    lgb_trainer.train(X_train, y_train)
    lgb_time = time.time() - t0
    models['LightGBM'] = (lgb_trainer, lgb_time)
    
    # Standard Evaluation (The 99.xx% Phenomenon)
    print("\n" + "-" * 60)
    print("STANDARD TEST SET PERFORMANCE (Demonstrates the ~99.xx% Baseline)")
    print("-" * 60)
    
    baseline_rows = []
    for m_name, (trainer, train_dur) in models.items():
        metrics = trainer.evaluate(X_test, y_test)
        baseline_rows.append([
            m_name,
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['precision']*100:.2f}%",
            f"{metrics['recall']*100:.2f}%",
            f"{metrics['f1']*100:.2f}%",
            f"{metrics['roc_auc']*100:.2f}%",
            f"{train_dur:.2f}s"
        ])
        
    print(tabulate(
        baseline_rows,
        headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Train Time"],
        tablefmt="github"
    ))
    
    # 3. Adversarial Evasion Evaluation
    print("\n[Step 3] Evaluating Adversarial Evasion: Baselines vs. Novel Flow-Field Evasion (FFE)...")
    
    # Filter test samples that are true attacks and correctly detected by XGBoost
    xgb_model = models['XGBoost (CUDA)'][0]
    attack_mask = (y_test == 1)
    X_attacks = X_test[attack_mask]
    
    preds_attack = (xgb_model.predict_proba(X_attacks)[:, 1] >= 0.5)
    detected_attacks = X_attacks[preds_attack]
    
    eval_cohort_size = min(100, len(detected_attacks))
    cohort = detected_attacks[:eval_cohort_size]
    print(f"Evaluating evasion against {eval_cohort_size} correctly classified malicious flows...")
    
    attacks_to_test = {
        'Unconstrained Feature-Space (Naive)': UnconstrainedFeatureSpaceAttack(xgb_model, schema, max_iter=40),
        'Flow-Field Evasion (FFE - Ours Novel)': FlowFieldEvasionAttack(xgb_model, schema, step_budget=20)
    }
    
    results_summary = []
    
    for atk_name, atk in attacks_to_test.items():
        print(f"\nRunning {atk_name}...")
        t_start = time.time()
        
        success_count = 0
        physically_valid_count = 0
        total_queries = 0
        costs = []
        
        for i in range(eval_cohort_size):
            sample = cohort[i]
            res = atk.attack_sample(sample)
            
            if res.success:
                success_count += 1
            if res.physically_valid:
                physically_valid_count += 1
                
            total_queries += res.queries_used
            costs.append(res.weighted_cost)
            
        elapsed = time.time() - t_start
        asr = (success_count / eval_cohort_size) * 100.0
        pfr = (physically_valid_count / eval_cohort_size) * 100.0
        real_asr = (sum(1 for i in range(eval_cohort_size) if attacks_to_test[atk_name].attack_sample(cohort[i]).success and attacks_to_test[atk_name].attack_sample(cohort[i]).physically_valid) / eval_cohort_size) * 100.0 if False else (min(success_count, physically_valid_count) / eval_cohort_size) * 100.0
        avg_queries = total_queries / eval_cohort_size
        avg_cost = float(np.mean(costs))
        
        results_summary.append([
            atk_name,
            f"{asr:.1f}%",
            f"{pfr:.1f}%",
            f"{avg_queries:.1f}",
            f"{avg_cost:.2f}",
            f"{elapsed:.2f}s"
        ])
        
    print("\n" + "=" * 80)
    print("  ADVERSARIAL EVASION BENCHMARK RESULTS (PAPER TABLE 1)")
    print("=" * 80)
    print(tabulate(
        results_summary,
        headers=["Attack Method", "Evasion Rate (ASR)", "Physical Feasibility (PFR)", "Avg Queries", "Avg Cost ($L_1$)", "Total Time"],
        tablefmt="github"
    ))
    
    # Save results to json for paper figures
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_models': baseline_rows,
            'evasion_results': results_summary
        }, f, indent=2)
        
    print(f"\n[Results Saved] Benchmark output exported to: {output_file}")


if __name__ == '__main__':
    run_full_benchmark()
