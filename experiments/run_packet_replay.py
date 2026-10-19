"""
Packet-level replay of FFE adversarial flows (synthetic traces; see src/validation/packet_replay.py).

1. Round trip: extract(trace(x)) vs x for real flows of each corpus.
2. FFE (default settings, XGBoost, cohort of the main benchmark, first N flows) -> replay the
   actions implied by x_adv - x_orig at packet level, compare features, re-score with XGBoost.

Usage:
    python experiments/run_packet_replay.py --n 300 --roundtrip 20000
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

from common import load_arrays, detected_by_all  # noqa: E402  (imports xgboost/lightgbm first)
from src.datasets.schema import get_cicids_schema, verify_physical_feasibility  # noqa: E402
from src.models.tree_models import NIDSEnsembleTrainer  # noqa: E402
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack  # noqa: E402
from src.validation.packet_replay import (  # noqa: E402
    CHAFF_MODES, extract_features, relative_error, replay, trace_from_record,
)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
SEED = 42
FEATURES = ['dst_port', 'protocol', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts',
            'flow_duration', 'init_fwd_win_byts', 'flow_bytes_s', 'flow_pkts_s', 'fwd_pkt_len_mean',
            'fwd_header_len']
MODEL_CONFIGS = {
    'xgboost': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
                'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED},
    'lightgbm': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': SEED},
}
TRACE_SEEDS = (0, 1, 2)


def quantiles(v: np.ndarray) -> Dict[str, float]:
    return {'median': float(np.median(v)), 'p95': float(np.quantile(v, 0.95)),
            'p99': float(np.quantile(v, 0.99)), 'max': float(np.max(v))}


def delta_agreement(xo, xa, xr, mask) -> Dict[str, float]:
    """Per feature: among flows whose schema-level change is non-zero, share where the replayed
    change equals it within 1% (relative to the larger of the two changes)."""
    out = {}
    for j, f in enumerate(FEATURES):
        d_adv, d_rep = (xa - xo)[mask, j], (xr - xo)[mask, j]
        nz = np.abs(d_adv) > 1e-9
        if not nz.any():
            out[f] = None
            continue
        rel = np.abs(d_adv - d_rep)[nz] / np.maximum(np.maximum(np.abs(d_adv), np.abs(d_rep))[nz], 1e-12)
        out[f] = float((rel < 0.01).mean())
    return out


def roundtrip(X: np.ndarray, n: int, rng: np.random.RandomState) -> Dict:
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    errs = np.zeros((len(idx), 12))
    for k, i in enumerate(idx):
        x = X[i]
        errs[k] = relative_error(x, extract_features(trace_from_record(x, np.random.default_rng(int(i)))))
    worst = errs.max(axis=1)
    return {
        'n': int(len(idx)),
        'share_exact_1e-9': float((worst < 1e-9).mean()),
        'share_within_1pct': float((worst < 0.01).mean()),
        'share_within_5pct': float((worst < 0.05).mean()),
        'worst_feature_error': quantiles(worst),
        'per_feature_max_error': {f: float(errs[:, j].max()) for j, f in enumerate(FEATURES)},
        'per_feature_share_exact': {f: float((errs[:, j] < 1e-9).mean()) for j, f in enumerate(FEATURES)},
        'single_packet_flows': int((X[idx, 2] + X[idx, 3] == 1).sum()),
    }


def attack_cohort(name: str, n: int) -> Dict:
    data = load_arrays(name)
    X_tr, y_tr, _ = data['train']
    X_te, y_te, _ = data['test']
    schema = get_cicids_schema()
    stds = X_tr.std(axis=0) + 1e-6
    trained = {}
    for m_type, cfg in MODEL_CONFIGS.items():
        tr = NIDSEnsembleTrainer(m_type, cfg)
        tr.train(X_tr, y_tr)
        if m_type == 'xgboost':
            tr.model.get_booster().set_param({'device': 'cpu'})
        trained[m_type] = tr
    # cohort exactly as run_full_benchmark.py (n_fast=1000, detected by both models), first n flows
    rng = np.random.RandomState(SEED)
    mal = np.where(y_te == 1)[0]
    det = mal[detected_by_all(list(trained.values()), X_te[mal])]
    cohort = X_te[det[rng.choice(len(det), size=min(1000, len(det)), replace=False)]][:n]
    xgb = trained['xgboost']
    atk = FlowFieldEvasionAttack(xgb, schema, step_budget=20, feature_stds=stds)
    res = [atk.attack_sample(f) for f in cohort]
    return {'xgb': xgb, 'schema': schema, 'res': res, 'X_te': X_te, 'y_te': y_te}


def evaluate_replay(name: str, n: int, n_roundtrip: int) -> Dict:
    t0 = time.time()
    ctx = attack_cohort(name, n)
    xgb, schema, res = ctx['xgb'], ctx['schema'], ctx['res']
    xo = np.array([r.x_original for r in res])
    xa = np.array([r.x_adversarial for r in res])
    success = np.array([r.success for r in res])
    valid = np.array([r.physically_valid for r in res])
    real = success & valid
    score = lambda X: xgb.predict_proba(X)[:, 1]
    out = {'dataset': name, 'n': len(res), 'schema_asr': 100 * float(success.mean()),
           'schema_real_asr': 100 * float(real.mean()), 'n_success': int(success.sum()),
           'n_real': int(real.sum())}
    out['roundtrip_test_flows'] = roundtrip(ctx['X_te'], n_roundtrip, np.random.RandomState(SEED))
    out['roundtrip_attacked_original_flows'] = roundtrip(xo, len(xo), np.random.RandomState(SEED))
    out['schema_adv_model_evasive'] = float((score(xa) < 0.5).mean())

    active = {'time': xa[:, 6] > xo[:, 6], 'chaff': xa[:, 2] > xo[:, 2],
              'pad': xa[:, 4] - xo[:, 4] > 40.0 * np.maximum(0, xa[:, 2] - xo[:, 2]) + 0.5,
              'win': xa[:, 7] != xo[:, 7]}
    out['action_usage_share_of_success'] = {k: float(v[success].mean()) for k, v in active.items()}

    out['modes'] = {}
    for mode in CHAFF_MODES:
        per_seed = []
        for seed in TRACE_SEEDS:
            reps = [replay(xo[i], xa[i], mode, seed * 100003 + i) for i in range(len(res))]
            xr = np.array([r['features'] for r in reps])
            xb = np.array([r['baseline_features'] for r in reps])
            err = relative_error(xa, xr)
            worst = err.max(axis=1)
            evasive_after = score(xr) < 0.5
            feas_after = np.array([verify_physical_feasibility(xo[i], xr[i], schema) for i in range(len(res))])
            row = {
                'trace_seed': seed,
                'retention_of_schema_evasions_pct': 100 * float(evasive_after[success].mean()),
                'retention_of_real_evasions_pct': 100 * float(evasive_after[real].mean()) if real.any() else None,
                'replayed_asr_pct_of_all': 100 * float((evasive_after & success).sum() / len(res)),
                'replayed_real_asr_pct_of_all': 100 * float((evasive_after & real & feas_after).sum() / len(res)),
                'baseline_trace_still_detected_pct': 100 * float((score(xb) >= 0.5).mean()),
                'share_flows_all_features_within_1e-6': float((worst[success] < 1e-6).mean()),
                'share_flows_all_features_within_1pct': float((worst[success] < 0.01).mean()),
                'share_flows_all_features_within_5pct': float((worst[success] < 0.05).mean()),
                'worst_feature_error_over_success': quantiles(worst[success]),
                'per_feature_error_success': {f: quantiles(err[success, j]) for j, f in enumerate(FEATURES)},
                'per_feature_share_within_1pct': {f: float((err[success, j] < 0.01).mean())
                                                   for j, f in enumerate(FEATURES)},
                'flows_pad_unmet': int(sum(r['pad_unmet_bytes'] > 0 for r, s in zip(reps, success) if s)),
                'flows_header_unmet': int(sum(r['header_unmet_bytes'] > 0 for r, s in zip(reps, success) if s)),
                'flows_header_deficit': int(sum(r['plan'].header_deficit > 0 for r, s in zip(reps, success) if s)),
                'per_feature_delta_share_within_1pct': delta_agreement(xo, xa, xr, success),
                'flows_header_extra_needed': int(sum(r['plan'].header_extra > 0 for r, s in zip(reps, success) if s)),
                'feasible_after_replay_pct_of_success': 100 * float(feas_after[success].mean()),
            }
            # flips are where the schema evasion is lost
            flips = np.where(success & ~evasive_after)[0]
            row['n_flips'] = int(len(flips))
            row['flip_flow_indices'] = flips.tolist()
            per_seed.append(row)
        out['modes'][mode] = per_seed
    out['seconds'] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--roundtrip', type=int, default=20000)
    args = ap.parse_args()
    results: List[Dict] = []
    for name in args.datasets:
        r = evaluate_replay(name, args.n, args.roundtrip)
        results.append(r)
        m = r['modes']['schema'][0]
        print(f"{name}: schema ASR {r['schema_asr']:.1f}  retention(schema) {m['retention_of_schema_evasions_pct']:.1f}  "
              f"within1% {m['share_flows_all_features_within_1pct']:.3f}  rt-exact "
              f"{r['roundtrip_test_flows']['share_exact_1e-9']:.4f}  ({r['seconds']:.0f}s)")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'packet_replay.json')
    with open(path, 'w') as fh:
        json.dump({'config': {'n_flows': args.n, 'roundtrip_flows': args.roundtrip, 'trace_seeds': list(TRACE_SEEDS),
                              'chaff_modes': list(CHAFF_MODES), 'seed': SEED}, 'datasets': results}, fh, indent=1)
    print('saved', path)


if __name__ == '__main__':
    main()
