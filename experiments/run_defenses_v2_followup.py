"""Follow-up checks for defense v2 (results/defense_v2_followup_<dataset>.json and .md).

(1) G35: monotone gate that also hides the backward columns 3 and 5 (server responses assumed unchanged
    by the attacker are then not trusted either); V2 rebuilt with G35 (validation-only selection, same rule).
(2) Capacity control: unconstrained 300-tree depth-8 model with the gate's hidden columns (no monotone constraint).
(3) Whole-cohort evasion = 1 - flagged-after-attack / cohort, benign FPR on the test split and FPR on benign
    flows in the top decile of forward packets. Rows for the models of run_defenses_v2.py are retrained
    (models are not persisted) and their attack outcomes are read from results/defense_v2_<dataset>.json.
(5) Optional FFE-v2 on the hardened models (--ffev2), first ``--ffev2-n`` cohort flows.

Usage: python experiments/run_defenses_v2_followup.py <dataset> [--libraries ...] [--ffev2]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.defenses.models import RobustTreeModel  # noqa: E402  (xgboost/lightgbm before torch)
from common import load_arrays  # noqa: E402
import run_defenses_v2 as rd  # noqa: E402
from src.defenses import recipe  # noqa: E402
from src.defenses.attackers import make_context, register_attack  # noqa: E402

G35_HIDDEN = (3, 5, 7, 8, 9, 10)
ATTACKS = ('ffe_m20', 'capgd', 'whitebox')
RESULTS = rd.RESULTS


def whole_cohort_evasion(codes: np.ndarray) -> float:
    """100 * (1 - flagged after attack / cohort); flows missed at clean time count as evaded."""
    real = int(np.sum(codes == rd.OUTCOME['real']))
    detected = int(np.sum(codes >= 0))
    return round(100.0 * (1.0 - (detected - real) / len(codes)), 1)


def fpr_stats(model, X_te: np.ndarray, y_te: np.ndarray) -> dict:
    ben = X_te[y_te == 0]
    flagged = model.predict_proba(ben)[:, 1] >= 0.5
    big = ben[:, 2] >= np.quantile(ben[:, 2], 0.9)
    return {'benign_fpr': round(100.0 * flagged.mean(), 3),
            'benign_fpr_top_decile_fwd_pkts': round(100.0 * flagged[big].mean(), 3),
            'benign_fpr_rest': round(100.0 * flagged[~big].mean(), 3)}


def model_row(model, X_te, y_te, cohort) -> dict:
    return {'clean': rd.clean_metrics(model, X_te, y_te), **fpr_stats(model, X_te, y_te),
            'cohort_flagged_clean': round(100.0 * np.mean(model.predict_proba(cohort)[:, 1] >= 0.5), 1),
            'build_seconds': round(float(getattr(model, 'build_seconds', 0.0)), 1)}


def attack_block(model, ctx, cohort, attacks, n_jobs: int) -> dict:
    out = {}
    for a in attacks:
        codes = rd.attack_outcomes(a, model, ctx, cohort, n_jobs)
        out[a] = {**rd.summarize(codes), 'whole_cohort_evasion': whole_cohort_evasion(codes)}
        print(f'      {a:9s} real ASR {out[a]["real_asr"]:5.1f} whole-cohort {out[a]["whole_cohort_evasion"]:5.1f}', flush=True)
    return out


def saved_attacks(saved: dict, name: str) -> dict:
    res = {}
    for a in ATTACKS:
        s = saved['models'][name]['attacks'][a]
        codes = np.array(s['outcomes'])
        res[a] = {k: v for k, v in s.items() if k != 'outcomes'}
        res[a]['whole_cohort_evasion'] = whole_cohort_evasion(codes)
    return res


def run_library(lib: str, data, ctx, saved: dict, args) -> dict:
    X_tr, y_tr, _ = data['train']
    X_va, y_va, _ = data['val']
    X_te, y_te, _ = data['test']
    sv = saved['models'][lib]
    print(f'\n=== {lib}', flush=True)
    base = rd.timed(lambda: RobustTreeModel(lib).train(X_tr, y_tr))
    val_cohort = rd.cohort_of(base, X_va, y_va, args.val_cohort, rd.SEED)
    cohort = rd.cohort_of(base, X_te, y_te, args.cohort, rd.SEED)
    ref = sv['models']['Undefended']
    same = (int(np.sum(base.predict_proba(cohort)[:, 1] >= 0.5)) == ref['cohort_detected']
            and abs(100 * base.evaluate(X_te, y_te)['accuracy'] - ref['clean']['accuracy']) < 0.01)
    print(f'  retrained undefended matches saved run: {same}', flush=True)
    gate = rd.timed(recipe.train_gate, lib, X_tr, y_tr)
    g35 = rd.timed(lambda: RobustTreeModel(lib, hidden=G35_HIDDEN, monotone=recipe.GATE_MONOTONE,
                                           params=recipe.GATE_CAPACITY[lib]).train(X_tr, y_tr))
    unc = rd.timed(lambda: RobustTreeModel(lib, hidden=recipe.GATE_HIDDEN,
                                           params=recipe.GATE_CAPACITY[lib]).train(X_tr, y_tr))
    mps = rd.timed(lambda: RobustTreeModel(lib, hidden=recipe.MPS_HIDDEN).train(X_tr, y_tr))
    lgat = rd.timed(recipe.train_lgat, lib, X_tr, y_tr, base, ctx, (), args.n_jobs)
    sel, sel_r = sv['validation']['selected'], sv['validation']['selected_relaxed']
    frees = {'undefended': base}
    for pick in (sel, sel_r):
        if pick['free'] == 'orbit_light' and 'orbit_light' not in frees:
            frees['orbit_light'] = rd.timed(recipe.train_orbit, lib, X_tr, y_tr, 42, 0.25)
        if pick['free'] == 'adversarial' and 'adversarial' not in frees:
            frees['adversarial'] = rd.timed(recipe.train_adversarial, lib, X_tr, y_tr, base, ctx, args.n_jobs)
    if 'orbit_light' not in frees:
        frees['orbit_light'] = rd.timed(recipe.train_orbit, lib, X_tr, y_tr, 42, 0.25)
    v2, v2r = rd.build_v2(frees, gate, sel), rd.build_v2(frees, gate, sel_r)
    tuning = rd.tune_on_validation({k: frees[k] for k in ('undefended', 'orbit_light')}, g35, base, ctx,
                                   X_va, y_va, val_cohort, args.n_jobs)
    sel35 = tuning['selected']
    print('  G35 SELECTED (validation):', {k: sel35[k] for k in ('free', 'tau', 'clean')}, flush=True)
    v2g35 = rd.build_v2(frees, g35, sel35)
    rows = {'Undefended': base, 'MPS': mps, 'LGAT': lgat, 'V2': v2, 'V2-R': v2r, 'Gate G': gate, 'Gate G35': g35,
            'V2 with G35': v2g35, 'Unconstrained 300x8 (gate columns hidden)': unc}
    out = {'retrained_matches_saved': bool(same), 'cohort_size': int(len(cohort)),
           'v2_selection': {'strict': {k: sel[k] for k in ('free', 'tau')}, 'relaxed': {k: sel_r[k] for k in ('free', 'tau')},
                            'g35': {k: sel35[k] for k in ('free', 'tau')}, 'g35_parity_met': tuning['parity_met']},
           'g35_validation_table': tuning['table'], 'models': {}}
    for name, model in rows.items():
        print(f'    -- {name}', flush=True)
        entry = model_row(model, X_te, y_te, cohort)
        if name in ('Undefended', 'MPS', 'LGAT', 'V2', 'V2-R'):
            entry['attacks'] = saved_attacks(sv, name)
        elif name in ('Gate G35', 'V2 with G35', 'Unconstrained 300x8 (gate columns hidden)'):
            entry['attacks'] = attack_block(model, ctx, cohort, ATTACKS, args.n_jobs)
        else:
            entry['attacks'] = saved_attacks(sv, 'Gate (certified)')
        out['models'][name] = entry
    if args.ffev2:
        out['ffev2'] = run_ffev2(lib, {'Undefended': base, 'V2': v2, 'V2 with G35': v2g35, 'Gate G35': g35},
                                 X_tr, y_tr, ctx, cohort[:args.ffev2_n], args.n_jobs)
    return out


def run_ffev2(lib, models, X_tr, y_tr, ctx, cohort, n_jobs: int) -> dict:
    from tune_ffev2 import make_config
    from src.attacks.ffe_v2 import FFEv2Attack, benign_window_candidates
    cfg = json.load(open(os.path.join(RESULTS, 'ffev2_config.json')))['snap']
    windows = list(benign_window_candidates(X_tr, y_tr, 12))
    register_attack('ffe_v2', lambda model, c: FFEv2Attack(model, c.schema, make_config(cfg, windows), c.stds))
    out = {'cohort_used': int(len(cohort)), 'config': cfg}
    for name, model in models.items():
        t0 = time.time()
        codes = rd.attack_outcomes('ffe_v2', model, ctx, cohort, n_jobs)
        out[name] = {**rd.summarize(codes), 'whole_cohort_evasion': whole_cohort_evasion(codes)}
        print(f'    FFE-v2 {name}: real ASR {out[name]["real_asr"]} whole {out[name]["whole_cohort_evasion"]} '
              f'({time.time() - t0:.0f}s)', flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--libraries', nargs='+', default=['xgboost', 'lightgbm', 'catboost'])
    ap.add_argument('--cohort', type=int, default=1000)
    ap.add_argument('--val-cohort', type=int, default=300)
    ap.add_argument('--n-jobs', type=int, default=12)
    ap.add_argument('--ffev2', action='store_true')
    ap.add_argument('--ffev2-n', type=int, default=300)
    args = ap.parse_args()
    saved = json.load(open(os.path.join(RESULTS, f'defense_v2_{args.dataset}.json')))
    data = load_arrays(args.dataset)
    ctx = make_context(data['train'][0])
    part_dir = os.path.join(RESULTS, 'defense_v2_followup_parts')
    os.makedirs(part_dir, exist_ok=True)
    for lib in args.libraries:
        res = run_library(lib, data, ctx, saved, args)
        with open(os.path.join(part_dir, f'{args.dataset}_{lib}.json'), 'w') as fh:
            json.dump(res, fh)
        merged = {}
        for fn in sorted(os.listdir(part_dir)):
            if fn.startswith(args.dataset + '_'):
                merged[fn[len(args.dataset) + 1:-5]] = json.load(open(os.path.join(part_dir, fn)))
        with open(os.path.join(RESULTS, f'defense_v2_followup_{args.dataset}.json'), 'w') as fh:
            json.dump({'dataset': args.dataset, 'models': merged}, fh)


if __name__ == '__main__':
    main()
