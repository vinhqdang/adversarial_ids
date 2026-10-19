"""Defense v2: certified monotone gate + hardened free classifier, three corpora, several libraries.

Usage: python experiments/run_defenses_v2.py <dataset> [--libraries xgboost lightgbm catboost]
       [--cohort 1000] [--val-cohort 300] [--plugin package.module] [--n-jobs 12]

Protocol. Design and tuning use the VALIDATION split only (clean parity constraint, validation
Real ASR of FFE M=20 and of the white-box attacker on a validation cohort). The selected configuration is
written to the result file BEFORE the test cohort is touched; the test cohort (malicious test flows the
undefended model of the same library detects, random sample) is then attacked once per model. Attacks:
FFE M=5, FFE M=20, CAPGD (surrogate trained on the defended model's own soft outputs) and the projected
white-box threshold-crossing attacker that reads the defended model's trees. Further attacks (e.g. FFE-v2)
plug in through ``src.defenses.attackers.register_attack`` and ``--plugin``.
Writes results/defense_v2_<dataset>.json (models.<library>.<defense>) with per-flow outcomes.
"""

import argparse
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scipy.stats import binomtest  # noqa: E402

from src.defenses.models import RobustTreeModel  # noqa: E402  (xgboost/lightgbm before torch)
from common import load_arrays  # noqa: E402
from src.defenses import recipe  # noqa: E402
from src.defenses.attackers import AttackContext, make_context, run_attack, wilson  # noqa: E402

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
TEST_ATTACKS = ('ffe_m5', 'ffe_m20', 'capgd', 'whitebox')
VAL_ATTACKS = ('ffe_m20', 'whitebox')
TAUS = (None, 0.5, 0.8, 0.95, 0.99)
PARITY = 0.5                        # strict tier: clean accuracy / F1 / recall may drop by at most 0.5 point
RELAXED = 3.0                       # robustness-prioritised tier (accuracy and F1 within 3 points)
OUTCOME = {'undetected': -1, 'fail': 0, 'real': 1, 'infeasible': 2}


def cohort_of(model, X: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    mal = np.where(y == 1)[0]
    det = mal[model.predict_proba(X[mal])[:, 1] >= 0.5]
    rng = np.random.RandomState(seed)
    return X[rng.choice(det, min(n, len(det)), replace=False)]


def attack_outcomes(name: str, model, ctx: AttackContext, cohort: np.ndarray, n_jobs: int) -> np.ndarray:
    """Per-cohort-flow outcome code (see OUTCOME); only flows the model detects are attacked."""
    det = model.predict_proba(cohort)[:, 1] >= 0.5
    codes = np.full(len(cohort), OUTCOME['undetected'], dtype=int)
    res = run_attack(name, model, ctx, cohort[det], n_jobs)
    codes[np.where(det)[0]] = [OUTCOME['fail'] if not s else OUTCOME['real'] if v else OUTCOME['infeasible']
                               for s, v, _, _ in res]
    return codes


def summarize(codes: np.ndarray) -> Dict[str, object]:
    n_det = int(np.sum(codes >= 0))
    real = int(np.sum(codes == OUTCOME['real']))
    return {'n_detected': n_det, 'real_asr': round(100.0 * real / max(1, n_det), 1),
            'real_asr_ci': wilson(real, n_det), 'asr': round(100.0 * np.sum(codes >= 1) / max(1, n_det), 1),
            'retained_detection': round(100.0 * (n_det - real) / len(codes), 1),
            'outcomes': codes.tolist()}


def clean_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    c = model.evaluate(X, y)
    return {k: round(100 * c[k], 2) for k in ('accuracy', 'f1', 'recall', 'precision', 'roc_auc')}


def within_parity(c: Dict[str, float], ref: Dict[str, float], tol: float = PARITY,
                  check_recall: bool = True) -> bool:
    ok = c['accuracy'] >= ref['accuracy'] - tol and c['f1'] >= ref['f1'] - tol
    return ok and (c['recall'] >= ref['recall'] - tol or not check_recall)


def tune_on_validation(frees: Dict[str, object], gate, base, ctx, X_val, y_val, val_cohort,
                       n_jobs: int) -> Dict[str, object]:
    """Grid over (free classifier, gate threshold). Two tiers, both chosen on validation only:
    ``selected`` = lowest validation ASR among candidates within PARITY of the undefended clean metrics;
    ``selected_relaxed`` = lowest validation ASR among candidates within RELAXED points (accuracy, F1)."""
    ref = clean_metrics(base, X_val, y_val)
    table: List[dict] = []
    for fname, free in frees.items():
        for tau in TAUS:
            if fname == 'undefended' and tau is None:
                continue
            model = recipe.fuse(free, gate, tau)
            row = {'free': fname, 'tau': tau, 'clean': clean_metrics(model, X_val, y_val)}
            row['parity'] = bool(within_parity(row['clean'], ref))
            row['relaxed'] = bool(within_parity(row['clean'], ref, RELAXED, check_recall=False))
            if row['relaxed']:
                asr = [summarize(attack_outcomes(a, model, ctx, val_cohort, n_jobs))['real_asr']
                       for a in VAL_ATTACKS]
                row.update(val_asr=dict(zip(VAL_ATTACKS, asr)), val_score=float(np.mean(asr)))
            table.append(row)
            print('  val', fname, tau, row['clean'], row.get('val_asr'), flush=True)
    strict = [r for r in table if r['parity']]
    relaxed = [r for r in table if r['relaxed']]
    pick = lambda rows: min(rows, key=lambda r: (r['val_score'], -r['clean']['f1']))
    return {'table': table, 'reference_clean': ref, 'parity_met': bool(strict),
            'selected': pick(strict) if strict else pick(relaxed),
            'selected_relaxed': pick(relaxed)}


def mcnemar(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar on Real-ASR flags over flows both models detect (a, b: outcome codes)."""
    both = (a >= 0) & (b >= 0)
    fa, fb = a[both] == OUTCOME['real'], b[both] == OUTCOME['real']
    only_a, only_b = int(np.sum(fa & ~fb)), int(np.sum(~fa & fb))
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {'n_paired': int(both.sum()), 'only_first': only_a, 'only_second': only_b, 'p': p}


def evaluate_models(models: Dict[str, object], X_te, y_te, cohort, ctx, attacks, n_jobs: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, model in models.items():
        entry = {'clean': clean_metrics(model, X_te, y_te),
                 'build_seconds': round(float(getattr(model, 'build_seconds', model.train_seconds)), 1),
                 'cohort_detected': int(np.sum(model.predict_proba(cohort)[:, 1] >= 0.5)), 'attacks': {}}
        for a in attacks:
            t0 = time.time()
            entry['attacks'][a] = summarize(attack_outcomes(a, model, ctx, cohort, n_jobs))
            print(f'    {name:14s} {a:9s} real ASR {entry["attacks"][a]["real_asr"]:5.1f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        out[name] = entry
    return out


def timed(fn, *args, **kwargs):
    t0 = time.time()
    model = fn(*args, **kwargs)
    model.build_seconds = time.time() - t0
    return model


def build_v2(frees: Dict[str, object], gate, sel: dict):
    free = frees[sel['free']]
    model = recipe.fuse(free, gate, sel['tau'])
    if model is free:                       # no gate: a fresh handle so the build cost is the free model's
        return free
    model.build_seconds = gate.build_seconds + free.build_seconds
    return model


def run_library(library: str, data, ctx, args) -> Dict[str, object]:
    X_tr, y_tr, _ = data['train']
    X_va, y_va, _ = data['val']
    X_te, y_te, _ = data['test']
    print(f'\n=== {library}', flush=True)
    base = timed(lambda: RobustTreeModel(library).train(X_tr, y_tr))
    val_cohort = cohort_of(base, X_va, y_va, args.val_cohort, SEED)
    cohort = cohort_of(base, X_te, y_te, args.cohort, SEED)
    print(f'  cohorts: val {len(val_cohort)}, test {len(cohort)}', flush=True)
    t0 = time.time()
    gate = timed(recipe.train_gate, library, X_tr, y_tr)
    orbit = timed(recipe.train_orbit, library, X_tr, y_tr)
    adv = timed(recipe.train_adversarial, library, X_tr, y_tr, base, ctx, args.n_jobs)
    print(f'  v2 components built in {time.time() - t0:.0f}s', flush=True)
    orbit_light = timed(recipe.train_orbit, library, X_tr, y_tr, 42, 0.25)
    frees = {'undefended': base, 'orbit': orbit, 'orbit_light': orbit_light, 'adversarial': adv}
    tuning = tune_on_validation(frees, gate, base, ctx, X_va, y_va, val_cohort, args.n_jobs)
    sel = tuning['selected']
    print('  SELECTED (frozen before test):', {k: sel[k] for k in ('free', 'tau', 'clean')}, flush=True)
    v2 = build_v2(frees, gate, sel)
    sel_r = tuning['selected_relaxed']
    v2r = build_v2(frees, gate, sel_r)
    print('  SELECTED relaxed:', {k: sel_r[k] for k in ('free', 'tau', 'clean')}, flush=True)
    mps = timed(lambda: RobustTreeModel(library, hidden=recipe.MPS_HIDDEN).train(X_tr, y_tr))
    lgat = timed(recipe.train_lgat, library, X_tr, y_tr, base, ctx, (), args.n_jobs)
    mps_lgat = timed(recipe.train_lgat, library, X_tr, y_tr, base, ctx, recipe.MPS_HIDDEN, args.n_jobs)
    models = {'Undefended': base, 'MPS': mps, 'LGAT': lgat, 'MPS+LGAT': mps_lgat,
              'Orbit': orbit, 'AdvTrain': adv, 'Gate (certified)': gate, 'V2': v2, 'V2-R': v2r}
    res = evaluate_models(models, X_te, y_te, cohort, ctx, TEST_ATTACKS, args.n_jobs)
    key = {a: {n: np.array(res[n]['attacks'][a]['outcomes']) for n in res} for a in TEST_ATTACKS}
    tests = {a: {f'V2 vs {n}': mcnemar(key[a]['V2'], key[a][n])
                 for n in ('Undefended', 'MPS', 'LGAT', 'MPS+LGAT', 'AdvTrain', 'V2-R')} for a in TEST_ATTACKS}
    return {'validation': tuning, 'cohort_size': int(len(cohort)), 'val_cohort_size': int(len(val_cohort)),
            'models': res, 'mcnemar_v2': tests}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--libraries', nargs='+', default=['xgboost', 'lightgbm', 'catboost'])
    ap.add_argument('--cohort', type=int, default=1000)
    ap.add_argument('--val-cohort', type=int, default=300)
    ap.add_argument('--n-jobs', type=int, default=12)
    ap.add_argument('--extra-attacks', nargs='*', default=[], help='registered attack names to add (e.g. ffe_v2)')
    ap.add_argument('--plugin', default=None, help='module that calls register_attack(...)')
    args = ap.parse_args()
    global TEST_ATTACKS
    if args.plugin:
        importlib.import_module(args.plugin)
    if args.extra_attacks:
        TEST_ATTACKS = TEST_ATTACKS + tuple(a for a in args.extra_attacks if a not in TEST_ATTACKS)
    data = load_arrays(args.dataset)
    ctx = make_context(data['train'][0])
    part_dir = os.path.join(RESULTS, 'defense_v2_parts')
    os.makedirs(part_dir, exist_ok=True)
    for lib in args.libraries:
        part = {'dataset': args.dataset, 'library': lib, **run_library(lib, data, ctx, args)}
        with open(os.path.join(part_dir, f'{args.dataset}_{lib}.json'), 'w') as fh:
            json.dump(part, fh)
    merge(args.dataset)


def merge(dataset: str) -> None:
    """Combine the per-library parts into results/defense_v2_<dataset>.json (models.<library>)."""
    part_dir = os.path.join(RESULTS, 'defense_v2_parts')
    models = {}
    for fn in sorted(os.listdir(part_dir)):
        if fn.startswith(dataset + '_'):
            with open(os.path.join(part_dir, fn)) as fh:
                p = json.load(fh)
            models[p.pop('library')] = {k: v for k, v in p.items() if k != 'dataset'}
    with open(os.path.join(RESULTS, f'defense_v2_{dataset}.json'), 'w') as fh:
        json.dump({'dataset': dataset, 'attacks': list(TEST_ATTACKS), 'models': models}, fh)


if __name__ == '__main__':
    main()
