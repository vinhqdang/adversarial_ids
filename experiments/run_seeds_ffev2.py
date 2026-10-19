"""
Multi-seed evaluation of the FROZEN FFE-v2 (results/ffev2_config.json, 'score' variant).

Per seed (as experiments/run_seeds.py): stratified split with that seed, retrained XGBoost (and, for
--tree-seeds, LightGBM and CatBoost), cohort of up to ``n`` malicious test flows detected by XGBoost,
surrogate for CAPGD. LightGBM / CatBoost are attacked on the part of that cohort they detect
themselves. Attacks: FFE-v2 (score-only), FFE-v1, white-box control (project in loop), CAPGD
(surrogate transfer, black-box against the target). Nothing is tuned here.

Usage: python experiments/run_seeds_ffev2.py --datasets cse2018 cic2017 unsw --seeds 1 2 3 4 5 --tree-seeds 1 2 3
Output: results/seeds_ffev2_<dataset>.json (resumable) and results/seeds_ffev2_report.md (--report-only to rebuild).
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import xgboost  # noqa: F401,E402  (before torch)
import lightgbm  # noqa: F401,E402
from common import XGB_CFG, detected_by_all, load_arrays  # noqa: E402
from ffev2_setup import SingleThread  # noqa: E402
from run_full_benchmark import run_attack  # noqa: E402
from seeds_stats import summarise_outcomes, real_flags  # noqa: E402
from tune_ffev2 import make_config  # noqa: E402
from src.attacks.ffe_v2 import FFEv2Attack, benign_window_candidates  # noqa: E402
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack  # noqa: E402
from src.attacks.modern_baselines import CAPGDTransferAttack, LogStandardizer, train_surrogate  # noqa: E402
from src.attacks.whitebox_tree import RawMarginModel, TreeDump, WhiteBoxLeafTupleAttack  # noqa: E402
from src.datasets.schema import get_cicids_schema  # noqa: E402
from src.models.tree_models import NIDSEnsembleTrainer  # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CONFIG = os.path.join(RESULTS, 'ffev2_config.json')
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ffev2_cache', 'seeds'))
ATTACKS = ['FFE-v2', 'FFE-v1', 'Control', 'CAPGD']
CHUNK = 8
_MODELS = {}


def tree_config(model_type: str, seed: int) -> dict:
    if model_type == 'xgboost':
        return dict(XGB_CFG, random_state=seed)
    if model_type == 'lightgbm':
        return {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': seed}
    return {'iterations': 60, 'depth': 6, 'learning_rate': 0.1, 'random_seed': seed, 'verbose': False}


def _loaded(path: str):
    if path not in _MODELS:
        _MODELS.clear()
        trained = pickle.load(open(path, 'rb'))
        trained['xgboost'].model.get_booster().set_param({'device': 'cpu'})
        _MODELS[path] = {k: SingleThread(v) for k, v in trained.items()}
    return _MODELS[path]


def build_attack(kind: str, model, mt: str, schema, stds, windows, spec):
    if kind == 'FFE-v2':
        return FFEv2Attack(model, schema, make_config(spec, windows), stds)
    if kind == 'FFE-v1':
        return FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds)
    dump = TreeDump(model.trainer.model) if mt == 'xgboost' else RawMarginModel(model.trainer)
    return WhiteBoxLeafTupleAttack(model, schema, project_in_loop=True, tree_dump=dump)


def chunk_task(args):
    path, mt, kind, X, stds, windows, spec, seed = args
    schema = get_cicids_schema()
    attack = build_attack(kind, _loaded(path)[mt], mt, schema, stds, windows, spec)
    np.random.seed(seed)
    return run_attack(attack, X, schema, stds)['outcomes']


def run_pool(ex, path, jobs, stds, windows, spec, seed):
    """jobs: {(mt, kind): X}; returns {(mt, kind): outcomes} with flow order preserved."""
    tasks, keys = [], []
    for (mt, kind), X in jobs.items():
        for i in range(0, len(X), CHUNK):
            tasks.append((path, mt, kind, X[i:i + CHUNK], stds, windows, spec, seed))
            keys.append((mt, kind))
    out = {k: [] for k in jobs}
    for k, res in zip(keys, ex.map(chunk_task, tasks)):
        out[k].extend(res)
    return out


def run_seed(ex, name: str, seed: int, n: int, model_types, spec) -> dict:
    schema = get_cicids_schema()
    data = load_arrays(name, seed=seed)
    X_tr, y_tr, _ = data['train']
    X_te, y_te, _ = data['test']
    stds = X_tr.std(axis=0) + 1e-6
    trained = {}
    for mt in model_types:
        t = NIDSEnsembleTrainer(mt, tree_config(mt, seed))
        t.train(X_tr, y_tr)
        if mt == 'xgboost':
            t.model.get_booster().set_param({'device': 'cpu'})
        trained[mt] = t
    mal_te = np.where(y_te == 1)[0]
    det = mal_te[detected_by_all(trained['xgboost'], X_te[mal_te])]
    cohort = X_te[np.random.RandomState(seed).choice(det, size=min(n, len(det)), replace=False)]
    cohorts = {mt: cohort[trained[mt].predict_proba(cohort)[:, 1] >= 0.5] for mt in model_types}
    windows = list(benign_window_candidates(X_tr, y_tr, 12))
    path = os.path.join(CACHE, f'{name}_{seed}.pkl')
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(trained, open(path, 'wb'))

    norm = LogStandardizer.fit(X_tr)
    surrogate = train_surrogate(X_tr, y_tr, norm, seed=seed)
    jobs = {(mt, k): cohorts[mt] for mt in model_types for k in ('FFE-v2', 'FFE-v1', 'Control')}
    res = run_pool(ex, path, jobs, stds, windows, spec, seed)
    for mt in model_types:        # CAPGD needs the GPU surrogate: main process, queries the target
        t0 = time.time()
        atk = CAPGDTransferAttack(trained[mt], schema, surrogate, norm)
        np.random.seed(seed)
        res[(mt, 'CAPGD')] = run_attack(atk, cohorts[mt], schema, stds)['outcomes']
        print(f'    CAPGD {mt} {time.time() - t0:.0f}s', flush=True)
    os.remove(path)
    out = {'seed': seed, 'n_cohort': int(len(cohort)), 'n_detected_pool': int(len(det)), 'targets': {}}
    for mt in model_types:
        out['targets'][mt] = {'clean': trained[mt].evaluate(X_te, y_te), 'attacks': {}}
        for k in ATTACKS:
            o = res[(mt, k)]
            s = {**summarise_outcomes(o), 'real': real_flags(o).astype(int).tolist()}
            out['targets'][mt]['attacks'][k] = s
            print(f'  [{name} s{seed}] {mt:8s} {k:7s} n={s["n"]} real_asr={s["real_asr"]:5.1f} q={s["avg_queries"]:6.1f}', flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--tree-seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()
    if not args.report_only:
        spec = json.load(open(CONFIG))['score']
        with ProcessPoolExecutor(args.workers) as ex:
            for name in args.datasets:
                fp = os.path.join(RESULTS, f'seeds_ffev2_{name}.json')
                done = json.load(open(fp)) if os.path.exists(fp) else {'dataset': name, 'config': spec, 'runs': []}
                have = {r['seed'] for r in done['runs']}
                for seed in args.seeds:
                    if seed in have:
                        continue
                    t0 = time.time()
                    mts = ['xgboost', 'lightgbm', 'catboost'] if seed in args.tree_seeds else ['xgboost']
                    done['runs'].append(run_seed(ex, name, seed, args.n, mts, spec))
                    done['runs'].sort(key=lambda r: r['seed'])
                    json.dump(done, open(fp, 'w'), indent=1)
                    print(f'  {name} seed {seed} done in {time.time() - t0:.0f}s', flush=True)
    from seeds_report import write_report
    write_report(RESULTS, args.datasets)


if __name__ == '__main__':
    main()
