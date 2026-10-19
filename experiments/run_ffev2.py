"""Final TEST evaluation of FFE-v2 (configuration frozen by experiments/tune_ffev2.py on the validation split).

Re-creates the models and the 1000-flow cohort of ``run_full_benchmark.py`` (checked against the saved
``results/full_<dataset>.json``) and attacks XGBoost and LightGBM on the full cohort, CatBoost on the
part of it that CatBoost itself detects. Writes ``results/ffev2_<dataset>.json`` in the per-attack schema
of ``results/full_<dataset>.json`` (``models.<model>.attacks.<name>``), with these entries:

* FFE-v2 (score-only) and FFE-v2 + snapping (this work, run now)
* FFE-v1 (copied from results/full_<dataset>.json for XGBoost/LightGBM, run now for CatBoost)
* White-box control, project-in-loop (copied from results/whitebox_<dataset>.json for XGBoost, run now
  for LightGBM and CatBoost through ``RawMarginModel``)
* CAPGD (XGBoost only, copied from results/full_<dataset>.json)

and exact McNemar tests of FFE-v2 against each of them on the same flows.

Usage: python experiments/run_ffev2.py --datasets cse2018 [--config results/ffev2_config.json]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ffev2_setup import cohorts_for, get_models, get_cicids_schema  # noqa: E402
import run_full_benchmark as rfb  # noqa: E402  (imports xgboost/lightgbm before torch)
from tune_ffev2 import make_config  # noqa: E402
from src.attacks.ffe_v2 import FFEv2Attack, benign_window_candidates  # noqa: E402
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack  # noqa: E402
from src.attacks.whitebox_tree import RawMarginModel, TreeDump, WhiteBoxLeafTupleAttack  # noqa: E402

RESULTS = rfb.RESULTS_DIR
V1, CAPGD = 'Flow-Field Evasion (FFE)', 'CAPGD (surrogate transfer)'
V2, V2S = 'FFE-v2 (score-only)', 'FFE-v2 + snapping'
CONTROL = 'White-box control (project in loop)'
CONFIG_PATH = os.path.join(RESULTS, 'ffev2_config.json')


def real_flags(a: dict) -> np.ndarray:
    return np.array([o['success'] and o['valid'] for o in a['outcomes']])


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {'v2_only': only_a, 'other_only': only_b, 'p': p}


def run_tracked(attack, X, schema, stds, extra):
    """rfb.run_attack plus per-flow ``extra(attack)`` fields (generator path, ledger, internal evals)."""
    log = []
    orig = attack.attack_sample

    def tracked(x):
        r = orig(x)
        log.append(extra(attack))
        return r
    attack.attack_sample = tracked
    np.random.seed(rfb.SEED)
    res = rfb.run_attack(attack, X, schema, stds)
    for o, e in zip(res['outcomes'], log):
        o.update(e)
    return res


def v2_extra(a):
    led = a.last_ledger
    return {'path': list(a.last_path), 'delay_us': led.delay_us, 'n_chaff': led.n_chaff,
            'n_segment': led.n_segment, 'pad_bytes': led.pad_bytes,
            'header_option_bytes': led.header_option_bytes, 'window': led.window}


def subset(res: dict, pos: np.ndarray) -> dict:
    """Restrict a copied attack result (full 1000-flow outcomes) to cohort positions ``pos``."""
    outcomes = [res['outcomes'][i] for i in pos]
    real = [o for o in outcomes if o['success'] and o['valid']]
    n = len(outcomes)
    return {**{k: v for k, v in res.items() if k not in ('outcomes',)}, 'n': n,
            'asr': 100 * sum(o['success'] for o in outcomes) / n,
            'pfr': 100 * sum(o['valid'] for o in outcomes) / n,
            'real_asr': 100 * len(real) / n,
            'avg_queries': float(np.mean([o['queries'] for o in outcomes])),
            'median_op_cost_real': float(np.median([o['op_cost'] for o in real])) if real else None,
            'outcomes': outcomes}


def run_dataset(name: str, cfgs: dict, n: int = 1000) -> dict:
    schema = get_cicids_schema()
    data, trained = get_models(name)
    ref = json.load(open(os.path.join(RESULTS, f'full_{name}.json')))
    wb_ref = json.load(open(os.path.join(RESULTS, f'whitebox_{name}.json')))
    X_tr, y_tr, _ = data['train']
    stds = X_tr.std(axis=0) + 1e-6
    coh = cohorts_for(trained, *data['test'], n)
    main_family = coh['xgboost']['family'].tolist()
    ref_family = ref['models']['xgboost']['cohort']['family']
    if main_family != ref_family:
        raise RuntimeError('cohort differs from results/full_*.json; flows would not be paired')
    windows = list(benign_window_candidates(X_tr, y_tr, 12))
    out = {'dataset': name, 'config': cfgs, 'window_pool': windows, 'cohort_matches_benchmark': True,
           'split_sizes': ref['split_sizes'], 'models': {}}
    for mt in ('xgboost', 'lightgbm', 'catboost'):
        model = trained[mt]
        X = coh[mt]['X']
        where = {int(i): k for k, i in enumerate(coh['xgboost']['idx'])}
        pos = np.array([where[int(i)] for i in coh[mt]['idx']])   # positions inside the 1000-flow cohort
        clean = model.evaluate(*data['test'][:2])
        entry = {'clean': clean, 'cohort': {'n_detected_cohort': int(len(X)), 'family': coh[mt]['family'].tolist(),
                                            'positions_in_main_cohort': pos.tolist()}, 'attacks': {}}
        atk = entry['attacks']
        for label, spec in (('score', V2), ('snap', V2S)):
            attack = FFEv2Attack(model, schema, make_config(cfgs[label], windows), stds)
            atk[spec] = run_tracked(attack, X, schema, stds, v2_extra)
        if mt == 'catboost':
            atk[V1] = rfb.run_attack(FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds), X, schema, stds)
        else:
            atk[V1] = subset(ref['models'][mt]['attacks'][V1], pos)
        if mt == 'xgboost':
            atk[CONTROL] = subset(wb_ref['models']['xgboost']['attacks']
                                  ['Leaf-tuple-style white-box (project in loop)'], pos)
            atk[CAPGD] = subset(ref['models']['xgboost']['attacks'][CAPGD], pos)
        else:
            margin = RawMarginModel(model)
            control = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=True, tree_dump=margin)
            atk[CONTROL] = run_tracked(control, X, schema, stds, lambda a: {'internal_evals': int(a.internal_evals)})
        v2 = real_flags(atk[V2])
        entry['mcnemar'] = {}
        for other in (V2S, V1, CONTROL, CAPGD):
            if other in atk:
                entry['mcnemar'][f'{V2} vs {other}'] = mcnemar(v2, real_flags(atk[other]))
        out['models'][mt] = entry
        for k, r in atk.items():
            print(f'  [{name}/{mt}] {k:36s} N={r["n"]:4d} Real ASR={r["real_asr"]:5.1f} q={r["avg_queries"]:6.1f} '
                  f'opcost med={r["median_op_cost_real"]}', flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--config', default=CONFIG_PATH)
    args = ap.parse_args()
    cfgs = json.load(open(args.config))
    for name in args.datasets:
        res = run_dataset(name, cfgs)
        with open(os.path.join(RESULTS, f'ffev2_{name}.json'), 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
