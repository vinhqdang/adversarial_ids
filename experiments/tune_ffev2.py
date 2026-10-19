"""Validation-split tuning of FFE-v2 (no test flow is ever touched here).

Every configuration is run on the 1000-flow cohort drawn from the VALIDATION split with the benchmark's
cohort procedure, against XGBoost, LightGBM and CatBoost on the three corpora. Results are appended to
``results/ffev2_tuning.json`` (one record per configuration and target).

Usage: python experiments/tune_ffev2.py --stage base|ablate|...  (stages are defined in STAGES)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ffev2_setup import cohorts_for, get_models, load_trained, get_cicids_schema  # noqa: E402
from src.attacks.cost import operational_cost  # noqa: E402
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack  # noqa: E402
from src.attacks.whitebox_tree import RawMarginModel, TreeDump, WhiteBoxLeafTupleAttack  # noqa: E402
from src.attacks.ffe_v2 import FFEv2Attack, FFEv2Config, benign_window_candidates, ALL_GENERATORS_V2  # noqa: E402

DATASETS = ['cse2018', 'cic2017', 'unsw']
MODELS = ['xgboost', 'lightgbm', 'catboost']
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'ffev2_tuning.json'))
N_COHORT = 1000
WINDOW_POOL = 12
GENERIC_WINDOWS = [8192, 14600, 29200, 64240, 65535]   # common OS defaults; needs no training data

_WORKER = {}


def make_config(spec: dict, windows) -> FFEv2Config:
    spec = dict(spec)
    k = spec.pop('window_k', 6)
    if spec.pop('generic_windows', False):
        windows, k = GENERIC_WINDOWS, len(GENERIC_WINDOWS)
    gens = frozenset(spec.pop('generators', ('time', 'pad', 'chaff', 'seg', 'win')))
    return FFEv2Config(generators=gens, window_values=tuple(windows[:k]), **spec)


def evaluate(attack, X: np.ndarray, schema) -> dict:
    rows = []
    for x in X:
        r = attack.attack_sample(x)
        rows.append((bool(r.success and r.physically_valid), r.queries_used,
                     operational_cost(r.x_original, r.x_adversarial)))
    real = np.array([r[0] for r in rows])
    return {'n': len(rows), 'real_asr': 100 * float(real.mean()),
            'avg_queries': float(np.mean([r[1] for r in rows])),
            'median_op_cost_real': float(np.median([r[2] for r in rows if r[0]])) if real.any() else None,
            'mean_op_cost_real': float(np.mean([r[2] for r in rows if r[0]])) if real.any() else None}


def task(args):
    name, mt, label, spec, windows, X, stds = args
    if name not in _WORKER:
        _WORKER[name] = load_trained(name, single_thread=True)
    schema = get_cicids_schema()
    t0 = time.time()
    kind = spec.get('attack', 'v2')
    model = _WORKER[name][mt]
    if kind == 'v1':
        attack = FlowFieldEvasionAttack(model, schema, step_budget=20, feature_stds=stds)
    elif kind == 'control':
        dump = TreeDump(model.trainer.model) if mt == 'xgboost' else RawMarginModel(model.trainer)
        attack = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=True, tree_dump=dump)
    else:
        attack = FFEv2Attack(model, schema, make_config(spec, windows), stds)
    res = evaluate(attack, X, schema)
    res.update(dataset=name, model=mt, label=label, spec=spec, seconds=time.time() - t0)
    return res


def prepare():
    prep = {}
    for name in DATASETS:
        data, trained = get_models(name)
        coh = cohorts_for(trained, *data['val'], N_COHORT)
        prep[name] = {'coh': coh, 'windows': list(benign_window_candidates(*data['train'][:2], WINDOW_POOL)),
                      'stds': data['train'][0].std(axis=0) + 1e-6}
    return prep


def run(specs: dict, workers: int = 16) -> list:
    prep = prepare()
    jobs = [(n, mt, label, spec, prep[n]['windows'], prep[n]['coh'][mt]['X'], prep[n]['stds'])
            for label, spec in specs.items() for n in DATASETS for mt in MODELS]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(task, jobs))
    old = json.load(open(OUT)) if os.path.exists(OUT) else []
    labels = set(specs)
    json.dump([r for r in old if r['label'] not in labels] + results, open(OUT, 'w'), indent=1)
    return results


def summarise(results: list) -> None:
    labels = list(dict.fromkeys(r['label'] for r in results))
    print(f'{"config":34s} ' + ' '.join(f'{n[:4]}-{m[:2]}' for n in DATASETS for m in MODELS) + '  mean  q   opmed(gm)')
    for lb in labels:
        rs = {(r['dataset'], r['model']): r for r in results if r['label'] == lb}
        asr = [rs[(n, m)]['real_asr'] for n in DATASETS for m in MODELS]
        q = np.mean([rs[(n, m)]['avg_queries'] for n in DATASETS for m in MODELS])
        gm = np.exp(np.mean([np.log(max(rs[(n, m)]['median_op_cost_real'] or 1e-9, 1e-9)) for n in DATASETS for m in MODELS]))
        print(f'{lb:34s} ' + ' '.join(f'{a:7.1f}' for a in asr) + f'  {np.mean(asr):5.1f} {q:5.1f} {gm:.2e}')


def _one_factor():
    g = ('time', 'pad', 'chaff', 'seg', 'win')
    specs = {'base': {}}
    for k in (2, 3, 6, 8):
        specs[f'ratio={k}'] = {'ladder_ratio': float(k)}
    for k in (4, 8, 10):
        specs[f'levels={k}'] = {'ladder_levels': k}
    for k in (2, 4):
        specs[f'beam={k}'] = {'beam': k}
    for k in (0.0, 0.5, 2.0):
        specs[f'lambda={k}'] = {'cost_lambda': k}
    for k in (0, 1, 6, 99):
        specs[f'relax={k}'] = {'relax_after': k}
    for k in (0, 3, 12):
        specs[f'win_k={k}'] = {'window_k': k}
    for k in (4, 12, 20):
        specs[f'iter={k}'] = {'max_iter': k}
    for k in (4, 8, 16):
        specs[f'snap_pf={k}'] = {'snap': True, 'snap_per_feature': k}
    specs['no_seg'] = {'generators': tuple(x for x in g if x != 'seg')}
    specs['no_chaff'] = {'generators': tuple(x for x in g if x != 'chaff')}
    specs['no_time'] = {'generators': tuple(x for x in g if x != 'time')}
    specs['no_pad'] = {'generators': tuple(x for x in g if x != 'pad')}
    specs['plus_hdr'] = {'generators': g + ('hdr',)}
    return specs


def _ladder_stage():
    specs = {}
    for k in (3, 4, 6, 8, 12, 16):
        specs[f'span ratio={k}'] = {'ladder_ratio': float(k)}
    for k in (1, 16, 100):
        specs[f'span pad_min={k}'] = {'ladder_ratio': 6.0, 'pad_min': float(k)}
    for k in (100, 10000):
        specs[f'span time_min={k}'] = {'ladder_ratio': 6.0, 'time_min_us': float(k)}
    for k in (512, 16384):
        specs[f'span pkt_max={k}'] = {'ladder_ratio': 6.0, 'pkt_max': k}
    return specs


def _compose_stage():
    specs = {'A ratio6': {}}
    for r in (3, 4, 6, 8):
        for k in (3, 6):
            specs[f'refine={k} ratio={r}'] = {'ladder_ratio': float(r), 'refine_steps': k}
    specs['refine=4 pkt16k'] = {'refine_steps': 4, 'pkt_max': 16384}
    specs['refine=4 beam2'] = {'refine_steps': 4, 'beam': 2}
    specs['refine=4 snap8'] = {'refine_steps': 4, 'snap': True, 'snap_per_feature': 8}
    specs['refine=4 snap16 beam2'] = {'refine_steps': 4, 'snap': True, 'snap_per_feature': 16, 'beam': 2}
    specs['refine=4 plus_hdr'] = {'refine_steps': 4, 'generators': ('time', 'pad', 'chaff', 'seg', 'win', 'hdr')}
    return specs


FROZEN = {'score': {'ladder_ratio': 6.0, 'refine_steps': 4, 'window_k': 6},
          'snap': {'ladder_ratio': 6.0, 'refine_steps': 4, 'window_k': 6, 'snap': True, 'snap_per_feature': 8}}


def _final_stage():
    return {'FFE-v2 frozen (score-only)': FROZEN['score'], 'FFE-v2 frozen (+snapping)': FROZEN['snap'],
            'FFE-v1': {'attack': 'v1'}, 'White-box control (project in loop)': {'attack': 'control'},
            'FFE-v2 generic windows': {**FROZEN['score'], 'window_k': 0, 'generic_windows': True}}


STAGES = {'final': _final_stage(), 'base': {'base': {}}, 'ablate': _one_factor(), 'ladder': _ladder_stage(), 'compose': _compose_stage()}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    summarise(run(STAGES[args.stage], args.workers))
