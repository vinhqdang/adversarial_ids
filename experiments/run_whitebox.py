"""White-box leaf-tuple-style control on the main benchmark cohorts (XGBoost).

Re-creates the exact models and cohort of ``run_full_benchmark.py`` (checked against the saved
``results/full_<dataset>.json``), runs the white-box attack with and without in-loop projection,
and writes ``results/whitebox_<dataset>.json`` with the same per-attack schema plus an exact
McNemar test against FFE on the same flows.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scipy.stats import binomtest  # noqa: E402

import run_full_benchmark as rfb  # noqa: E402  (imports xgboost/lightgbm before torch)
from common import load_arrays  # noqa: E402
from src.attacks.whitebox_tree import TreeDump, WhiteBoxLeafTupleAttack  # noqa: E402
from src.datasets.schema import get_cicids_schema  # noqa: E402
from src.models.tree_models import NIDSEnsembleTrainer  # noqa: E402

FFE = 'Flow-Field Evasion (FFE)'
VARIANTS = {'Leaf-tuple-style white-box (project at end)': False,
            'Leaf-tuple-style white-box (project in loop)': True}


def real_flags(a: dict) -> np.ndarray:
    return np.array([o['success'] and o['valid'] for o in a['outcomes']])


def mcnemar(ffe: np.ndarray, other: np.ndarray) -> dict:
    only_f, only_o = int(np.sum(ffe & ~other)), int(np.sum(~ffe & other))
    p = 1.0 if only_f + only_o == 0 else float(binomtest(only_f, only_f + only_o, 0.5).pvalue)
    return {'ffe_only': only_f, 'whitebox_only': only_o, 'p': p}


def run_dataset(name: str, n: int) -> dict:
    ref = json.load(open(os.path.join(rfb.RESULTS_DIR, f'full_{name}.json')))
    schema = get_cicids_schema()
    data = load_arrays(name)
    X_tr, y_tr, _ = data['train']
    X_te, y_te, fam_te = data['test']
    stds = X_tr.std(axis=0) + 1e-6
    trained = {}
    for m_type, cfg in rfb.MODEL_CONFIGS.items():
        trainer = NIDSEnsembleTrainer(m_type, cfg)
        trainer.train(X_tr, y_tr)
        if m_type == 'xgboost':
            trainer.model.get_booster().set_param({'device': 'cpu'})
        trained[m_type] = trainer
    clean = trained['xgboost'].evaluate(X_te, y_te)
    cohort = rfb.pick_cohort(list(trained.values()), X_te, y_te, fam_te, 1000, np.random.RandomState(rfb.SEED))
    ref_x = ref['models']['xgboost']
    same_cohort = cohort['family'] == ref_x['cohort']['family']
    print(f'{name}: acc={clean["accuracy"]:.4f} (ref {ref_x["clean"]["accuracy"]:.4f}) '
          f'cohort identical to benchmark: {same_cohort}')
    if not same_cohort:
        raise RuntimeError('cohort differs from results/full_*.json; flows would not be paired')

    model = trained['xgboost']
    dump = TreeDump(model.model)
    gap = np.abs(1 / (1 + np.exp(-dump.margin(X_te[:5000]))) - model.predict_proba(X_te[:5000])[:, 1]).max()
    print(f'  tree-dump vs predict_proba max |dp| = {gap:.2e}')
    sub = cohort['X'][:n]
    out = {'dataset': name, 'tree_dump_max_prob_gap': float(gap), 'cohort_matches_benchmark': True,
           'models': {'xgboost': {'clean': clean, 'cohort': {k: cohort[k] for k in ('n_malicious_test', 'n_detected', 'family')},
                                  'attacks': {}}},
           'mcnemar_vs_ffe': {}}
    ffe_res = ref_x['attacks'][FFE]
    ffe_o = {**ffe_res, 'outcomes': ffe_res['outcomes'][:len(sub)]}
    for label, in_loop in VARIANTS.items():
        atk = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=in_loop, tree_dump=dump)
        internal = []
        orig_attack = atk.attack_sample

        def tracked(x, _a=atk, _f=orig_attack, _l=internal):
            r = _f(x)
            _l.append((_a.internal_evals, _a.raw_success))
            return r
        atk.attack_sample = tracked
        np.random.seed(rfb.SEED)
        res = rfb.run_attack(atk, sub, schema, stds)
        for o, (ie, rs) in zip(res['outcomes'], internal):
            o['internal_evals'], o['raw_success'] = int(ie), bool(rs)
        res['mean_internal_evals'] = float(np.mean([i for i, _ in internal]))
        res['raw_feature_space_asr'] = 100 * float(np.mean([r for _, r in internal]))
        out['models']['xgboost']['attacks'][label] = res
        f = real_flags(ffe_o)
        mc = mcnemar(f, real_flags(res))
        mc.update(n=len(sub), ffe_real_asr_same_flows=100 * float(f.mean()), whitebox_real_asr=res['real_asr'])
        out['mcnemar_vs_ffe'][label] = mc
        print(f'  {label}: N={res["n"]} ASR={res["asr"]:.1f} PFR={res["pfr"]:.1f} RealASR={res["real_asr"]:.1f} '
              f'q={res["avg_queries"]:.1f} internal={res["mean_internal_evals"]:.0f} rawASR={res["raw_feature_space_asr"]:.1f} '
              f'opcost_med={res["median_op_cost_real"]} | FFE {mc["ffe_real_asr_same_flows"]:.1f} '
              f'McNemar ffe_only={mc["ffe_only"]} wb_only={mc["whitebox_only"]} p={mc["p"]:.3g}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--n', type=int, default=1000)
    args = ap.parse_args()
    for name in args.datasets:
        result = run_dataset(name, args.n)
        with open(os.path.join(rfb.RESULTS_DIR, f'whitebox_{name}.json'), 'w') as fh:
            json.dump(result, fh, indent=1)


if __name__ == '__main__':
    main()
