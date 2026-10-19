"""Step-1 diagnosis of FFE-v1 failures (validation-split cohorts only; no test flow is used).

For every corpus and target model: per-family success of FFE-v1, which moves the projected white-box
control makes on flows where FFE-v1 fails, the support of FFE-v1's fixed window target in the training
data, and a reachability test (multi-start random search over all admissible generators jointly, every
sample projected onto the flow manifold) on the flows FFE-v1 fails: if no sample crosses the boundary
the flow is (empirically) not reachable by any combination of the generators, whichever the algorithm.

Writes results/ffev2_diagnosis.json.   Usage: python experiments/diagnose_ffev2.py [--n 1000]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ffev2_setup import cohorts_for, get_models, get_cicids_schema  # noqa: E402
from src.attacks.ffe_v2 import chaff_header  # noqa: E402
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack  # noqa: E402
from src.attacks.whitebox_tree import RawMarginModel, TreeDump, WhiteBoxLeafTupleAttack, project_batch_with_origin  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'ffev2_diagnosis.json'))
N_SAMPLES = 20000
MAX_FLOWS = 80


def window_support(X: np.ndarray, y: np.ndarray) -> dict:
    tcp = X[:, 1] == 6
    out = {}
    for label, m in (('benign', tcp & (y == 0)), ('attack', tcp & (y == 1))):
        w = X[m, 7]
        out[label] = {'n_tcp': int(m.sum()), 'share_eq_64240': float((w == 64240).mean()) if m.any() else None,
                      'share_ge_60000': float((w >= 60000).mean()) if m.any() else None,
                      'share_eq_65535': float((w == 65535).mean()) if m.any() else None}
    return out


def reachable(model, x: np.ndarray, windows: np.ndarray, rng, schema) -> tuple:
    """Random multi-start search over all generators jointly. Returns (crosses, crossing share, min score)."""
    n = N_SAMPLES
    h0, l0 = chaff_header(x)
    on = {k: rng.rand(n) < 0.5 for k in ('time', 'pad', 'chaff', 'seg', 'win')}
    R = np.tile(x, (n, 1))
    R[:, 6] = np.where(on['time'], x[6] * np.exp(rng.uniform(0, np.log(1.1e8 / max(x[6], 1)), n)), x[6])
    pad = np.where(on['pad'], np.exp(rng.uniform(0, np.log(1500 * x[2] + 1), n)), 0)
    chaff = np.where(on['chaff'], np.exp(rng.uniform(0, np.log(4096), n)).astype(int), 0)
    seg = np.where(on['seg'], np.exp(rng.uniform(0, np.log(max(2.0, x[4])), n)).astype(int), 0)
    seg = np.minimum(seg, np.maximum(0, np.floor(x[4] + chaff * l0 + pad) - x[2] - chaff))
    R[:, 4], R[:, 2], R[:, 11] = x[4] + pad + chaff * l0, x[2] + chaff + seg, x[11] + (chaff + seg) * h0
    R[:, 7] = np.where(on['win'], windows[rng.randint(0, len(windows), n)], x[7]) if x[1] == 6 else 0
    p = model.predict_proba(project_batch_with_origin(R, x, schema))[:, 1]
    return bool((p < 0.5).any()), float((p < 0.5).mean()), float(p.min())


def control_profile(rows) -> dict:
    """Which feature moves the projected control makes on the flows it evades (x_orig, x_adv pairs)."""
    d = np.array([a - o for o, a in rows])
    if len(d) == 0:
        return {}
    return {'n': len(d), 'share_fwd_pkts_up': float((d[:, 2] > 0).mean()),
            'share_pure_segmentation_like (dfwd>0, dlen=0)': float(((d[:, 2] > 0) & (d[:, 4] == 0)).mean()),
            'share_duration_up': float((d[:, 6] > 0).mean()), 'share_window_changed': float((d[:, 7] != 0).mean()),
            'share_payload_up': float((d[:, 4] > 0).mean()),
            'median_fwd_pkts_added': float(np.median(d[d[:, 2] > 0, 2])) if (d[:, 2] > 0).any() else 0.0}


def diagnose(name: str, n: int) -> dict:
    schema = get_cicids_schema()
    data, trained = get_models(name)
    stds = data['train'][0].std(axis=0) + 1e-6
    coh = cohorts_for(trained, *data['val'], n)
    rng = np.random.RandomState(0)
    windows = np.array(sorted(set(data['train'][0][data['train'][0][:, 1] == 6][:, 7].astype(int))) or [0])
    rep = {'window_support_train': window_support(*data['train'][:2]), 'models': {}}
    for mt, model in trained.items():
        X, fam = coh[mt]['X'], coh[mt]['family']
        v1 = FlowFieldEvasionAttack(model, schema, feature_stds=stds)
        ok = np.array([v1.attack_sample(x).success for x in X])
        entry = {'n': int(len(X)), 'v1_asr': 100 * float(ok.mean()),
                 'v1_by_family': {f: [int((fam == f).sum()), 100 * float(ok[fam == f].mean())] for f in sorted(set(fam))}}
        fail = np.where(~ok)[0]
        entry['v1_failures'] = int(len(fail))
        sel = fail[rng.permutation(len(fail))[:MAX_FLOWS]]
        res = [reachable(model, X[i], windows, rng, schema) for i in sel]
        entry['reachability_sample'] = {
            'flows_tested': int(len(sel)), 'samples_per_flow': N_SAMPLES,
            'share_with_any_crossing': float(np.mean([r[0] for r in res])) if res else None,
            'median_crossing_share_when_reachable': float(np.median([r[1] for r in res if r[0]])) if any(r[0] for r in res) else None,
            'median_min_score': float(np.median([r[2] for r in res])) if res else None,
            'by_family_unreachable': {f: int(sum(1 for i, r in zip(sel, res) if fam[i] == f and not r[0]))
                                      for f in sorted(set(fam[sel]))}}
        dump = TreeDump(model.model) if mt == 'xgboost' else RawMarginModel(model)
        ctrl = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=True, tree_dump=dump)
        won = [(X[i], r.x_adversarial) for i in fail if (r := ctrl.attack_sample(X[i])).success]
        entry['control_wins_on_v1_failures'] = len(won)
        entry['control_profile_on_v1_failures'] = control_profile(won)
        rep['models'][mt] = entry
        print(name, mt, json.dumps({k: entry[k] for k in ('v1_asr', 'v1_failures', 'control_wins_on_v1_failures')}),
              entry['reachability_sample']['share_with_any_crossing'], flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    args = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in args.datasets:
        out[name] = diagnose(name, args.n)
        json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
