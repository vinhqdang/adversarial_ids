"""Packet-level replay of FFE-v2 adversarial flows (frozen configuration, test cohort, XGBoost target).

Each successful flow's integer action ledger (delay, chaff packets, re-segmented packets, padding, window)
is applied to a synthetic packet trace of the original flow (see src/validation/packet_replay.py), the 12
features are re-extracted and the target is re-scored. Writes results/ffev2_replay.json.

Usage: python experiments/run_ffev2_replay.py [--n 300]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ffev2_setup import cohorts_for, get_models, get_cicids_schema  # noqa: E402
import run_full_benchmark as rfb  # noqa: E402
from tune_ffev2 import make_config  # noqa: E402
from src.attacks.ffe_v2 import FFEv2Attack, benign_window_candidates  # noqa: E402
from src.datasets.schema import verify_physical_feasibility  # noqa: E402
from src.validation.packet_replay import CHAFF_MODES, plan_from_ledger, relative_error, replay_plan  # noqa: E402

OUT = os.path.join(rfb.RESULTS_DIR, 'ffev2_replay.json')


def run_dataset(name: str, spec: dict, n: int) -> dict:
    schema = get_cicids_schema()
    data, trained = get_models(name)
    stds = data['train'][0].std(axis=0) + 1e-6
    model = trained['xgboost']
    coh = cohorts_for(trained, *data['test'], 1000)['xgboost']
    X = coh['X'][:n]
    windows = list(benign_window_candidates(*data['train'][:2], 12))
    atk = FFEv2Attack(model, schema, make_config(spec, windows), stds)
    flows = []
    for x in X:
        r = atk.attack_sample(x)
        if r.success and r.physically_valid:
            flows.append((x, r.x_adversarial, atk.last_ledger))
    rep = {'n_flows': len(X), 'n_successful': len(flows), 'modes': {}}
    for mode in CHAFF_MODES:
        close, kept, unmet_seg, unmet_pad, feas, uses_seg = 0, 0, 0, 0, 0, 0
        for i, (x, adv, led) in enumerate(flows):
            plan = plan_from_ledger(x, led.delay_us, led.n_chaff, led.n_segment, led.pad_bytes,
                                    led.header_option_bytes, led.window, mode)
            out = replay_plan(x, plan, seed=i)
            feat = out['features']
            close += int(relative_error(adv, feat).max() < 0.01)
            feas += int(verify_physical_feasibility(x, feat, schema))
            kept += int(model.predict_proba(feat)[0, 1] < 0.5)
            unmet_seg += int(out['segment_unmet'] > 0)
            unmet_pad += int(out['pad_unmet_bytes'] > 0)
            uses_seg += int(led.n_segment > 0)
        k = max(1, len(flows))
        rep['modes'][mode] = {'features_within_1pct_pct': 100 * close / k, 'still_feasible_pct': 100 * feas / k,
                              'evasion_retained_pct': 100 * kept / k, 'flows_with_unmet_segmentation': unmet_seg,
                              'flows_with_unmet_padding': unmet_pad, 'flows_using_segmentation': uses_seg,
                              'replayed_asr_of_all_pct': 100 * kept / len(X)}
        print(name, mode, rep['modes'][mode], flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--datasets', nargs='+', default=['cse2018', 'cic2017', 'unsw'])
    ap.add_argument('--config', default=os.path.join(rfb.RESULTS_DIR, 'ffev2_config.json'))
    args = ap.parse_args()
    spec = json.load(open(args.config))['score']
    out = {name: run_dataset(name, spec, args.n) for name in args.datasets}
    json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
