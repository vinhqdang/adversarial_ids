"""Builds results/seeds_ffev2_report.md from results/seeds_ffev2_<dataset>.json."""

import json
import os

import numpy as np

from seeds_stats import exact_p, mcnemar_counts, mean_std, pooled_mcnemar, win_tie_loss

NAMES = {'cse2018': 'CSE-CIC-IDS2018', 'cic2017': 'CIC-IDS2017', 'unsw': 'UNSW-NB15'}
OTHERS = ['FFE-v1', 'Control', 'CAPGD']
LABEL = {'FFE-v2': 'FFE-v2', 'FFE-v1': 'FFE-v1', 'Control': 'White-box control', 'CAPGD': 'CAPGD'}


def _fmt(vals, digits=1):
    m, s = mean_std(vals)
    return 'n/a' if np.isnan(m) else f'{m:.{digits}f} +- {s:.{digits}f}'


def _cell(runs, mt, atk, key):
    return [r['targets'][mt]['attacks'][atk][key] for r in runs if mt in r['targets']]


def write_report(results_dir: str, datasets) -> None:
    lines = ['# Multi-seed evaluation of the frozen FFE-v2', '',
             'Each seed retrains the targets on a new stratified split and redraws the cohort (up to 300 flows detected by '
             'XGBoost; LightGBM / CatBoost are attacked on the part of that cohort they also detect). Config frozen '
             '(results/ffev2_config.json, score-only). No tuning on these runs. Values are mean +- std (sample, ddof=1) over seeds.', '']
    for ds in datasets:
        fp = os.path.join(results_dir, f'seeds_ffev2_{ds}.json')
        if not os.path.exists(fp):
            continue
        runs = json.load(open(fp))['runs']
        lines += [f'## {NAMES[ds]}', '']
        for mt in ('xgboost', 'lightgbm', 'catboost'):
            sub = [r for r in runs if mt in r['targets']]
            if not sub:
                continue
            ns = [r['targets'][mt]['attacks']['FFE-v2']['n'] for r in sub]
            lines += [f'### {mt} (seeds {[r["seed"] for r in sub]}, n per seed {ns})', '',
                      '| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |', '|---|---|---|---|---|---|']
            for atk in ['FFE-v2'] + OTHERS:
                per = _cell(sub, mt, atk, 'real_asr')
                lines.append(f'| {LABEL[atk]} | {_fmt(per)} | {_fmt(_cell(sub, mt, atk, "pfr"))} | '
                             f'{_fmt(_cell(sub, mt, atk, "avg_queries"))} | {_fmt(_cell(sub, mt, atk, "median_op_cost_real"), 4)} | '
                             f'{", ".join(f"{v:.1f}" for v in per)} |')
            lines += ['', '| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |',
                      '|---|---|---|---|---|---|']
            for o in OTHERS:
                v2 = _cell(sub, mt, 'FFE-v2', 'real_asr')
                w, t, l = win_tie_loss(v2, _cell(sub, mt, o, 'real_asr'))
                pairs = [mcnemar_counts(r['targets'][mt]['attacks']['FFE-v2']['real'], r['targets'][mt]['attacks'][o]['real']) for r in sub]
                pool = pooled_mcnemar(pairs)
                lines.append(f'| {LABEL[o]} | {w}/{t}/{l} | {"; ".join(f"{a}/{b}" for a, b in pairs)} | '
                             f'{"; ".join(f"{exact_p(a, b):.2g}" for a, b in pairs)} | {pool["v2_only"]} / {pool["other_only"]} | {pool["p"]:.3g} |')
            lines.append('')
    with open(os.path.join(results_dir, 'seeds_ffev2_report.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
