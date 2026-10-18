"""
Turn the raw result JSON files into the LaTeX tables and statistics quoted in the paper.

Writes paper/tables/*.tex and results/summary.json. Nothing in the manuscript tables is typed
by hand: re-running the experiments and this script regenerates every number.
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import XGB_CFG, load_arrays  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'paper', 'tables')
DATASETS = [('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15')]
FFE = 'Flow-Field Evasion (FFE)'
SHORT = {'Unconstrained Feature-Space': 'Unconstrained feature space', 'ZOO-NIDS': 'ZOO-NIDS',
         'HopSkipJump-NIDS': 'HopSkipJump-NIDS', 'CAA / MOEVA': 'CAA/MOEVA',
         'Traffic Manipulator PSO': 'Traffic Manipulator PSO',
         'CAPGD (surrogate transfer)': 'CAPGD (surrogate)',
         'NIDSGAN-style generator': 'NIDSGAN-style', FFE: 'FFE (ours)'}
CITE = {'ZOO-NIDS': r'\cite{chen2017zoo}', 'HopSkipJump-NIDS': r'\cite{chen2020hopskipjump}',
        'CAA / MOEVA': r'\cite{simonetto2024constrained, ghamizi2020moeva}',
        'Traffic Manipulator PSO': r'\cite{han2021traffic}',
        'CAPGD (surrogate transfer)': r'\cite{simonetto2024constrained}',
        'NIDSGAN-style generator': r'\cite{zolbayar2022nidsgan}'}


def load(name: str) -> Dict:
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def real_flags(attack: Dict, n: int) -> np.ndarray:
    return np.array([o['success'] and o['valid'] for o in attack['outcomes'][:n]])


def fmt_cost(v) -> str:
    if v is None:
        return '--'
    return f'{v:,.1f}' if v >= 100 else f'{v:.3f}'


def master_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
            r'\textbf{Method} & \textbf{ASR} & \textbf{PFR} & \textbf{Real ASR [95\% CI]} & '
            r'\textbf{Queries} & \textbf{Cost} & \textbf{ms/flow} \\']
    for key, label in DATASETS:
        d = data[key]
        det = d['models']['xgboost']['cohort']
        rows += [r'\midrule', r'\multicolumn{7}{@{}l}{\textbf{' + label + r'} (XGBoost: accuracy '
                 + f"{100 * d['models']['xgboost']['clean']['accuracy']:.2f}" + r'\%, test attacks '
                 + f"{det['n_malicious_test']:,}" + r', detected ' + f"{det['n_detected']:,}" + r')} \\',
                 r'\midrule']
        for name, a in d['models']['xgboost']['attacks'].items():
            lo, hi = a['real_asr_ci95']
            label_m = SHORT[name] + (' ' + CITE[name] if name in CITE else '')
            cell = f"{a['real_asr']:.1f} [{lo:.1f}, {hi:.1f}]"
            if name == FFE:
                label_m, cell = r'\textbf{FFE (ours)}', r'\textbf{' + cell + '}'
            rows.append(f"{label_m} & {a['asr']:.1f} & {a['pfr']:.1f} & {cell} & {a['avg_queries']:.1f} & "
                        f"{fmt_cost(a['mean_cost_all'])} & {a['ms_per_flow']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def family_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Attack family} & \textbf{$n$} & \textbf{Real ASR (\%)} & '
            r'\textbf{Queries} \\']
    for key, label in DATASETS:
        d = data[key]
        fam = np.array(d['models']['xgboost']['cohort']['family'])
        out = d['models']['xgboost']['attacks'][FFE]['outcomes']
        rows.append(r'\midrule')
        for i, f in enumerate(sorted(set(fam), key=lambda s: (-np.sum(fam == s), s))):
            sel = [o for o, fm in zip(out, fam) if fm == f]
            if len(sel) < 5:
                continue
            real = np.mean([o['success'] and o['valid'] for o in sel]) * 100
            q = np.mean([o['queries'] for o in sel])
            rows.append(f"{label if i == 0 else ''} & {f} & {len(sel)} & {real:.1f} & {q:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def model_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Model} & \textbf{Acc.\ (\%)} & \textbf{F1 (\%)} & '
            r'\textbf{AUC (\%)} & \textbf{FFE Real ASR (\%)} & \textbf{PFR (\%)} & \textbf{Queries} \\',
            r'\midrule']
    for key, label in DATASETS:
        for m, ml in (('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')):
            c = data[key]['models'][m]['clean']
            a = data[key]['models'][m]['attacks'][FFE]
            lo, hi = a['real_asr_ci95']
            rows.append(f"{label if m == 'xgboost' else ''} & {ml} & {100 * c['accuracy']:.2f} & "
                        f"{100 * c['f1']:.2f} & {100 * c['roc_auc']:.2f} & "
                        f"{a['real_asr']:.1f} [{lo:.1f}, {hi:.1f}] & {a['pfr']:.1f} & {a['avg_queries']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def ablation_table(ab: Dict) -> str:
    def block(title, items, key_fmt=lambda k: k):
        out = [r'\midrule', r'\multicolumn{5}{@{}l}{\textit{' + title + r'}} \\']
        for k, v in items.items():
            out.append(f"{key_fmt(k)} & {v['asr']:.1f} & {v['pfr']:.1f} & {v['avg_queries']:.1f} & "
                       f"{v['mean_delay_ms'] / 1000:.2f} \\\\")
        return out
    rows = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
            r'\textbf{Configuration} & \textbf{ASR (\%)} & \textbf{PFR (\%)} & \textbf{Queries} & '
            r'\textbf{Added delay (s)} \\']
    rows += block('Panel A: step budget $M$', ab['budget'], lambda k: f'$M={k}$')
    rows += block('Panel B: generator subset ($M=20$)', ab['generators'])
    rows += block('Panel C: operational cost regime ($M=20$)', ab['costs'])
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def transfer_table(tr: Dict) -> str:
    names = tr['model_names']
    rows = [r'\begin{tabular}{@{}l' + 'c' * len(names) + r'@{}}', r'\toprule',
            r'\textbf{Source $\backslash$ Target} & ' + ' & '.join(r'\textbf{' + n + '}' for n in names) + r' \\',
            r'\midrule']
    for s in names:
        cells = [(r'\textbf{%.1f}' if s == t else '%.1f') % tr['transferability_matrix'][s][t] for t in names]
        rows.append(r'\textbf{' + s + '} & ' + ' & '.join(cells) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def defense_table(df: Dict) -> str:
    rows = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
            r'\textbf{Model} & \textbf{Acc.\ (\%)} & \textbf{AUC (\%)} & \textbf{Robust recall (\%)} & '
            r'\textbf{FFE ASR (\%)} \\', r'\midrule']
    for name, m in df['models'].items():
        rows.append(f"{name} & {m['clean_accuracy']:.2f} & {m['clean_roc_auc']:.2f} & "
                    f"{m['robust_recall']:.1f} & {m['ffe_asr']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def root_splits(name: str) -> Dict:
    """Which features sit at the root of the XGBoost trees (retrained with the benchmark config)."""
    from src.models.tree_models import NIDSEnsembleTrainer
    from src.datasets.full_data import FEATURES
    X, y, _ = load_arrays(name)['train']
    t = NIDSEnsembleTrainer('xgboost', XGB_CFG)
    t.train(X, y)
    df = t.model.get_booster().trees_to_dataframe()
    roots = df[(df['Node'] == 0) & (df['Feature'] != 'Leaf')]
    names = [FEATURES[int(f[1:])] if f.startswith('f') and f[1:].isdigit() else f for f in roots['Feature']]
    gain = {FEATURES[int(f[1:])] if f.startswith('f') else f: float(g)
            for f, g in roots.groupby('Feature')['Gain'].mean().items()}
    return {'n_trees': int(len(roots)), 'root_feature_counts': dict(Counter(names)), 'mean_root_gain': gain}


def pairwise(data: Dict[str, Dict]) -> Dict:
    """Exact McNemar (two-sided binomial on discordant pairs) of FFE against every baseline."""
    out = {}
    for key, _ in DATASETS:
        atk = data[key]['models']['xgboost']['attacks']
        out[key] = {}
        for name, a in atk.items():
            if name == FFE:
                continue
            n = a['n']
            f, b = real_flags(atk[FFE], n), real_flags(a, n)
            only_f, only_b = int(np.sum(f & ~b)), int(np.sum(~f & b))
            p = 1.0 if only_f + only_b == 0 else binomtest(only_f, only_f + only_b, 0.5).pvalue
            out[key][name] = {'n': n, 'ffe_only': only_f, 'baseline_only': only_b, 'p': float(p)}
    return out


def write(name: str, text: str) -> None:
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


def main() -> None:
    data = {k: load(f'full_{k}.json') for k, _ in DATASETS}
    write('tab_master.tex', master_table(data))
    write('tab_family.tex', family_table(data))
    write('tab_models.tex', model_table(data))
    write('tab_ablation.tex', ablation_table(load('ablation_results.json')))
    write('tab_transfer.tex', transfer_table(load('transferability_results.json')))
    write('tab_defense.tex', defense_table(load('defense_results.json')))
    summary = {'pairwise_mcnemar': pairwise(data),
               'root_splits': {k: root_splits(k) for k, _ in DATASETS},
               'dedup': {k: data[k]['dedup'] for k, _ in DATASETS},
               'split_sizes': {k: data[k]['split_sizes'] for k, _ in DATASETS}}
    with open(os.path.join(RES, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
