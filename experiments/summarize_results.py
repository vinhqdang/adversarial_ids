"""
Turn the raw result JSON files into the LaTeX tables and statistics quoted in the paper.

Writes paper/tables/*.tex and results/summary.json. Nothing in the manuscript tables is typed
by hand: re-running the experiments and this script regenerates every number.
"""

import json
import os
import sys
from collections import Counter
from typing import Dict

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import XGB_CFG, load_arrays  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'paper', 'tables')
DATASETS = [('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15')]
FFE = 'Flow-Field Evasion (FFE)'
TSBS_OFF = 'FFE without TSBS'
SHORT = {'Unconstrained Feature-Space': 'Unconstrained feature space', 'ZOO-NIDS': 'ZOO-NIDS',
         'HopSkipJump-NIDS': 'HopSkipJump-NIDS', 'CAA / MOEVA': 'MOEVA (search stage of CAA)',
         'Traffic Manipulator PSO': 'PSO (Traffic Manipulator, projected)',
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


def fmt_num(v) -> str:
    if v is None:
        return '--'
    if v >= 1000:
        return f'{v:,.0f}'
    if v >= 10:
        return f'{v:.1f}'
    return f'{v:.3f}'


def pfr_of_evasive(a: Dict) -> str:
    """Share of the evasive candidates that are schema-feasible (real ASR / ASR)."""
    return '--' if a['asr'] == 0 else f"{100 * a['real_asr'] / a['asr']:.1f}"


def master_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
            r'\textbf{Method} & \textbf{ASR} & \textbf{PFR} & \textbf{Real ASR [95\% CI]} & '
            r'\textbf{Queries} & \textbf{Op.\ cost} & \textbf{$D_{L_1}$} & \textbf{ms/flow} \\']
    for key, label in DATASETS:
        d = data[key]['models']['xgboost']
        c = d['cohort']
        rows += [r'\midrule', r'\multicolumn{8}{@{}l}{\textbf{' + label + r'} (XGBoost accuracy '
                 + f"{100 * d['clean']['accuracy']:.2f}" + r'\%, test attacks ' + f"{c['n_malicious_test']:,}"
                 + r', detected by both models ' + f"{c['n_detected']:,}" + r')} \\', r'\midrule']
        for name, a in d['attacks'].items():
            if name == TSBS_OFF:
                continue
            lo, hi = a['real_asr_ci95']
            label_m = SHORT[name] + (' ' + CITE[name] if name in CITE else '')
            cell = f"{a['real_asr']:.1f} [{lo:.1f}, {hi:.1f}]"
            if name == FFE:
                label_m, cell = r'\textbf{FFE (ours)}', r'\textbf{' + cell + '}'
            rows.append(f"{label_m} & {a['asr']:.1f} & {pfr_of_evasive(a)} & {cell} & {a['avg_queries']:.1f} & "
                        f"{fmt_num(a['median_op_cost_real'])} & {fmt_num(a['median_dl1_real'])} & {a['ms_per_flow']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def corpora_table() -> str:
    rows = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Raw rows} & \textbf{Valid} & \textbf{Duplicates} & '
            r'\textbf{Kept} & \textbf{Attacks} & \textbf{Train} & \textbf{Test} \\', r'\midrule']
    for key, label in DATASETS:
        with open(os.path.join(ROOT, 'data', 'full', f'{key}_stats.json')) as fh:
            st = json.load(fh)
        split = load(f'full_{key}.json')['split_sizes']
        rows.append(f"{label} & {st['raw_rows']:,} & {st['valid_records']:,} & {st['duplicates_removed']:,} & "
                    f"{st['after_dedup']:,} & {st['attacks']:,} & {split['train']:,} & {split['test']:,} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def family_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Attack family} & \textbf{$n$} & \textbf{Real ASR (\%)} & '
            r'\textbf{Queries} \\']
    for key, label in DATASETS:
        d = data[key]['models']['xgboost']
        fam = np.array(d['cohort']['family'])
        out = d['attacks'][FFE]['outcomes']
        rows.append(r'\midrule')
        first = True
        for f in sorted(set(fam), key=lambda s: (-np.sum(fam == s), s)):
            sel = [o for o, fm in zip(out, fam) if fm == f]
            real = np.mean([o['success'] and o['valid'] for o in sel]) * 100
            q = np.mean([o['queries'] for o in sel])
            rows.append(f"{label if first else ''} & {f} & {len(sel)} & {real:.1f} & {q:.1f} \\\\")
            first = False
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def model_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}llccccccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Model} & \textbf{Acc.\ (\%)} & \textbf{F1 (\%)} & '
            r'\textbf{AUC (\%)} & \textbf{FFE Real ASR (\%)} & \textbf{PFR (\%)} & \textbf{Queries} & '
            r'\textbf{Surr.\ agr.\ (\%)} \\', r'\midrule']
    for key, label in DATASETS:
        for m, ml in (('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM')):
            d = data[key]['models'][m]
            c, a = d['clean'], d['attacks'][FFE]
            lo, hi = a['real_asr_ci95']
            rows.append(f"{label if m == 'xgboost' else ''} & {ml} & {100 * c['accuracy']:.2f} & "
                        f"{100 * c['f1']:.2f} & {100 * c['roc_auc']:.2f} & "
                        f"{a['real_asr']:.1f} [{lo:.1f}, {hi:.1f}] & {pfr_of_evasive(a)} & "
                        f"{a['avg_queries']:.1f} & {100 * d['surrogate_agreement']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def tsbs_table(data: Dict[str, Dict]) -> str:
    rows = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Variant} & \textbf{Real ASR (\%)} & \textbf{Queries} & \textbf{ms/flow} \\',
            r'\midrule']
    for key, label in DATASETS:
        atk = data[key]['models']['xgboost']['attacks']
        for i, (name, vl) in enumerate(((FFE, 'FFE'), (TSBS_OFF, 'FFE without TSBS'))):
            a = atk[name]
            lo, hi = a['real_asr_ci95']
            rows.append(f"{label if i == 0 else ''} & {vl} & {a['real_asr']:.1f} [{lo:.1f}, {hi:.1f}] & "
                        f"{a['avg_queries']:.1f} & {a['ms_per_flow']:.1f} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows)


def seeds_table() -> str:
    rows = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Acc.\ (\%)} & \textbf{FFE} & \textbf{CAPGD} & \textbf{NIDSGAN-style} \\',
            r'\midrule']
    for key, label in DATASETS:
        runs = load(f'seeds_{key}.json')['runs']

        def ms(vals, scale=1.0, dec=1):
            v = np.array(vals) * scale
            return f'{v.mean():.{dec}f} $\\pm$ {v.std(ddof=1):.{dec}f}'
        rows.append(f"{label} ({len(runs)} seeds) & {ms([r['accuracy'] for r in runs], 100, 2)} & "
                    + ' & '.join(ms([r['attacks'][k]['real_asr'] for r in runs]) for k in ('FFE', 'CAPGD', 'NIDSGAN-style'))
                    + r' \\')
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
    rows = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
            r'\textbf{Model} & \textbf{Acc.} & \textbf{F1} & \textbf{Attack recall} & '
            r'\textbf{FFE $M{=}5$} & \textbf{FFE $M{=}20$} & \textbf{CAPGD} \\', r'\midrule']
    for name, m in df['models'].items():
        rows.append(f"{name} & {m['clean_accuracy']:.2f} & {m['clean_f1']:.2f} & {m['clean_attack_recall']:.2f} & "
                    f"{m['ffe_m5']['real_asr']:.1f} & {m['ffe_m20']['real_asr']:.1f} & {m['capgd']['real_asr']:.1f} \\\\")
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


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise(data: Dict[str, Dict]) -> Dict:
    """Exact McNemar (two-sided binomial on discordant pairs) of FFE against every baseline."""
    out, flat = {}, []
    for key, _ in DATASETS:
        atk = data[key]['models']['xgboost']['attacks']
        out[key] = {}
        for name, a in atk.items():
            if name in (FFE, TSBS_OFF):
                continue
            n = a['n']
            f, b = real_flags(atk[FFE], n), real_flags(a, n)
            only_f, only_b = int(np.sum(f & ~b)), int(np.sum(~f & b))
            p = 1.0 if only_f + only_b == 0 else binomtest(only_f, only_f + only_b, 0.5).pvalue
            out[key][name] = {'n': n, 'ffe_only': only_f, 'baseline_only': only_b, 'p': float(p),
                              'ffe_real_asr_same_flows': 100 * float(f.mean()),
                              'baseline_real_asr': 100 * float(b.mean())}
            flat.append((key, name, p))
    adj = holm([p for _, _, p in flat])
    for (key, name, _), q in zip(flat, adj):
        out[key][name]['p_holm'] = float(q)
    return out


def efficiency_ratios(data: Dict[str, Dict]) -> Dict:
    """Queries and time per flow of the surrogate-free constrained searches relative to FFE."""
    out = {}
    for key, _ in DATASETS:
        atk = data[key]['models']['xgboost']['attacks']
        f = atk[FFE]
        out[key] = {name: {'queries_x': atk[name]['avg_queries'] / f['avg_queries'],
                           'time_x': atk[name]['ms_per_flow'] / f['ms_per_flow']}
                    for name in ('CAA / MOEVA', 'Traffic Manipulator PSO')}
    return out


def seed_wins() -> Dict:
    """In how many seeds FFE beats each surrogate attack, and the smallest gap."""
    out = {}
    for key, _ in DATASETS:
        runs = load(f'seeds_{key}.json')['runs']
        out[key] = {k: {'ffe_better_in': int(sum(r['attacks']['FFE']['real_asr'] > r['attacks'][k]['real_asr']
                                                  for r in runs)), 'n_seeds': len(runs),
                        'min_gap': float(min(r['attacks']['FFE']['real_asr'] - r['attacks'][k]['real_asr']
                                             for r in runs))}
                    for k in ('CAPGD', 'NIDSGAN-style')}
    return out


def write(name: str, text: str) -> None:
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


def main() -> None:
    data = {k: load(f'full_{k}.json') for k, _ in DATASETS}
    write('tab_master.tex', master_table(data))
    write('tab_corpora.tex', corpora_table())
    write('tab_family.tex', family_table(data))
    write('tab_models.tex', model_table(data))
    write('tab_tsbs.tex', tsbs_table(data))
    write('tab_seeds.tex', seeds_table())
    write('tab_ablation.tex', ablation_table(load('ablation_results.json')))
    write('tab_transfer.tex', transfer_table(load('transferability_results.json')))
    write('tab_defense.tex', defense_table(load('defense_results.json')))
    summary = {'pairwise_mcnemar': pairwise(data), 'efficiency_ratios': efficiency_ratios(data),
               'seed_wins': seed_wins(),
               'root_splits': {k: root_splits(k) for k, _ in DATASETS},
               'dedup': {k: json.load(open(os.path.join(ROOT, 'data', 'full', f'{k}_stats.json')))
                         for k, _ in DATASETS},
               'split_sizes': {k: data[k]['split_sizes'] for k, _ in DATASETS},
               'surrogate_agreement': {k: {m: data[k]['models'][m].get('surrogate_agreement')
                                           for m in data[k]['models']} for k, _ in DATASETS},
               'seeds': {k: load(f'seeds_{k}.json')['runs'] for k, _ in DATASETS}}
    with open(os.path.join(RES, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'seeds'}, indent=1)[:6000])


if __name__ == '__main__':
    main()
