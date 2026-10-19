"""
LaTeX tables for the FFE-v2 and defense-v2 results (imported by summarize_results.py).

Every number comes from results/ffev2_*.json, results/seeds_ffev2_*.json, results/ffev2_tuning.json,
results/ffev2_replay.json, results/whitebox_*.json and results/defense_v2_*.json.
"""

import json
import os
from typing import Callable, Dict, List

import numpy as np
from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
DATASETS = [('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15')]
LIBS = [('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM'), ('catboost', 'CatBoost')]
V2, V2S, V1 = 'FFE-v2 (score-only)', 'FFE-v2 + snapping', 'Flow-Field Evasion (FFE)'
CTRL, CAPGD = 'White-box control (project in loop)', 'CAPGD (surrogate transfer)'
FAMILY_ORDER = None


def load(name: str) -> Dict:
    with open(os.path.join(RES, name), encoding='utf-8') as fh:
        return json.load(fh)


def _real(attack: Dict) -> np.ndarray:
    return np.array([o['success'] and o['valid'] for o in attack['outcomes']])


def _tab(spec: str, header: str, body: List[str]) -> str:
    return '\n'.join([r'\begin{tabular}{@{}' + spec + r'@{}}', r'\toprule', header] + body
                     + [r'\bottomrule', r'\end{tabular}'])


def _sci(p: float) -> str:
    if p >= 0.001:
        return f'{p:.2g}' if p < 0.995 else '1'
    e = int(np.floor(np.log10(p))) if p > 0 else -300
    return rf'${p / 10 ** e:.1f}\times10^{{{e}}}$'


def crosslib_table(v2: Dict[str, Dict]) -> str:
    head = (r'\textbf{Corpus} & \textbf{Target} & \textbf{$N$} & \textbf{Acc.} & \textbf{FFE-v1} & '
            r'\textbf{Control} & \textbf{FFE-v2} & \textbf{+snap} & \textbf{$q$ v1} & \textbf{$q$ v2} \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            e = v2[key]['models'][m]
            a = e['attacks']
            body.append(f"{label if i == 0 else ''} & {ml} & {a[V2]['n']} & {100 * e['clean']['accuracy']:.2f} & "
                        f"{a[V1]['real_asr']:.1f} & {a[CTRL]['real_asr']:.1f} & "
                        f"\\textbf{{{a[V2]['real_asr']:.1f}}} & {a[V2S]['real_asr']:.1f} & "
                        f"{a[V1]['avg_queries']:.1f} & {a[V2]['avg_queries']:.1f} \\\\")
    return _tab('llccccccrr', head, body)


def _ms(vals, dec=1) -> str:
    v = np.asarray(vals, dtype=float)
    return f'{v.mean():.{dec}f} $\\pm$ {v.std(ddof=1):.{dec}f}'


def seeds_tables() -> Dict[str, str]:
    """Main five-seed table and the supplementary pooled-McNemar table."""
    head = (r'\textbf{Corpus} & \textbf{Target} & \textbf{Seeds} & \textbf{FFE-v2} & \textbf{FFE-v1} & '
            r'\textbf{Control} & \textbf{CAPGD} & \textbf{$q$ (v2)} \\')
    head2 = (r'\textbf{Corpus} & \textbf{Target} & \textbf{vs FFE-v1} & \textbf{vs control} & \textbf{vs CAPGD} \\')
    body, body2 = [], []
    for key, label in DATASETS:
        runs = load(f'seeds_ffev2_{key}.json')['runs']
        body.append(r'\midrule')
        body2.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            rs = [r['targets'][m]['attacks'] for r in runs if m in r['targets']]
            cap = _ms([r['CAPGD']['real_asr'] for r in rs])
            body.append(f"{label if i == 0 else ''} & {ml} & {len(rs)} & "
                        f"\\textbf{{{_ms([r['FFE-v2']['real_asr'] for r in rs])}}} & "
                        f"{_ms([r['FFE-v1']['real_asr'] for r in rs])} & "
                        f"{_ms([r['Control']['real_asr'] for r in rs])} & {cap} & "
                        f"{np.mean([r['FFE-v2']['avg_queries'] for r in rs]):.1f} \\\\")
            cells = []
            for other in ('FFE-v1', 'Control', 'CAPGD'):
                f = np.concatenate([np.array(r['FFE-v2']['real'], bool) for r in rs])
                o = np.concatenate([np.array(r[other]['real'], bool) for r in rs])
                a_, b_ = int(np.sum(f & ~o)), int(np.sum(~f & o))
                p = 1.0 if a_ + b_ == 0 else binomtest(a_, a_ + b_, 0.5).pvalue
                wins = sum(r['FFE-v2']['real_asr'] > r[other]['real_asr'] for r in rs)
                ties = sum(r['FFE-v2']['real_asr'] == r[other]['real_asr'] for r in rs)
                cells.append(f"{a_}/{b_}, {_sci(p)} ({wins}W {ties}T)")
            body2.append(f"{label if i == 0 else ''} & {ml} & " + ' & '.join(cells) + r' \\')
    return {'tab_seeds.tex': _tab('llcccccr', head, body), 'tab_seeds_pooled.tex': _tab('llccc', head2, body2)}


def _fmt3(v) -> str:
    return '--' if v is None else f'{v:.3g}'


def detail_table(v2: Dict[str, Dict]) -> str:
    """Queries and median operational cost per cell, and exact McNemar of FFE-v2 against each method."""
    head = (r'\textbf{Corpus} & \textbf{Target} & \textbf{$q$ ctrl} & \textbf{$q$ snap} & '
            r'\textbf{cost v1} & \textbf{cost ctrl} & \textbf{cost v2} & \textbf{v2 vs v1} & '
            r'\textbf{v2 vs ctrl} & \textbf{v2 vs CAPGD} \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            e = v2[key]['models'][m]
            a, mc = e['attacks'], e['mcnemar']

            def cell(k):
                name = f'{V2} vs {k}'
                if name not in mc:
                    return '--'
                r = mc[name]
                return f"{r['v2_only']}/{r['other_only']}, {_sci(r['p'])}"
            body.append(f"{label if i == 0 else ''} & {ml} & {a[CTRL]['avg_queries']:.1f} & "
                        f"{a[V2S]['avg_queries']:.1f} & {_fmt3(a[V1]['median_op_cost_real'])} & "
                        f"{_fmt3(a[CTRL]['median_op_cost_real'])} & {_fmt3(a[V2]['median_op_cost_real'])} & "
                        f"{cell(V1)} & {cell(CTRL)} & {cell(CAPGD)} \\\\")
    return _tab('llrrrrrccc', head, body)


def paired_cost_table(v2: Dict[str, Dict]) -> str:
    head = (r'\textbf{Corpus} & \textbf{Target} & \textbf{v2 / v1 (flows both evade)} & '
            r'\textbf{v2 / control (flows both evade)} \\')
    body = []

    def paired(a, b):
        both = _real(a) & _real(b)
        if not both.any():
            return '--'
        ca = np.array([o['op_cost'] for o in a['outcomes']])[both]
        cb = np.array([o['op_cost'] for o in b['outcomes']])[both]
        return f"{np.median(ca):.3g} / {np.median(cb):.3g} ($n$={int(both.sum())})"
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            a = v2[key]['models'][m]['attacks']
            ctrl = paired(a[V2], a[CTRL]) if 'outcomes' in a[CTRL] else '--'
            body.append(f"{label if i == 0 else ''} & {ml} & {paired(a[V2], a[V1])} & {ctrl} \\\\")
    return _tab('llcc', head, body)


def usage_table(v2: Dict[str, Dict]) -> str:
    gens = ['time', 'pad', 'chaff', 'seg', 'win']
    head = r'\textbf{Corpus} & \textbf{Target} & ' + ' & '.join(r'\textbf{' + g + '}' for g in gens) + r' \\'
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            oc = [o for o in v2[key]['models'][m]['attacks'][V2]['outcomes']
                  if o['success'] and o['valid'] and o['path']]
            cells = [f"{100 * np.mean([g in o['path'] for o in oc]):.0f}" if oc else '--' for g in gens]
            body.append(f"{label if i == 0 else ''} & {ml} & " + ' & '.join(cells) + r' \\')
    return _tab('llccccc', head, body)


def family_table(v2: Dict[str, Dict]) -> str:
    head = (r'\textbf{Corpus} & \textbf{Family} & \textbf{XGBoost} & \textbf{LightGBM} & \textbf{CatBoost} \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        fams = {}
        for m, _ in LIBS:
            e = v2[key]['models'][m]
            fam = np.array(e['cohort']['family'])
            fl = _real(e['attacks'][V2])
            for f in set(fam):
                fams.setdefault(f, {})[m] = (100 * fl[fam == f].mean(), int((fam == f).sum()))
        order = sorted(fams, key=lambda f: (-fams[f]['xgboost'][1], f))
        for i, f in enumerate(order):
            cells = [f"{fams[f][m][0]:.1f} ({fams[f][m][1]})" if m in fams[f] else '--' for m, _ in LIBS]
            body.append(f"{label if i == 0 else ''} & {f} & " + ' & '.join(cells) + r' \\')
    return _tab('llccc', head, body)


def tuning_table() -> str:
    """Validation-split tuning table (rows: configurations; columns: 9 cells, mean, queries, cost)."""
    rows = load('ffev2_tuning.json')
    labels: List[str] = []
    for r in rows:
        if r['label'] not in labels:
            labels.append(r['label'])
    cells = [(d, m) for d, _ in DATASETS for m, _ in LIBS]
    short = {'cse2018': 'C18', 'cic2017': 'C17', 'unsw': 'UN'}
    ms = {'xgboost': 'X', 'lightgbm': 'L', 'catboost': 'C'}
    head = (r'\textbf{Configuration} & ' + ' & '.join(f'{short[d]}-{ms[m]}' for d, m in cells)
            + r' & \textbf{Mean} & \textbf{$q$} & \textbf{Cost} \\')
    body = []
    for lab in labels:
        by = {(r['dataset'], r['model']): r for r in rows if r['label'] == lab}
        if len(by) < 9:
            continue
        asr = [by[c]['real_asr'] for c in cells]
        cost = np.exp(np.mean([np.log(max(by[c]['median_op_cost_real'], 1e-12)) for c in cells]))
        name = lab.replace('_', r'\_').replace('=', '{=}').replace('&', r'\&')
        if lab in ('FFE-v2 frozen (score-only)', 'FFE-v1'):
            body.append(r'\midrule')
        body.append(f"{name} & " + ' & '.join(f'{v:.1f}' for v in asr)
                    + f" & {np.mean(asr):.1f} & {np.mean([by[c]['avg_queries'] for c in cells]):.1f} & {cost:.1e} \\\\")
    return _tab('l' + 'c' * 9 + 'ccc', head, body)


def replay_table() -> str:
    rep = load('ffev2_replay.json')
    head = (r'\textbf{Corpus} & \textbf{Flows} & \textbf{Evasions} & \textbf{Schema} & '
            r'\textbf{Empty, as padding} & \textbf{Empty, dropped} & \textbf{Segmenting flows} \\')
    body = []
    for key, label in DATASETS:
        r = rep[key]
        md = r['modes']
        body.append(f"{label} & {r['n_flows']} & {r['n_successful']} & {md['schema']['evasion_retained_pct']:.1f} & "
                    f"{md['empty_compensated']['evasion_retained_pct']:.1f} & "
                    f"{md['empty_literal']['evasion_retained_pct']:.1f} & {md['schema']['flows_using_segmentation']} \\\\")
    return _tab('lcccccc', head, body)


def whitebox_table() -> str:
    head = (r'\textbf{Corpus} & \textbf{Project at end} & \textbf{Project in loop} & \textbf{FFE-v1} & '
            r'\textbf{$p$ (in loop vs v1)} \\')
    body = []
    for key, label in DATASETS:
        w = load(f'whitebox_{key}.json')
        at = w['models']['xgboost']['attacks']
        end, loop = at['Leaf-tuple-style white-box (project at end)'], at['Leaf-tuple-style white-box (project in loop)']
        p = w['mcnemar_vs_ffe']['Leaf-tuple-style white-box (project in loop)']['p']
        v1 = w['mcnemar_vs_ffe']['Leaf-tuple-style white-box (project in loop)']['ffe_real_asr_same_flows']
        body.append(f"{label} & {end['real_asr']:.1f} & {loop['real_asr']:.1f} & {v1:.1f} & {_sci(p)} \\\\")
    return _tab('lcccc', head, body)


# ------------------------------------------------------------------------------------------
# defenses
# ------------------------------------------------------------------------------------------
def _dcell(dv: Dict, lib: str, model: str) -> Dict:
    return dv['models'][lib]['models'][model]


def defense_main_table(dv: Dict[str, Dict]) -> str:
    head = (r'\textbf{Corpus} & \textbf{Library} & \textbf{Acc.\ U/V2} & \textbf{Rec.\ U/V2} & '
            r'\multicolumn{3}{c}{\textbf{Undefended}} & \multicolumn{3}{c}{\textbf{LGAT}} & '
            r'\multicolumn{3}{c}{\textbf{V2}} \\' + '\n'
            r'& & & & FFE & CAPGD & WB & FFE & CAPGD & WB & FFE & CAPGD & WB \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            u, g, v = _dcell(dv[key], m, 'Undefended'), _dcell(dv[key], m, 'LGAT'), _dcell(dv[key], m, 'V2')

            def trio(x, bold=False):
                vals = [f"{x['attacks'][k]['real_asr']:.1f}" for k in ('ffe_m20', 'capgd', 'whitebox')]
                return ' & '.join((r'\textbf{' + t + '}') if bold else t for t in vals)
            body.append(f"{label if i == 0 else ''} & {ml} & {u['clean']['accuracy']:.2f}/{v['clean']['accuracy']:.2f} & "
                        f"{u['clean']['recall']:.1f}/{v['clean']['recall']:.1f} & {trio(u)} & {trio(g)} & "
                        f"{trio(v, True)} \\\\")
    return _tab('llcc' + 'c' * 9, head, body)


def defense_full_tables(dv: Dict[str, Dict]) -> Dict[str, str]:
    names = ['Undefended', 'MPS', 'LGAT', 'MPS+LGAT', 'Orbit', 'AdvTrain', 'Gate (certified)', 'V2', 'V2-R']
    head = (r'\textbf{Library} & \textbf{Model} & \textbf{Acc.} & \textbf{F1} & \textbf{Rec.} & '
            r'\textbf{FFE5} & \textbf{FFE20} & \textbf{CAPGD} & \textbf{WB} & \textbf{Kept} & \textbf{Ret.} \\')
    out = {}
    for key, _ in DATASETS:
        body = []
        for m, ml in LIBS:
            body.append(r'\midrule')
            for i, n in enumerate(names):
                c = _dcell(dv[key], m, n)
                at = c['attacks']
                ret = min(at[k]['retained_detection'] for k in at)
                body.append(f"{ml if i == 0 else ''} & {n} & {c['clean']['accuracy']:.2f} & {c['clean']['f1']:.1f} & "
                            f"{c['clean']['recall']:.1f} & "
                            + ' & '.join(f"{at[k]['real_asr']:.1f}" for k in ('ffe_m5', 'ffe_m20', 'capgd', 'whitebox'))
                            + f" & {c['cohort_detected']} & {ret:.1f} \\\\")
        out[f'tab_defense_full_{key}.tex'] = _tab('llcccccccrr', head, body)
    return out


def defense_gate_table(dv: Dict[str, Dict]) -> str:
    head = (r'\textbf{Corpus} & \textbf{Library} & \multicolumn{3}{c}{\textbf{Gate minus undefended}} & '
            r'\textbf{Gate kept} & \textbf{V2 kept} \\' + '\n'
            r'& & Acc. & F1 & Rec. & & \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            u, g, v = _dcell(dv[key], m, 'Undefended'), _dcell(dv[key], m, 'Gate (certified)'), _dcell(dv[key], m, 'V2')
            d = lambda k: g['clean'][k] - u['clean'][k]
            body.append(f"{label if i == 0 else ''} & {ml} & {d('accuracy'):+.2f} & {d('f1'):+.1f} & {d('recall'):+.1f} & "
                        f"{g['cohort_detected']} & {v['cohort_detected']} \\\\")
    return _tab('llcccrr', head, body)


def v2_tables() -> Dict[str, str]:
    v2 = {k: load(f'ffev2_{k}.json') for k, _ in DATASETS}
    dv = {k: load(f'defense_v2_{k}.json') for k, _ in DATASETS}
    out = {'tab_crosslib.tex': crosslib_table(v2), 'tab_ffev2_detail.tex': detail_table(v2),
           'tab_ffev2_paired.tex': paired_cost_table(v2), 'tab_ffev2_usage.tex': usage_table(v2),
           'tab_ffev2_family.tex': family_table(v2), 'tab_ffev2_tuning.tex': tuning_table(),
           'tab_replay.tex': replay_table(), 'tab_whitebox.tex': whitebox_table(),
           'tab_defense.tex': defense_main_table(dv), 'tab_defense_gate.tex': defense_gate_table(dv),
           'tab_defense_sel.tex': defense_selection_table(dv)}
    out.update(seeds_tables())
    out.update(defense_full_tables(dv))
    return out


def pairwise_v2(full: Dict[str, Dict]) -> Dict:
    """Exact McNemar of FFE-v2 (score-only) against the seven baselines of the benchmark, Holm-adjusted."""
    out, flat = {}, []
    skip = {'Flow-Field Evasion (FFE)', 'FFE without TSBS'}
    for key, _ in DATASETS:
        v2 = load(f'ffev2_{key}.json')['models']['xgboost']['attacks'][V2]
        f_all = _real(v2)
        atk = full[key]['models']['xgboost']['attacks']
        out[key] = {}
        for name, a in atk.items():
            if name in skip:
                continue
            n = a['n']
            f, b = f_all[:n], _real(a)[:n]
            only_f, only_b = int(np.sum(f & ~b)), int(np.sum(~f & b))
            p = 1.0 if only_f + only_b == 0 else binomtest(only_f, only_f + only_b, 0.5).pvalue
            out[key][name] = {'n': n, 'v2_only': only_f, 'baseline_only': only_b, 'p': float(p),
                              'v2_real_asr_same_flows': 100 * float(f.mean()),
                              'baseline_real_asr': 100 * float(b.mean())}
            flat.append((key, name, p))
    order = np.argsort([p for _, _, p in flat])
    m, running = len(flat), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * flat[i][2])
        out[flat[i][0]][flat[i][1]]['p_holm'] = float(min(1.0, running))
    return out


def defense_selection_table(dv: Dict[str, Dict]) -> str:
    """Components chosen on the validation split for V2 and V2-R (free classifier, gate threshold)."""
    head = (r'\textbf{Corpus} & \textbf{Library} & \textbf{V2 free} & \textbf{$\tau$} & \textbf{val.\ FFE-v1} & '
            r'\textbf{val.\ WB} & \textbf{V2-R free} & \textbf{$\tau$} \\')
    body = []
    for key, label in DATASETS:
        body.append(r'\midrule')
        for i, (m, ml) in enumerate(LIBS):
            v = dv[key]['models'][m]['validation']
            s, r = v['selected'], v['selected_relaxed']
            body.append(f"{label if i == 0 else ''} & {ml} & {s['free'].replace('_', ' ')} & {s['tau']} & "
                        f"{s['val_asr']['ffe_m20']:.1f} & {s['val_asr']['whitebox']:.1f} & "
                        f"{r['free'].replace('_', ' ')} & {r['tau']} \\\\")
    return _tab('llcccccc', head, body)
