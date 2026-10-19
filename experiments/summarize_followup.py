"""
LaTeX tables for the defense follow-up checks (results/defense_v2_followup_<dataset>.json):
whole-cohort evasion, hidden-backward-column gate (G35), capacity control, benign FPR.
"""

import json
import os
from typing import Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
DATASETS = [('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15')]
LIBS = [('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM'), ('catboost', 'CatBoost')]
ATTACKS = ('ffe_m20', 'capgd', 'whitebox')
UNC = 'Unconstrained 300x8 (gate columns hidden)'
EOL = chr(92) * 2


def load_followup() -> Dict[str, Dict]:
    out = {}
    for key, _ in DATASETS:
        with open(os.path.join(RES, f'defense_v2_followup_{key}.json'), encoding='utf-8') as fh:
            out[key] = json.load(fh)
    return out


def _tab(spec: str, header: str, body) -> str:
    mid = [] if body and body[0].startswith(chr(92) + 'midrule') else [chr(92) + 'midrule']
    b = chr(92)
    return '\n'.join([b + 'begin{tabular}{@{}' + spec + '@{}}', b + 'toprule', header] + mid + body
                     + [b + 'bottomrule', b + 'end{tabular}'])


def whole_max(fu: Dict, key: str, lib: str, model: str) -> float:
    """Largest whole-cohort evasion (%) over FFE-v1 M=20, CAPGD and the white-box attacker."""
    at = fu[key]['models'][lib]['models'][model]['attacks']
    return max(at[a]['whole_cohort_evasion'] for a in ATTACKS)


def clean_table(fu: Dict[str, Dict]) -> str:
    b = chr(92)
    head = (b + 'textbf{Corpus} & ' + b + 'textbf{Library} & ' + ' & '.join(b + 'textbf{' + h + '}' for h in
            ('Acc.\\ U', 'G', 'G35', 'Unc.', 'F1 U', 'G', 'G35', 'Unc.', 'FPR U', 'V2', 'G35')) + ' ' + EOL)
    body = []
    for key, label in DATASETS:
        body.append(b + 'midrule')
        for i, (m, ml) in enumerate(LIBS):
            md = fu[key]['models'][m]['models']
            u, g, g35, unc, v2 = (md['Undefended'], md['Gate G'], md['Gate G35'], md[UNC], md['V2'])
            cells = [f"{x['clean']['accuracy']:.2f}" for x in (u, g, g35, unc)]
            cells += [f"{x['clean']['f1']:.1f}" for x in (u, g, g35, unc)]
            cells += [f"{x['benign_fpr']:.2f}" for x in (u, v2, g35)]
            body.append(f"{label if i == 0 else ''} & {ml} & " + ' & '.join(cells) + ' ' + EOL)
    return _tab('ll' + 'c' * 11, head, body)


def whole_table(fu: Dict[str, Dict]) -> str:
    b = chr(92)
    cols = []
    for g in ('V2', 'V2 with G35', 'Unc.'):
        cols += [b + 'multicolumn{3}{c}{' + b + 'textbf{' + g + '}}']
    head = (b + 'textbf{Corpus} & ' + b + 'textbf{Library} & ' + ' & '.join(cols) + ' ' + EOL + '\n'
            + '& & ' + ' & '.join(['FFE', 'CAPGD', 'WB'] * 3) + ' ' + EOL)
    body = []
    for key, label in DATASETS:
        body.append(b + 'midrule')
        for i, (m, ml) in enumerate(LIBS):
            md = fu[key]['models'][m]['models']
            cells = []
            for name in ('V2', 'V2 with G35', UNC):
                cells += [f"{md[name]['attacks'][a]['whole_cohort_evasion']:.1f}" for a in ATTACKS]
            body.append(f"{label if i == 0 else ''} & {ml} & " + ' & '.join(cells) + ' ' + EOL)
    return _tab('ll' + 'c' * 9, head, body)


def followup_tables() -> Dict[str, str]:
    fu = load_followup()
    return {'tab_followup_clean.tex': clean_table(fu), 'tab_followup_whole.tex': whole_table(fu)}


def main_defense_table(dv: Dict[str, Dict], fu: Dict[str, Dict]) -> str:
    """Main defense table: conditional Real ASR per attacker plus V2 whole-cohort evasion (max over attackers)."""
    b = chr(92)
    head = (b + 'textbf{Corpus} & ' + b + 'textbf{Library} & ' + b + 'textbf{Acc.\ U/V2} & ' + b + 'textbf{Rec.\ U/V2} & '
            + b + 'multicolumn{3}{c}{' + b + 'textbf{Undefended}} & ' + b + 'multicolumn{3}{c}{' + b + 'textbf{LGAT}} & '
            + b + 'multicolumn{3}{c}{' + b + 'textbf{V2}} & ' + b + 'textbf{V2 whole} ' + EOL + '\n'
            + '& & & & FFE & CAPGD & WB & FFE & CAPGD & WB & FFE & CAPGD & WB & cohort ' + EOL)
    body = []
    for key, label in DATASETS:
        body.append(b + 'midrule')
        for i, (m, ml) in enumerate(LIBS):
            md = dv[key]['models'][m]['models']
            u, g, v = md['Undefended'], md['LGAT'], md['V2']

            def trio(x, bold=False):
                vals = [f"{x['attacks'][k]['real_asr']:.1f}" for k in ('ffe_m20', 'capgd', 'whitebox')]
                return ' & '.join((b + 'textbf{' + t + '}') if bold else t for t in vals)
            body.append(f"{label if i == 0 else ''} & {ml} & {u['clean']['accuracy']:.2f}/{v['clean']['accuracy']:.2f} & "
                        f"{u['clean']['recall']:.1f}/{v['clean']['recall']:.1f} & {trio(u)} & {trio(g)} & "
                        f"{trio(v, True)} & {whole_max(fu, key, m, 'V2'):.1f} " + EOL)
    return _tab('llcc' + 'c' * 10, head, body)
