"""Render markdown tables (validation tuning, test results, McNemar) from results/defense_v2_<dataset>.json.

Usage: python experiments/render_defense_v2_tables.py > tables.md
"""

import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
DATASETS = (('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15'))
ATTACK_LABELS = {'ffe_m5': 'FFE M=5', 'ffe_m20': 'FFE M=20', 'capgd': 'CAPGD', 'whitebox': 'White-box'}


def load(ds: str):
    path = os.path.join(RESULTS, f'defense_v2_{ds}.json')
    return json.load(open(path)) if os.path.exists(path) else None


def cell(a: dict) -> str:
    lo, hi = a['real_asr_ci']
    return f"{a['real_asr']:.1f} [{lo:.0f}-{hi:.0f}]"


def validation_table(lib: str, v: dict) -> str:
    ref = v['reference_clean']
    lines = [f"Reference (undefended) validation: acc {ref['accuracy']}, F1 {ref['f1']}, recall {ref['recall']}.", '',
             '| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |',
             '|---|---|---|---|---|---|---|---|---|']
    sel = v['selected']
    for r in v['table']:
        asr = r.get('val_asr', {})
        mark = 'yes' if (r['free'], r['tau']) == (sel['free'], sel['tau']) else ''
        lines.append(f"| {r['free']} | {r['tau']} | {r['clean']['accuracy']} | {r['clean']['f1']} | {r['clean']['recall']} | "
                     f"{'ok' if r['parity'] else 'no'} | {asr.get('ffe_m20', '-')} | {asr.get('whitebox', '-')} | {mark} |")
    return '\n'.join(lines)


def test_table(lib: str, res: dict, attacks) -> str:
    head = ('| model | acc | F1 | recall | cohort kept | ' + ' | '.join(ATTACK_LABELS.get(a, a) for a in attacks)
            + ' | retained (min over attacks) | build s |')
    lines = [head, '|' + '---|' * (6 + len(attacks) + 1)]
    for name, m in res['models'].items():
        c = m['clean']
        asr = ' | '.join(cell(m['attacks'][a]) for a in attacks)
        kept = min(m['attacks'][a]['retained_detection'] for a in attacks)
        lines.append(f"| {name} | {c['accuracy']:.2f} | {c['f1']:.2f} | {c['recall']:.2f} | {m['cohort_detected']}/{res['cohort_size']} "
                     f"| {asr} | {kept:.1f} | {m['build_seconds']:.0f} |")
    return '\n'.join(lines)


def mcnemar_table(res: dict, attacks) -> str:
    lines = ['| comparison | ' + ' | '.join(ATTACK_LABELS.get(a, a) for a in attacks) + ' |', '|' + '---|' * (1 + len(attacks))]
    for key in res['mcnemar_v2'][attacks[0]]:
        cells = []
        for a in attacks:
            t = res['mcnemar_v2'][a][key]
            cells.append(f"{t['only_second']}/{t['only_first']} p={t['p']:.2g}")
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> None:
    for ds, label in DATASETS:
        data = load(ds)
        if not data:
            continue
        print(f'\n## {label}\n')
        for lib, res in data['models'].items():
            print(f'### {label}, {lib}\n\nValidation tuning (selection frozen before test):\n\n{validation_table(lib, res["validation"])}\n')
            print(f'Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort '
                  f'still detected after the attack):\n\n{test_table(lib, res, data["attacks"])}\n')
            print('Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the '
                  'other model / only against V2):\n\n' + mcnemar_table(res, data['attacks']) + '\n')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
