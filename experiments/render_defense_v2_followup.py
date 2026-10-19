"""Render results/defense_v2_followup.md from results/defense_v2_followup_<dataset>.json."""

import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
DATASETS = (('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15'))
ATT = (('ffe_m20', 'FFE M=20'), ('capgd', 'CAPGD'), ('whitebox', 'White-box'))
LIBS = ('xgboost', 'lightgbm', 'catboost')

HEAD = """# Defense v2 follow-up checks

Produced by `experiments/run_defenses_v2_followup.py` (+ `render_defense_v2_followup.py`); data in
`results/defense_v2_followup_<dataset>.json`. Models are not persisted, so every model is retrained with the same seeds; the retrained
undefended model reproduces the saved cohort and clean accuracy exactly in all cells (`retrained_matches_saved`). Attack outcomes of
Undefended, MPS, LGAT, V2 and V2-R are the saved ones from `results/defense_v2_<dataset>.json` (same cohort); G35, V2 with G35 and
the unconstrained control were attacked in this run with the same attacks (FFE M=20, CAPGD, white-box).

## Definitions and selection rules

* **Whole-cohort evasion** = 100 x (1 - flagged after attack / cohort), cohort = the 1,000 malicious test flows detected by the undefended
  model of the same library. A flow is "flagged after attack" if the model flags the original and the attack does not produce a real
  (schema-feasible) evasion; flows the defended model already misses count as evaded. Real ASR (conditional on detection) is given next to it.
* **G35**: the gate of Section 2 of `defense_v2_report.md` with the backward columns 3 (tot_bwd_pkts) and 5 (totlen_bwd_pkts) hidden as well
  (hidden columns 3, 5, 7, 8, 9, 10; monotone +1 on 2, 4, 6, 11; same capacity). **V2 with G35** uses the same selection rule as V2.
* **Selection rule of V2 (as run, `tune_on_validation` in `experiments/run_defenses_v2.py`)**: grid over free classifier in
  {undefended, orbit, orbit_light, adversarial} x gate threshold tau in {none, 0.5, 0.8, 0.95, 0.99} (undefended without gate excluded).
  Keep the candidates whose VALIDATION accuracy, F1 and recall are each within 0.5 point of the undefended model of the same library; among
  them take the lowest mean validation Real ASR of FFE M=20 and the white-box attacker on a 300-flow validation cohort (ties: higher F1).
* **V2-R**: same grid, but the filter is accuracy and F1 within 3.0 points (recall unrestricted), then the same minimum. V2-R was
  defined while building the pipeline, after a smoke run on UNSW-NB15/LightGBM (100-flow test cohort, only the undefended model's FFE rates
  reached the log) had shown, on validation, that the 0.5-point filter leaves only weak candidates on that corpus; it was fixed in code
  before any full test result existed. It is therefore a post-hoc second tier chosen after seeing validation tables of one cell, not a
  pre-registered one. V2 (strict) is the headline defense; V2-R equals V2 in all CSE-CIC-IDS2018 and CIC-IDS2017 cells.
* **G35 selection** (this follow-up): free classifier in {undefended, orbit_light} only (no adversarial or full orbit arm), strict tier rule.

"""


def cell(a: dict) -> str:
    return f"{a['real_asr']:.1f} / {a['whole_cohort_evasion']:.1f}"


def selection_table(data: dict) -> str:
    lines = ['| corpus | library | V2 free + tau | V2-R free + tau | V2 with G35 free + tau | V2 uses undefended free classifier |',
             '|---|---|---|---|---|---|']
    for ds, label in DATASETS:
        d = data.get(ds)
        if not d:
            continue
        for lib in LIBS:
            if lib not in d['models']:
                continue
            s = d['models'][lib]['v2_selection']
            f = lambda k: f"{s[k]['free']} + {s[k]['tau']}"
            lines.append(f"| {label} | {lib} | {f('strict')} | {f('relaxed')} | {f('g35')} | "
                         f"{'yes' if s['strict']['free'] == 'undefended' else 'no'} |")
    return '\n'.join(lines)


def clean_table(res: dict) -> str:
    lines = ['| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |',
             '|---|---|---|---|---|---|---|---|']
    for name, m in res['models'].items():
        c = m['clean']
        lines.append(f"| {name} | {c['accuracy']:.2f} | {c['f1']:.2f} | {c['recall']:.2f} | {m['benign_fpr']:.3f} | "
                     f"{m['benign_fpr_top_decile_fwd_pkts']:.3f} | {m['benign_fpr_rest']:.3f} | {m['cohort_flagged_clean']:.1f} |")
    return '\n'.join(lines)


def attack_table(res: dict) -> str:
    lines = ['| model | ' + ' | '.join(f'{l} (Real ASR / whole-cohort)' for _, l in ATT) + ' |', '|---|---|---|---|']
    for name, m in res['models'].items():
        lines.append(f'| {name} | ' + ' | '.join(cell(m['attacks'][a]) for a, _ in ATT) + ' |')
    return '\n'.join(lines)


def ffev2_table(res: dict) -> str:
    f = res.get('ffev2')
    if not f:
        return ''
    lines = [f"FFE-v2 (score + snapping configuration of `results/ffev2_config.json`) on the first {f['cohort_used']} cohort flows:", '',
             '| model | Real ASR % [Wilson] | whole-cohort evasion % |', '|---|---|---|']
    for name, v in f.items():
        if isinstance(v, dict) and 'real_asr' in v:
            lines.append(f"| {name} | {v['real_asr']} {v['real_asr_ci']} | {v['whole_cohort_evasion']} |")
    return '\n'.join(lines)


def main() -> None:
    data = {}
    for ds, _ in DATASETS:
        path = os.path.join(RESULTS, f'defense_v2_followup_{ds}.json')
        if os.path.exists(path):
            data[ds] = json.load(open(path))
    out = [HEAD, '## Selections\n', selection_table(data), '\n## Results\n',
           'Capacity control row: "Unconstrained 300x8 (gate columns hidden)" has the gate\'s capacity and hidden columns but no monotone '
           'constraint; its difference to "Gate G" is the cost of monotonicity, its difference to "Undefended" (60x6, all columns) the cost of '
           'hiding columns plus the capacity change.\n']
    for ds, label in DATASETS:
        for lib in LIBS:
            res = data.get(ds, {}).get('models', {}).get(lib)
            if not res:
                out.append(f'### {label}, {lib}\n\nNot run.\n')
                continue
            out += [f'### {label}, {lib}\n', clean_table(res), '', attack_table(res), '', ffev2_table(res), '']
    open(os.path.join(RESULTS, 'defense_v2_followup.md'), 'w', encoding='utf-8').write('\n'.join(out))


if __name__ == '__main__':
    main()
