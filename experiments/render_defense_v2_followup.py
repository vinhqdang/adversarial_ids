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

TAIL = """
## Findings

* **G35 (backward columns hidden) is not a free tightening.** It costs much more than the gate itself: clean accuracy / F1 drop by 3.9 / 19 points on CSE-CIC-IDS2018, 4.7-4.9 / 9 points on CIC-IDS2017 and 12-14 accuracy points on UNSW-NB15 (all three libraries), the benign false-positive rate rises to 2.1 % (CSE-CIC-IDS2018), 6 % (CIC-IDS2017) and 36 % (UNSW-NB15), and the top-decile-fwd_pkts FPR to 5 / 14-16 / 44 %. It still flags 74-76 % (CSE-CIC-IDS2018), 97-99 % (CIC-IDS2017) and 89-90 % (UNSW-NB15) of the cohort and is certified under the weaker assumption (backward traffic may change). V2 rebuilt with G35 under the same parity rule is clearly weaker than V2 where the gate matters (CSE-CIC-IDS2018 whole-cohort evasion 35.1-36.3 % FFE M=20, 23-70 % CAPGD, 78-79 % white-box, versus 0.2-8.3 %, 0.2-0.5 %, 7.9-9.0 % for V2; UNSW-NB15 XGBoost 20 / 6 / 39 % versus 12 / 2 / 26 %), because the parity rule forces a high tau. On CIC-IDS2017 it stays at 1-6 %. So V2's guarantee relies on the assumption that backward packets and bytes do not change; if that assumption is dropped, the hardened model keeps only a part of its strength at a large accuracy price on these three corpora.
* **Capacity control.** The unconstrained 300x8 model with the gate's hidden columns is as accurate as, or more accurate than, the undefended 60x6 model (CIC-IDS2017 +0.03, UNSW-NB15 +0.7 to +0.8 accuracy, CSE-CIC-IDS2018 -0.14 accuracy / -0.6 F1) and is as evadable as it (whole-cohort evasion 17-96 %, CIC-IDS2017 white-box 34-48 %). Hence the stand-alone gate's accuracy cost is the monotonicity: against the unconstrained control it is -0.55 accuracy / -3.1 F1 (CSE-CIC-IDS2018), -2.6 to -2.7 accuracy (UNSW-NB15 XGBoost/LightGBM; -2.8 CatBoost) and -0.06 (CIC-IDS2017). Capacity alone buys nothing against the attackers.
* **Whole-cohort numbers** change the picture for the gate-only and V2 rows by the missed share: V2 whole-cohort evasion is 0.2-9.0 % on CSE-CIC-IDS2018, 0.1-0.8 % on CIC-IDS2017 and 1.0-32.4 % on UNSW-NB15 (versus conditional Real ASR within 0.5-1 point of it, since V2 keeps 98-100 % of the cohort flagged before the attack). The stand-alone gate's whole-cohort evasion is its miss rate (7.9-9.0 % on CSE-CIC-IDS2018, 0.1-0.2 % CIC-IDS2017, 4.5-6.0 % UNSW-NB15).
* **False positives and monotone drift.** Benign FPR on the test split: CSE-CIC-IDS2018 0.03-0.07 % undefended versus 0.11-0.14 % for V2; CIC-IDS2017 0.01-0.02 % versus 0.02-0.05 %; UNSW-NB15 11-16 % versus 13-19 % (XGBoost 11.1 -> 13.3 %, LightGBM 11.1 -> 13.9 %, CatBoost 15.7 -> 18.6 %). On the top decile of benign forward-packet counts the FPR is 0.01-0.18 % (CSE-CIC-IDS2018), 0.01-0.03 % (CIC-IDS2017) and 2.2-5.1 % (UNSW-NB15) for V2, close to the undefended model on UNSW-NB15 (2.4 / 2.6 / 4.4 %) and not elevated relative to the rest; the stand-alone gate (rows "Gate G") shows no large-flow drift either (UNSW-NB15 top-decile FPR 2.9 / 2.1 / 2.4 % versus 4.4 / 2.6 / 2.4 % undefended). G35 is the exception (above). Large benign flows are rare in these 12-feature corpora, so this is weak evidence about drift in other traffic mixes.
* **FFE-v2 (CIC-IDS2017, XGBoost, first 300 cohort flows, retrained models, configuration of `results/ffev2_config.json`, "score + snapping")**: undefended 100.0 % Real ASR; V2 1.0 % (whole-cohort 1.3 %); V2 with G35 8.4 % (8.7 %); stand-alone G35 gate 0.0 % (whole-cohort 2.3 %, the miss rate). FFE-v2 was run only here, not on the other corpora or libraries.
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
    out.append(TAIL)
    open(os.path.join(RESULTS, 'defense_v2_followup.md'), 'w', encoding='utf-8').write('\n'.join(out))


if __name__ == '__main__':
    main()
