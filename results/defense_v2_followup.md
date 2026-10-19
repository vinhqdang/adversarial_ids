# Defense v2 follow-up checks

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


## Selections

| corpus | library | V2 free + tau | V2-R free + tau | V2 with G35 free + tau | V2 uses undefended free classifier |
|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | orbit_light + 0.5 | orbit_light + 0.5 | orbit_light + 0.95 | no |
| CSE-CIC-IDS2018 | lightgbm | undefended + 0.5 | undefended + 0.5 | orbit_light + 0.8 | yes |
| CSE-CIC-IDS2018 | catboost | orbit_light + 0.5 | orbit_light + 0.5 | orbit_light + 0.8 | no |
| CIC-IDS2017 | xgboost | orbit_light + 0.95 | orbit_light + 0.95 | orbit_light + 0.8 | no |
| CIC-IDS2017 | lightgbm | undefended + 0.8 | undefended + 0.8 | orbit_light + 0.8 | yes |
| CIC-IDS2017 | catboost | undefended + 0.5 | undefended + 0.5 | orbit_light + 0.8 | yes |
| UNSW-NB15 | xgboost | orbit_light + 0.8 | adversarial + 0.5 | orbit_light + 0.95 | no |
| UNSW-NB15 | lightgbm | orbit_light + 0.8 | adversarial + 0.8 | undefended + 0.8 | no |
| UNSW-NB15 | catboost | undefended + 0.5 | adversarial + 0.5 | undefended + 0.8 | yes |

## Results

Capacity control row: "Unconstrained 300x8 (gate columns hidden)" has the gate's capacity and hidden columns but no monotone constraint; its difference to "Gate G" is the cost of monotonicity, its difference to "Undefended" (60x6, all columns) the cost of hiding columns plus the capacity change.

### CSE-CIC-IDS2018, xgboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 97.30 | 87.31 | 77.62 | 0.025 | 0.011 | 0.027 | 100.0 |
| MPS | 97.30 | 87.30 | 77.61 | 0.026 | 0.013 | 0.028 | 99.9 |
| LGAT | 97.28 | 87.26 | 77.83 | 0.075 | 0.013 | 0.083 | 99.8 |
| V2 | 97.29 | 87.40 | 78.40 | 0.137 | 0.022 | 0.151 | 99.8 |
| V2-R | 97.29 | 87.40 | 78.40 | 0.137 | 0.022 | 0.151 | 99.8 |
| Gate G | 96.60 | 83.56 | 72.20 | 0.083 | 0.014 | 0.092 | 91.7 |
| Gate G35 | 93.20 | 67.36 | 58.58 | 2.086 | 5.177 | 1.697 | 73.5 |
| V2 with G35 | 97.28 | 87.27 | 77.80 | 0.068 | 0.015 | 0.075 | 99.8 |
| Unconstrained 300x8 (gate columns hidden) | 97.14 | 86.66 | 77.50 | 0.183 | 0.006 | 0.205 | 98.8 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 99.5 / 99.5 | 99.0 / 99.0 | 99.5 / 99.5 |
| MPS | 61.9 / 61.9 | 98.3 / 98.3 | 99.5 / 99.5 |
| LGAT | 73.1 / 73.2 | 96.1 / 96.1 | 99.5 / 99.5 |
| V2 | 8.1 / 8.3 | 0.3 / 0.5 | 8.1 / 8.3 |
| V2-R | 8.1 / 8.3 | 0.3 / 0.5 | 8.1 / 8.3 |
| Gate G | 0.0 / 8.3 | 0.0 / 8.3 | 0.0 / 8.3 |
| Gate G35 | 0.0 / 26.5 | 0.0 / 26.5 | 0.0 / 26.5 |
| V2 with G35 | 36.2 / 36.3 | 50.3 / 50.4 | 79.0 / 79.0 |
| Unconstrained 300x8 (gate columns hidden) | 85.9 / 86.1 | 16.4 / 17.4 | 95.1 / 95.2 |



### CSE-CIC-IDS2018, lightgbm

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 97.28 | 87.28 | 77.83 | 0.069 | 0.012 | 0.077 | 100.0 |
| MPS | 97.28 | 87.26 | 77.65 | 0.045 | 0.038 | 0.045 | 99.8 |
| LGAT | 97.28 | 87.25 | 77.82 | 0.078 | 0.015 | 0.086 | 100.0 |
| V2 | 97.27 | 87.29 | 78.23 | 0.138 | 0.028 | 0.152 | 100.0 |
| V2-R | 97.27 | 87.29 | 78.23 | 0.138 | 0.028 | 0.152 | 100.0 |
| Gate G | 96.58 | 83.43 | 72.00 | 0.081 | 0.015 | 0.090 | 92.1 |
| Gate G35 | 93.39 | 68.55 | 60.18 | 2.094 | 5.141 | 1.710 | 75.7 |
| V2 with G35 | 97.28 | 87.28 | 77.88 | 0.080 | 0.020 | 0.088 | 100.0 |
| Unconstrained 300x8 (gate columns hidden) | 97.15 | 86.69 | 77.57 | 0.189 | 0.015 | 0.211 | 98.5 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 99.8 / 99.8 | 99.5 / 99.5 | 99.8 / 99.8 |
| MPS | 42.3 / 42.4 | 98.5 / 98.5 | 99.8 / 99.8 |
| LGAT | 56.8 / 56.8 | 96.3 / 96.3 | 99.8 / 99.8 |
| V2 | 7.9 / 7.9 | 0.2 / 0.2 | 7.9 / 7.9 |
| V2-R | 7.9 / 7.9 | 0.2 / 0.2 | 7.9 / 7.9 |
| Gate G | 0.0 / 7.9 | 0.0 / 7.9 | 0.0 / 7.9 |
| Gate G35 | 0.0 / 24.3 | 0.0 / 24.3 | 0.0 / 24.3 |
| V2 with G35 | 35.9 / 35.9 | 23.1 / 23.1 | 78.8 / 78.8 |
| Unconstrained 300x8 (gate columns hidden) | 88.9 / 89.1 | 26.2 / 27.3 | 96.3 / 96.4 |



### CSE-CIC-IDS2018, catboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 97.26 | 87.16 | 77.63 | 0.068 | 0.124 | 0.061 | 100.0 |
| MPS | 97.28 | 87.21 | 77.56 | 0.042 | 0.103 | 0.034 | 99.8 |
| LGAT | 97.25 | 87.11 | 77.52 | 0.063 | 0.133 | 0.054 | 99.8 |
| V2 | 97.25 | 87.16 | 77.89 | 0.114 | 0.180 | 0.106 | 99.8 |
| V2-R | 97.25 | 87.16 | 77.89 | 0.114 | 0.180 | 0.106 | 99.8 |
| Gate G | 96.58 | 83.43 | 71.87 | 0.058 | 0.009 | 0.064 | 91.0 |
| Gate G35 | 93.38 | 68.47 | 60.02 | 2.081 | 5.207 | 1.688 | 74.3 |
| V2 with G35 | 97.27 | 87.21 | 77.71 | 0.067 | 0.176 | 0.053 | 99.8 |
| Unconstrained 300x8 (gate columns hidden) | 97.14 | 86.61 | 77.22 | 0.148 | 0.005 | 0.166 | 98.7 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 59.8 / 59.8 | 99.3 / 99.3 | 100.0 / 100.0 |
| MPS | 39.8 / 39.9 | 99.2 / 99.2 | 100.0 / 100.0 |
| LGAT | 45.6 / 45.7 | 83.8 / 83.8 | 100.0 / 100.0 |
| V2 | 0.0 / 0.2 | 0.0 / 0.2 | 8.8 / 9.0 |
| V2-R | 0.0 / 0.2 | 0.0 / 0.2 | 8.8 / 9.0 |
| Gate G | 0.0 / 9.0 | 0.0 / 9.0 | 0.0 / 9.0 |
| Gate G35 | 0.0 / 25.7 | 0.0 / 25.7 | 0.0 / 25.7 |
| V2 with G35 | 35.0 / 35.1 | 69.5 / 69.6 | 77.7 / 77.7 |
| Unconstrained 300x8 (gate columns hidden) | 93.1 / 93.2 | 29.6 / 30.5 | 98.1 / 98.1 |



### CIC-IDS2017, xgboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 99.96 | 99.91 | 99.85 | 0.008 | 0.011 | 0.008 | 100.0 |
| MPS | 99.91 | 99.82 | 99.73 | 0.028 | 0.007 | 0.031 | 99.9 |
| LGAT | 99.96 | 99.92 | 99.87 | 0.011 | 0.015 | 0.011 | 100.0 |
| V2 | 99.97 | 99.93 | 99.94 | 0.023 | 0.022 | 0.023 | 99.9 |
| V2-R | 99.97 | 99.93 | 99.94 | 0.023 | 0.022 | 0.023 | 99.9 |
| Gate G | 99.93 | 99.86 | 99.85 | 0.038 | 0.019 | 0.040 | 99.8 |
| Gate G35 | 95.07 | 90.50 | 98.35 | 5.964 | 15.247 | 4.867 | 98.9 |
| V2 with G35 | 99.80 | 99.58 | 99.87 | 0.221 | 0.330 | 0.208 | 99.9 |
| Unconstrained 300x8 (gate columns hidden) | 99.99 | 99.99 | 99.99 | 0.006 | 0.011 | 0.005 | 100.0 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 99.1 / 99.1 | 77.0 / 77.0 | 100.0 / 100.0 |
| MPS | 92.8 / 92.8 | 47.7 / 47.8 | 98.1 / 98.1 |
| LGAT | 31.9 / 31.9 | 52.6 / 52.6 | 100.0 / 100.0 |
| V2 | 0.7 / 0.8 | 0.7 / 0.8 | 0.7 / 0.8 |
| V2-R | 0.7 / 0.8 | 0.7 / 0.8 | 0.7 / 0.8 |
| Gate G | 0.0 / 0.2 | 0.0 / 0.2 | 0.0 / 0.2 |
| Gate G35 | 0.0 / 1.1 | 0.0 / 1.1 | 0.0 / 1.1 |
| V2 with G35 | 1.2 / 1.3 | 1.5 / 1.6 | 5.4 / 5.5 |
| Unconstrained 300x8 (gate columns hidden) | 32.4 / 32.4 | 5.0 / 5.0 | 33.9 / 33.9 |

FFE-v2 (score + snapping configuration of `results/ffev2_config.json`) on the first 300 cohort flows:

| model | Real ASR % [Wilson] | whole-cohort evasion % |
|---|---|---|
| Undefended | 100.0 [98.7, 100.0] | 100.0 |
| V2 | 1.0 [0.3, 2.9] | 1.3 |
| V2 with G35 | 8.4 [5.7, 12.1] | 8.7 |
| Gate G35 | 0.0 [0.0, 1.3] | 2.3 |

### CIC-IDS2017, lightgbm

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 99.96 | 99.92 | 99.87 | 0.008 | 0.015 | 0.008 | 100.0 |
| MPS | 99.91 | 99.82 | 99.73 | 0.030 | 0.007 | 0.032 | 99.9 |
| LGAT | 99.96 | 99.92 | 99.87 | 0.010 | 0.019 | 0.008 | 100.0 |
| V2 | 99.97 | 99.93 | 99.94 | 0.027 | 0.015 | 0.029 | 100.0 |
| V2-R | 99.97 | 99.93 | 99.94 | 0.027 | 0.015 | 0.029 | 100.0 |
| Gate G | 99.91 | 99.81 | 99.76 | 0.040 | 0.007 | 0.044 | 99.9 |
| Gate G35 | 94.76 | 89.90 | 97.57 | 6.117 | 16.053 | 4.943 | 97.1 |
| V2 with G35 | 99.84 | 99.67 | 99.86 | 0.163 | 0.229 | 0.155 | 100.0 |
| Unconstrained 300x8 (gate columns hidden) | 99.99 | 99.99 | 99.98 | 0.004 | 0.007 | 0.004 | 100.0 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 34.6 / 34.6 | 57.9 / 57.9 | 35.7 / 35.7 |
| MPS | 34.1 / 34.2 | 31.6 / 31.7 | 35.4 / 35.5 |
| LGAT | 33.4 / 33.4 | 36.3 / 36.3 | 35.6 / 35.6 |
| V2 | 0.4 / 0.4 | 0.4 / 0.4 | 0.4 / 0.4 |
| V2-R | 0.4 / 0.4 | 0.4 / 0.4 | 0.4 / 0.4 |
| Gate G | 0.0 / 0.1 | 0.0 / 0.1 | 0.0 / 0.1 |
| Gate G35 | 0.0 / 2.9 | 0.0 / 2.9 | 0.0 / 2.9 |
| V2 with G35 | 2.0 / 2.0 | 2.0 / 2.0 | 6.3 / 6.3 |
| Unconstrained 300x8 (gate columns hidden) | 34.3 / 34.3 | 30.5 / 30.5 | 47.9 / 47.9 |



### CIC-IDS2017, catboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 99.85 | 99.68 | 99.41 | 0.017 | 0.007 | 0.018 | 100.0 |
| MPS | 99.82 | 99.63 | 99.38 | 0.040 | 0.007 | 0.044 | 99.7 |
| LGAT | 99.86 | 99.70 | 99.46 | 0.022 | 0.007 | 0.023 | 100.0 |
| V2 | 99.93 | 99.85 | 99.87 | 0.053 | 0.026 | 0.057 | 100.0 |
| V2-R | 99.93 | 99.85 | 99.87 | 0.053 | 0.026 | 0.057 | 100.0 |
| Gate G | 99.92 | 99.83 | 99.79 | 0.042 | 0.022 | 0.044 | 99.9 |
| Gate G35 | 95.16 | 90.68 | 98.51 | 5.888 | 14.273 | 4.897 | 98.7 |
| V2 with G35 | 99.74 | 99.46 | 99.50 | 0.181 | 0.262 | 0.171 | 100.0 |
| Unconstrained 300x8 (gate columns hidden) | 99.98 | 99.96 | 99.96 | 0.012 | 0.011 | 0.012 | 100.0 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 32.0 / 32.0 | 5.4 / 5.4 | 33.6 / 33.6 |
| MPS | 32.0 / 32.2 | 25.0 / 25.2 | 32.8 / 33.0 |
| LGAT | 31.2 / 31.2 | 31.0 / 31.0 | 33.2 / 33.2 |
| V2 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 |
| V2-R | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 |
| Gate G | 0.0 / 0.1 | 0.0 / 0.1 | 0.0 / 0.1 |
| Gate G35 | 0.0 / 1.3 | 0.0 / 1.3 | 0.0 / 1.3 |
| V2 with G35 | 1.1 / 1.1 | 2.8 / 2.8 | 3.1 / 3.1 |
| Unconstrained 300x8 (gate columns hidden) | 32.2 / 32.2 | 0.7 / 0.7 | 32.5 / 32.5 |



### UNSW-NB15, xgboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 88.51 | 87.32 | 87.99 | 11.067 | 2.361 | 12.099 | 100.0 |
| MPS | 87.80 | 86.43 | 86.40 | 11.067 | 1.111 | 12.247 | 96.3 |
| LGAT | 87.75 | 87.15 | 92.34 | 16.012 | 2.569 | 17.605 | 99.6 |
| V2 | 88.38 | 87.50 | 90.43 | 13.304 | 2.153 | 14.626 | 99.8 |
| V2-R | 86.38 | 86.23 | 94.83 | 20.537 | 3.264 | 22.584 | 99.4 |
| Gate G | 86.74 | 85.47 | 86.70 | 13.230 | 2.431 | 14.510 | 95.5 |
| Gate G35 | 74.38 | 75.38 | 87.17 | 36.078 | 43.889 | 35.152 | 88.6 |
| V2 with G35 | 88.26 | 87.35 | 90.06 | 13.208 | 1.875 | 14.551 | 99.6 |
| Unconstrained 300x8 (gate columns hidden) | 89.30 | 87.96 | 86.88 | 8.720 | 1.111 | 9.621 | 96.4 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 62.2 / 62.2 | 24.4 / 24.4 | 90.9 / 90.9 |
| MPS | 33.6 / 36.1 | 16.3 / 19.4 | 45.3 / 47.3 |
| LGAT | 15.0 / 15.3 | 6.2 / 6.6 | 81.7 / 81.8 |
| V2 | 12.2 / 12.4 | 1.6 / 1.8 | 25.4 / 25.5 |
| V2-R | 2.1 / 2.7 | 0.6 / 1.2 | 2.6 / 3.2 |
| Gate G | 0.0 / 4.5 | 0.0 / 4.5 | 0.0 / 4.5 |
| Gate G35 | 0.0 / 11.4 | 0.0 / 11.4 | 0.0 / 11.4 |
| V2 with G35 | 19.9 / 20.2 | 5.7 / 6.1 | 38.5 / 38.7 |
| Unconstrained 300x8 (gate columns hidden) | 47.4 / 49.3 | 21.5 / 24.3 | 74.3 / 75.2 |



### UNSW-NB15, lightgbm

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 88.47 | 87.28 | 87.92 | 11.082 | 2.639 | 12.082 | 100.0 |
| MPS | 87.67 | 86.23 | 85.79 | 10.787 | 2.500 | 11.770 | 95.2 |
| LGAT | 87.77 | 87.09 | 91.71 | 15.460 | 2.708 | 16.971 | 99.5 |
| V2 | 88.09 | 87.24 | 90.52 | 13.893 | 2.431 | 15.251 | 99.2 |
| V2-R | 86.31 | 86.10 | 94.27 | 20.206 | 4.722 | 22.041 | 99.4 |
| Gate G | 86.91 | 85.69 | 87.15 | 13.289 | 2.083 | 14.617 | 95.1 |
| Gate G35 | 74.04 | 75.18 | 87.41 | 36.902 | 43.750 | 36.091 | 90.2 |
| V2 with G35 | 88.41 | 87.22 | 87.92 | 11.185 | 2.639 | 12.198 | 100.0 |
| Unconstrained 300x8 (gate columns hidden) | 89.22 | 87.86 | 86.71 | 8.727 | 1.667 | 9.564 | 95.5 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 66.4 / 66.4 | 32.7 / 32.7 | 89.5 / 89.5 |
| MPS | 31.2 / 34.5 | 10.7 / 15.0 | 48.9 / 51.4 |
| LGAT | 14.5 / 14.9 | 6.3 / 6.8 | 72.3 / 72.4 |
| V2 | 22.0 / 22.6 | 2.1 / 2.9 | 31.9 / 32.4 |
| V2-R | 1.3 / 1.9 | 0.6 / 1.2 | 1.3 / 1.9 |
| Gate G | 0.0 / 4.9 | 0.0 / 4.9 | 0.0 / 4.9 |
| Gate G35 | 0.0 / 9.8 | 0.0 / 9.8 | 0.0 / 9.8 |
| V2 with G35 | 58.5 / 58.5 | 15.2 / 15.2 | 81.6 / 81.6 |
| Unconstrained 300x8 (gate columns hidden) | 56.4 / 58.4 | 27.7 / 31.0 | 86.2 / 86.8 |



### UNSW-NB15, catboost

| model | acc | F1 | recall | benign FPR % | FPR % top-decile fwd_pkts | FPR % other | cohort flagged before attack % |
|---|---|---|---|---|---|---|---|
| Undefended | 86.67 | 85.81 | 89.58 | 15.703 | 4.375 | 17.045 | 100.0 |
| MPS | 84.71 | 84.01 | 89.27 | 19.014 | 4.375 | 20.749 | 97.4 |
| LGAT | 86.20 | 85.92 | 93.62 | 19.875 | 4.375 | 21.712 | 99.3 |
| V2 | 86.42 | 85.98 | 92.58 | 18.617 | 5.069 | 20.222 | 100.0 |
| V2-R | 84.31 | 84.75 | 96.90 | 25.990 | 5.069 | 28.469 | 99.8 |
| Gate G | 86.59 | 85.43 | 87.39 | 14.062 | 2.917 | 15.383 | 94.0 |
| Gate G35 | 74.36 | 75.33 | 87.02 | 36.004 | 43.750 | 35.086 | 89.5 |
| V2 with G35 | 86.50 | 85.67 | 89.71 | 16.130 | 4.444 | 17.514 | 100.0 |
| Unconstrained 300x8 (gate columns hidden) | 89.39 | 88.06 | 86.96 | 8.624 | 1.458 | 9.473 | 94.7 |

| model | FFE M=20 (Real ASR / whole-cohort) | CAPGD (Real ASR / whole-cohort) | White-box (Real ASR / whole-cohort) |
|---|---|---|---|
| Undefended | 72.6 / 72.6 | 27.5 / 27.5 | 60.1 / 60.1 |
| MPS | 61.6 / 62.6 | 27.5 / 29.4 | 65.2 / 66.1 |
| LGAT | 27.1 / 27.6 | 9.0 / 9.6 | 45.4 / 45.8 |
| V2 | 4.9 / 4.9 | 1.0 / 1.0 | 5.7 / 5.7 |
| V2-R | 1.0 / 1.2 | 0.5 / 0.7 | 4.7 / 4.9 |
| Gate G | 0.0 / 6.0 | 0.0 / 6.0 | 0.0 / 6.0 |
| Gate G35 | 0.0 / 10.5 | 0.0 / 10.5 | 0.0 / 10.5 |
| V2 with G35 | 61.0 / 61.0 | 13.8 / 13.8 | 70.9 / 70.9 |
| Unconstrained 300x8 (gate columns hidden) | 49.7 / 52.4 | 17.7 / 22.1 | 66.7 / 68.5 |




## Findings

* **G35 (backward columns hidden) is not a free tightening.** It costs much more than the gate itself: clean accuracy / F1 drop by 3.9 / 19 points on CSE-CIC-IDS2018, 4.7-4.9 / 9 points on CIC-IDS2017 and 12-14 accuracy points on UNSW-NB15 (all three libraries), the benign false-positive rate rises to 2.1 % (CSE-CIC-IDS2018), 6 % (CIC-IDS2017) and 36 % (UNSW-NB15), and the top-decile-fwd_pkts FPR to 5 / 14-16 / 44 %. It still flags 74-76 % (CSE-CIC-IDS2018), 97-99 % (CIC-IDS2017) and 89-90 % (UNSW-NB15) of the cohort and is certified under the weaker assumption (backward traffic may change). V2 rebuilt with G35 under the same parity rule is clearly weaker than V2 where the gate matters (CSE-CIC-IDS2018 whole-cohort evasion 35.1-36.3 % FFE M=20, 23-70 % CAPGD, 78-79 % white-box, versus 0.2-8.3 %, 0.2-0.5 %, 7.9-9.0 % for V2; UNSW-NB15 XGBoost 20 / 6 / 39 % versus 12 / 2 / 26 %), because the parity rule forces a high tau. On CIC-IDS2017 it stays at 1-6 %. So V2's guarantee relies on the assumption that backward packets and bytes do not change; if that assumption is dropped, the hardened model keeps only a part of its strength at a large accuracy price on these three corpora.
* **Capacity control.** The unconstrained 300x8 model with the gate's hidden columns is as accurate as, or more accurate than, the undefended 60x6 model (CIC-IDS2017 +0.03, UNSW-NB15 +0.7 to +0.8 accuracy, CSE-CIC-IDS2018 -0.14 accuracy / -0.6 F1) and is as evadable as it (whole-cohort evasion 17-96 %, CIC-IDS2017 white-box 34-48 %). Hence the stand-alone gate's accuracy cost is the monotonicity: against the unconstrained control it is -0.55 accuracy / -3.1 F1 (CSE-CIC-IDS2018), -2.6 to -2.7 accuracy (UNSW-NB15 XGBoost/LightGBM; -2.8 CatBoost) and -0.06 (CIC-IDS2017). Capacity alone buys nothing against the attackers.
* **Whole-cohort numbers** change the picture for the gate-only and V2 rows by the missed share: V2 whole-cohort evasion is 0.2-9.0 % on CSE-CIC-IDS2018, 0.1-0.8 % on CIC-IDS2017 and 1.0-32.4 % on UNSW-NB15 (versus conditional Real ASR within 0.5-1 point of it, since V2 keeps 98-100 % of the cohort flagged before the attack). The stand-alone gate's whole-cohort evasion is its miss rate (7.9-9.0 % on CSE-CIC-IDS2018, 0.1-0.2 % CIC-IDS2017, 4.5-6.0 % UNSW-NB15).
* **False positives and monotone drift.** Benign FPR on the test split: CSE-CIC-IDS2018 0.03-0.07 % undefended versus 0.11-0.14 % for V2; CIC-IDS2017 0.01-0.02 % versus 0.02-0.05 %; UNSW-NB15 11-16 % versus 13-19 % (XGBoost 11.1 -> 13.3 %, LightGBM 11.1 -> 13.9 %, CatBoost 15.7 -> 18.6 %). On the top decile of benign forward-packet counts the FPR is 0.01-0.18 % (CSE-CIC-IDS2018), 0.01-0.03 % (CIC-IDS2017) and 2.2-5.1 % (UNSW-NB15) for V2, close to the undefended model on UNSW-NB15 (2.4 / 2.6 / 4.4 %) and not elevated relative to the rest; the stand-alone gate (rows "Gate G") shows no large-flow drift either (UNSW-NB15 top-decile FPR 2.9 / 2.1 / 2.4 % versus 4.4 / 2.6 / 2.4 % undefended). G35 is the exception (above). Large benign flows are rare in these 12-feature corpora, so this is weak evidence about drift in other traffic mixes.
* **FFE-v2 (CIC-IDS2017, XGBoost, first 300 cohort flows, retrained models, configuration of `results/ffev2_config.json`, "score + snapping")**: undefended 100.0 % Real ASR; V2 1.0 % (whole-cohort 1.3 %); V2 with G35 8.4 % (8.7 %); stand-alone G35 gate 0.0 % (whole-cohort 2.3 %, the miss rate). FFE-v2 was run only here, not on the other corpora or libraries.
