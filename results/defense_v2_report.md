# Defense v2: certified monotone gate plus hardened free classifier

All numbers come from `experiments/run_defenses_v2.py`; per-flow outcomes are stored in
`results/defense_v2_<dataset>.json` (`models.<library>.models.<defense>.attacks.<attack>.outcomes`;
codes: -1 not detected, 0 attack failed, 1 real evasion, 2 evasion but not schema-feasible).
Libraries: XGBoost, LightGBM, CatBoost (identical recipe and protocol). Corpora: CSE-CIC-IDS2018, CIC-IDS2017, UNSW-NB15. All 9 library x corpus cells ran.
Not run: random forest (the recipe supports it, `--libraries random_forest`, and the gate's certificate test passes for it) and FFE-v2 (see Section 6).

## 1. Summary

* The new defense **V2** keeps clean accuracy, F1 and attack recall within 0.5 point of the undefended model of the same library (enforced on validation, test values in the tables) and cuts Real ASR from 5-100 % (undefended) to 0.0-8.8 % on CSE-CIC-IDS2018, 0.1-0.7 % on CIC-IDS2017 and 1.0-32 % on UNSW-NB15. This includes the adaptive white-box attacker, which reads the defended trees, and CAPGD with a surrogate trained on the defended model's own outputs.
* The old defenses do not survive these attackers: MPS and LGAT leave 33-100 % white-box Real ASR on every corpus and library (99.5-100 % for LGAT on CSE-CIC-IDS2018 and for XGBoost on CIC-IDS2017), and LGAT's protection against FFE collapses from M=5 to M=20 (XGBoost CSE-CIC-IDS2018: 25.6 to 73.1 %).
* The strength comes from a certificate, not from tuning against one attack: the gate component can never lower its score under any schema-feasible transformation (Section 2), so any flow it flags cannot be evaded within the paper's threat model (the stand-alone gate has 0.0 % Real ASR for every attack, library and corpus). What is not certified is the residual: malicious flows that the gate does not flag but the free classifier does. That is where V2 can still be evaded (CSE-CIC-IDS2018: about 8 % of the cohort for all three libraries; UNSW-NB15: 5-32 %).
* The price of the certificate is real. The stand-alone gate loses about 0.7 accuracy, 4 F1 and 6 recall points on CSE-CIC-IDS2018, 0.1-1.8 accuracy points on UNSW-NB15 and nothing on CIC-IDS2017, and it misses 0.1-9 % of the cohort (flows the undefended model detects but the monotone model does not). V2 recovers clean parity by OR-fusing the gate with a free classifier and choosing the gate's threshold on validation.

## 2. Design

**Threat model (the paper's).** The attacker moves a malicious flow by temporal dilation, payload padding, chaff packets and window morphing, projected onto the flow manifold: destination port, protocol, backward packets and backward bytes are immutable; forward packets, forward bytes, duration and transport-header bytes can only increase; rates and forward mean length are recomputed from them; the TCP window is free.

**Component 1, certified gate (library-agnostic).** A tree ensemble that (i) is never shown the window, the two rates and the forward mean length (columns that are free or are derived functions of movable ones) and (ii) carries monotone constraints (+1) on the four increase-only columns, with larger capacity (300 trees, depth 8 for the boosters) to offset the constraint. Argument: every feasible evasion keeps the immutable columns fixed and moves the four constrained columns upward; a score that is non-decreasing in those columns, with everything else it reads fixed, cannot decrease. So `gate(x) >= tau` implies `gate(x') >= tau` for every feasible `x'`, and the attacker additionally needs the other component to miss the flow. The property is enforced by the library (XGBoost, LightGBM and CatBoost `monotone_constraints`, scikit-learn `monotonic_cst`) and checked by test: `tests/test_defenses_v2.py::test_gate_score_never_decreases_under_feasible_transformations` pushes random feasible transformations through gates of all four libraries and asserts the margin never falls; a companion test shows an unconstrained model does fall, so the check is not vacuous. This is the tree-ensemble analogue of monotonic classification for malware detection (Incer et al., DOI 10.1145/3180445.3180449). Related provable-robustness work for trees: Treant (Calzavara et al., DOI 10.1007/s10618-020-00694-9), feature partitioning (DOI 10.1186/s13635-021-00127-0), verifiable learning for robust tree ensembles (DOI 10.1145/3576915.3623100), and cost-aware robust tree ensembles for security (Chen et al., USENIX Security 2021, arXiv:1912.01149, no DOI). None of these handle protocol-coupled features, which is why the derived columns are hidden here.

**Component 2, free classifier (three options tried, all library-agnostic).** `orbit`: invariance by augmentation with 100,000 randomly transformed copies (random mixes of the four actions, log-uniform magnitudes up to the 110 s and 1000-chaff caps) of flows of both classes with their true labels, so dilation, padding and chaff are not evidence of either class (benign augmentation). `orbit_light`: the same with augmentation weight 0.25. `adversarial` (AdvTrain): iterative adversarial training, two rounds, each crafting 3,000 malicious flows against the current model (round 0: FFE M=5 and M=20; round 1: FFE M=20 and the projected white-box attacker), true labels, plus the orbit set, retrained from scratch.

**Fusion and tuning.** `score = max(free, gate shifted to fire at tau)`. For each library and corpus the pair (free classifier, tau in {none, 0.5, 0.8, 0.95, 0.99}) is chosen on the VALIDATION split only: filter by clean parity (accuracy, F1, recall each within 0.5 point of the undefended model on validation), then take the lowest mean validation Real ASR of FFE M=20 and the white-box attacker on a 300-flow validation cohort. **V2** is that selection. **V2-R** is a robustness-prioritised tier (accuracy and F1 within 3 points). The selection is logged before the test cohort is attacked. V2-R coincides with V2 on CSE-CIC-IDS2018 and CIC-IDS2017 and differs on UNSW-NB15.

**Tried and not used.** MPS alone and LGAT (kept as baselines); extra monotone-safe ratio features (backward/forward ratios; no gain in exploration, left as an option in `RobustTreeModel`); the manipulation-signature detector (`src/defenses/anomaly.py`, implemented but not evaluated: the gate made it unnecessary and time ran out). Capacity matters inside the constrained gate: in exploration runs on validation, raising XGBoost from 60x6 to 300x8 lifted the monotone model's accuracy from 84.9 to 86.4 on UNSW-NB15 and from 99.85 to 99.94 on CIC-IDS2017. A separate arm with larger unconstrained ensembles was not run.

**Protocol.** Cohort: 1,000 random malicious test flows detected by the undefended model of the same library (cohorts differ across libraries). Each model is attacked on the cohort flows it still detects ("cohort kept"); "retained" is the share of the whole cohort still detected after the attack (minimum over attacks). Attacks: FFE-v1 M=5 and M=20, CAPGD (surrogate MLP fitted to the defended model's soft outputs on 300k public flows, trained and run on CPU), and the projected white-box threshold-crossing attacker (`src/defenses/whitebox_generic.py`): the paper's control (`src/attacks/whitebox_tree.py`) generalised to any library through the raw ensemble margin, every candidate projected in the loop, also trying one step either side of each threshold, so for XGBoost it is similar to but not byte-identical with the paper's control. It reads the trees of the defended model including those of the fused gate, so it is the adaptive white-box attacker. Real ASR = evades and schema-feasible, Wilson 95 % CIs in brackets. McNemar is exact on pairs detected by both models; no multiplicity correction is applied (the key p-values are far below any threshold). One seed and one split per cell.

**Cost.** "build s" is wall-clock model construction on a shared, heavily loaded machine (XGBoost on the shared GPU, LightGBM and CatBoost on CPU, attack crafting on 8 worker processes), so only ratios are meaningful. V2 costs 5-19x the undefended training time (gate plus free model), dominated by the capacity-boosted gate; inference runs two ensembles.

## 3. Results

Attack columns are Real ASR in %. Models: Undefended; MPS; LGAT; Orbit and AdvTrain (free classifiers alone, no gate); Gate (certified, stand-alone); V2 (strict tier); V2-R (relaxed tier). MPS+LGAT appears in the McNemar rows only in the JSON. In the McNemar tables "a/b" means flows evaded only against the other model / only against V2.


## CSE-CIC-IDS2018

### CSE-CIC-IDS2018, catboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 97.25, F1 87.12, recall 77.57.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 97.25 | 87.18 | 77.99 | ok | 8.7 | 8.7 |  |
| undefended | 0.8 | 97.24 | 87.07 | 77.63 | ok | 8.7 | 8.7 |  |
| undefended | 0.95 | 97.24 | 87.07 | 77.58 | ok | 11.0 | 11.0 |  |
| undefended | 0.99 | 97.25 | 87.1 | 77.57 | ok | 30.3 | 33.3 |  |
| orbit | None | 97.23 | 87.01 | 77.4 | ok | 3.7 | 77.3 |  |
| orbit | 0.5 | 97.24 | 87.08 | 77.83 | ok | 0.0 | 8.4 |  |
| orbit | 0.8 | 97.22 | 86.98 | 77.47 | ok | 0.0 | 8.4 |  |
| orbit | 0.95 | 97.22 | 86.96 | 77.4 | ok | 2.0 | 10.7 |  |
| orbit | 0.99 | 97.23 | 86.99 | 77.4 | ok | 2.0 | 33.1 |  |
| orbit_light | None | 97.24 | 87.05 | 77.42 | ok | 30.8 | 99.7 |  |
| orbit_light | 0.5 | 97.24 | 87.12 | 77.85 | ok | 0.0 | 8.4 | yes |
| orbit_light | 0.8 | 97.23 | 87.02 | 77.5 | ok | 8.4 | 8.4 |  |
| orbit_light | 0.95 | 97.23 | 87.0 | 77.43 | ok | 10.4 | 10.7 |  |
| orbit_light | 0.99 | 97.24 | 87.03 | 77.43 | ok | 24.7 | 33.1 |  |
| adversarial | None | 97.24 | 87.04 | 77.41 | ok | 3.7 | 77.9 |  |
| adversarial | 0.5 | 97.24 | 87.11 | 77.84 | ok | 0.0 | 8.4 |  |
| adversarial | 0.8 | 97.23 | 87.01 | 77.48 | ok | 0.0 | 8.4 |  |
| adversarial | 0.95 | 97.23 | 86.99 | 77.41 | ok | 2.0 | 10.7 |  |
| adversarial | 0.99 | 97.24 | 87.02 | 77.41 | ok | 2.0 | 33.1 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 97.26 | 87.16 | 77.63 | 1000/1000 | 56.1 [53-59] | 59.8 [57-63] | 99.3 [99-100] | 100.0 [100-100] | 0.0 | 51 |
| MPS | 97.28 | 87.21 | 77.56 | 998/1000 | 39.8 [37-43] | 39.8 [37-43] | 99.2 [98-100] | 100.0 [100-100] | 0.0 | 109 |
| LGAT | 97.25 | 87.11 | 77.52 | 998/1000 | 5.5 [4-7] | 45.6 [42-49] | 83.8 [81-86] | 100.0 [100-100] | 0.0 | 86 |
| MPS+LGAT | 97.22 | 86.97 | 77.57 | 999/1000 | 3.2 [2-4] | 3.2 [2-4] | 99.3 [99-100] | 100.0 [100-100] | 0.0 | 134 |
| Orbit | 97.24 | 87.04 | 77.42 | 998/1000 | 1.6 [1-3] | 3.1 [2-4] | 49.9 [47-53] | 80.2 [78-82] | 19.8 | 102 |
| AdvTrain | 97.25 | 87.06 | 77.43 | 998/1000 | 1.8 [1-3] | 3.1 [2-4] | 46.6 [44-50] | 80.0 [77-82] | 20.0 | 188 |
| Gate (certified) | 96.58 | 83.43 | 71.87 | 910/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 91.0 | 442 |
| V2 | 97.25 | 87.16 | 77.89 | 998/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 8.8 [7-11] | 91.0 | 508 |
| V2-R | 97.25 | 87.16 | 77.89 | 998/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 8.8 [7-11] | 91.0 | 508 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 559/0 p=1.1e-168 | 596/0 p=7.7e-180 | 991/0 p=9.6e-299 | 910/0 p=2.3e-274 |
| V2 vs MPS | 396/0 p=1.2e-119 | 396/0 p=1.2e-119 | 989/0 p=3.8e-298 | 910/0 p=2.3e-274 |
| V2 vs LGAT | 55/0 p=5.6e-17 | 455/0 p=2.1e-137 | 836/0 p=4.4e-252 | 910/0 p=2.3e-274 |
| V2 vs MPS+LGAT | 31/0 p=9.3e-10 | 31/0 p=9.3e-10 | 991/0 p=9.6e-299 | 910/0 p=2.3e-274 |
| V2 vs AdvTrain | 18/0 p=7.6e-06 | 31/0 p=9.3e-10 | 465/0 p=2.1e-140 | 710/0 p=3.7e-214 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |

### CSE-CIC-IDS2018, lightgbm

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 97.29, F1 87.3, recall 77.85.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 97.27 | 87.29 | 78.2 | ok | 8.0 | 8.0 | yes |
| undefended | 0.8 | 97.28 | 87.26 | 77.89 | ok | 8.3 | 8.3 |  |
| undefended | 0.95 | 97.28 | 87.26 | 77.86 | ok | 9.7 | 9.7 |  |
| undefended | 0.99 | 97.28 | 87.26 | 77.86 | ok | 10.3 | 10.3 |  |
| orbit | None | 97.27 | 87.27 | 78.09 | ok | 11.4 | 76.6 |  |
| orbit | 0.5 | 97.26 | 87.25 | 78.31 | ok | 8.0 | 8.0 |  |
| orbit | 0.8 | 97.26 | 87.24 | 78.13 | ok | 8.0 | 8.0 |  |
| orbit | 0.95 | 97.26 | 87.24 | 78.1 | ok | 8.7 | 9.4 |  |
| orbit | 0.99 | 97.26 | 87.24 | 78.1 | ok | 9.4 | 10.0 |  |
| orbit_light | None | 97.29 | 87.3 | 77.88 | ok | 36.5 | 76.6 |  |
| orbit_light | 0.5 | 97.27 | 87.29 | 78.22 | ok | 8.0 | 8.0 |  |
| orbit_light | 0.8 | 97.28 | 87.26 | 77.92 | ok | 8.0 | 8.0 |  |
| orbit_light | 0.95 | 97.28 | 87.26 | 77.89 | ok | 9.0 | 9.4 |  |
| orbit_light | 0.99 | 97.28 | 87.26 | 77.89 | ok | 9.7 | 10.0 |  |
| adversarial | None | 97.27 | 87.25 | 78.03 | ok | 9.4 | 76.6 |  |
| adversarial | 0.5 | 97.26 | 87.27 | 78.36 | ok | 8.0 | 8.0 |  |
| adversarial | 0.8 | 97.27 | 87.24 | 78.05 | ok | 8.0 | 8.0 |  |
| adversarial | 0.95 | 97.27 | 87.24 | 78.03 | ok | 8.7 | 9.4 |  |
| adversarial | 0.99 | 97.27 | 87.24 | 78.03 | ok | 9.4 | 10.0 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 97.28 | 87.28 | 77.83 | 1000/1000 | 68.4 [66-71] | 99.8 [99-100] | 99.5 [99-100] | 99.8 [99-100] | 0.2 | 50 |
| MPS | 97.28 | 87.26 | 77.65 | 998/1000 | 42.3 [39-45] | 42.3 [39-45] | 98.5 [98-99] | 99.8 [99-100] | 0.2 | 6 |
| LGAT | 97.28 | 87.25 | 77.82 | 1000/1000 | 12.8 [11-15] | 56.8 [54-60] | 96.3 [95-97] | 99.8 [99-100] | 0.2 | 15 |
| MPS+LGAT | 97.29 | 87.27 | 77.68 | 998/1000 | 38.3 [35-41] | 38.3 [35-41] | 98.7 [98-99] | 99.7 [99-100] | 0.3 | 12 |
| Orbit | 97.27 | 87.26 | 78.09 | 1000/1000 | 8.8 [7-11] | 12.2 [10-14] | 9.4 [8-11] | 81.6 [79-84] | 18.4 | 29 |
| AdvTrain | 97.27 | 87.23 | 78.00 | 1000/1000 | 1.3 [1-2] | 8.7 [7-11] | 7.4 [6-9] | 81.5 [79-84] | 18.5 | 143 |
| Gate (certified) | 96.58 | 83.43 | 72.00 | 921/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 92.1 | 500 |
| V2 | 97.27 | 87.29 | 78.23 | 1000/1000 | 7.9 [6-10] | 7.9 [6-10] | 0.2 [0-1] | 7.9 [6-10] | 92.1 | 549 |
| V2-R | 97.27 | 87.29 | 78.23 | 1000/1000 | 7.9 [6-10] | 7.9 [6-10] | 0.2 [0-1] | 7.9 [6-10] | 92.1 | 549 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 605/0 p=1.5e-182 | 919/0 p=4.5e-277 | 993/0 p=2.4e-299 | 919/0 p=4.5e-277 |
| V2 vs MPS | 346/1 p=2.4e-102 | 346/1 p=2.4e-102 | 983/0 p=2.4e-296 | 919/0 p=4.5e-277 |
| V2 vs LGAT | 50/1 p=4.6e-14 | 490/1 p=1.5e-145 | 961/0 p=1e-289 | 919/0 p=4.5e-277 |
| V2 vs MPS+LGAT | 306/1 p=2.4e-90 | 306/1 p=2.4e-90 | 985/0 p=6.1e-297 | 918/0 p=9e-277 |
| V2 vs AdvTrain | 9/75 p=4.3e-14 | 9/1 p=0.021 | 72/0 p=4.2e-22 | 736/0 p=5.5e-222 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |

### CSE-CIC-IDS2018, xgboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 97.3, F1 87.3, recall 77.62.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 97.3 | 87.42 | 78.19 | ok | 6.0 | 6.0 |  |
| undefended | 0.8 | 97.29 | 87.26 | 77.64 | ok | 6.0 | 6.0 |  |
| undefended | 0.95 | 97.29 | 87.27 | 77.63 | ok | 7.3 | 7.3 |  |
| undefended | 0.99 | 97.29 | 87.27 | 77.63 | ok | 7.3 | 7.3 |  |
| orbit | None | 97.28 | 87.25 | 77.85 | ok | 18.0 | 76.3 |  |
| orbit | 0.5 | 97.28 | 87.36 | 78.4 | ok | 6.0 | 6.0 |  |
| orbit | 0.8 | 97.27 | 87.21 | 77.86 | ok | 6.0 | 6.0 |  |
| orbit | 0.95 | 97.27 | 87.23 | 77.85 | ok | 7.3 | 7.3 |  |
| orbit | 0.99 | 97.27 | 87.23 | 77.85 | ok | 7.3 | 7.3 |  |
| orbit_light | None | 97.29 | 87.3 | 77.83 | ok | 31.7 | 99.7 |  |
| orbit_light | 0.5 | 97.3 | 87.43 | 78.44 | ok | 6.0 | 6.0 | yes |
| orbit_light | 0.8 | 97.28 | 87.28 | 77.89 | ok | 6.0 | 6.0 |  |
| orbit_light | 0.95 | 97.29 | 87.3 | 77.88 | ok | 6.0 | 7.3 |  |
| orbit_light | 0.99 | 97.29 | 87.3 | 77.87 | ok | 7.3 | 7.3 |  |
| adversarial | None | 97.27 | 87.23 | 77.86 | ok | 7.3 | 75.7 |  |
| adversarial | 0.5 | 97.28 | 87.36 | 78.41 | ok | 6.0 | 6.0 |  |
| adversarial | 0.8 | 97.27 | 87.22 | 77.87 | ok | 6.0 | 6.0 |  |
| adversarial | 0.95 | 97.27 | 87.23 | 77.86 | ok | 7.3 | 7.3 |  |
| adversarial | 0.99 | 97.27 | 87.23 | 77.86 | ok | 7.3 | 7.3 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 97.30 | 87.31 | 77.62 | 1000/1000 | 73.3 [70-76] | 99.5 [99-100] | 99.0 [98-100] | 99.5 [99-100] | 0.5 | 44 |
| MPS | 97.30 | 87.30 | 77.61 | 999/1000 | 61.9 [59-65] | 61.9 [59-65] | 98.3 [97-99] | 99.5 [99-100] | 0.5 | 10 |
| LGAT | 97.28 | 87.26 | 77.83 | 998/1000 | 25.6 [23-28] | 73.1 [70-76] | 96.1 [95-97] | 99.5 [99-100] | 0.5 | 21 |
| MPS+LGAT | 97.30 | 87.30 | 77.62 | 998/1000 | 45.5 [42-49] | 45.5 [42-49] | 95.3 [94-96] | 99.5 [99-100] | 0.5 | 23 |
| Orbit | 97.28 | 87.24 | 77.81 | 998/1000 | 9.5 [8-12] | 20.4 [18-23] | 3.9 [3-5] | 81.5 [79-84] | 18.5 | 27 |
| AdvTrain | 97.27 | 87.22 | 77.82 | 998/1000 | 1.4 [1-2] | 9.0 [7-11] | 6.8 [5-8] | 80.7 [78-83] | 19.3 | 2280 |
| Gate (certified) | 96.60 | 83.56 | 72.20 | 917/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 91.7 | 350 |
| V2 | 97.29 | 87.40 | 78.40 | 998/1000 | 8.1 [7-10] | 8.1 [7-10] | 0.3 [0-1] | 8.1 [7-10] | 91.7 | 368 |
| V2-R | 97.29 | 87.40 | 78.40 | 998/1000 | 8.1 [7-10] | 8.1 [7-10] | 0.3 [0-1] | 8.1 [7-10] | 91.7 | 368 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 650/0 p=4.3e-196 | 912/0 p=5.8e-275 | 985/0 p=6.1e-297 | 912/0 p=5.8e-275 |
| V2 vs MPS | 536/0 p=8.9e-162 | 536/0 p=8.9e-162 | 977/0 p=1.6e-294 | 912/0 p=5.8e-275 |
| V2 vs LGAT | 173/0 p=1.7e-52 | 648/0 p=1.7e-195 | 954/0 p=1.3e-287 | 911/0 p=1.2e-274 |
| V2 vs MPS+LGAT | 373/0 p=1e-112 | 373/0 p=1e-112 | 947/0 p=1.7e-285 | 912/0 p=5.8e-275 |
| V2 vs AdvTrain | 9/76 p=2.4e-14 | 9/0 p=0.0039 | 65/0 p=5.4e-20 | 724/0 p=2.3e-218 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |


## CIC-IDS2017

### CIC-IDS2017, catboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 99.85, F1 99.69, recall 99.43.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 99.92 | 99.84 | 99.87 | ok | 0.0 | 0.0 | yes |
| undefended | 0.8 | 99.92 | 99.83 | 99.77 | ok | 0.0 | 0.0 |  |
| undefended | 0.95 | 99.88 | 99.74 | 99.56 | ok | 0.3 | 0.3 |  |
| undefended | 0.99 | 99.85 | 99.69 | 99.43 | ok | 3.3 | 4.7 |  |
| orbit | None | 99.8 | 99.58 | 99.34 | ok | 2.3 | 34.7 |  |
| orbit | 0.5 | 99.91 | 99.82 | 99.87 | ok | 0.0 | 0.0 |  |
| orbit | 0.8 | 99.89 | 99.78 | 99.73 | ok | 0.0 | 0.0 |  |
| orbit | 0.95 | 99.85 | 99.68 | 99.54 | ok | 0.3 | 0.3 |  |
| orbit | 0.99 | 99.8 | 99.59 | 99.34 | ok | 1.3 | 4.3 |  |
| orbit_light | None | 99.85 | 99.68 | 99.49 | ok | 9.7 | 36.0 |  |
| orbit_light | 0.5 | 99.93 | 99.84 | 99.9 | ok | 0.0 | 0.0 |  |
| orbit_light | 0.8 | 99.92 | 99.83 | 99.8 | ok | 0.0 | 0.0 |  |
| orbit_light | 0.95 | 99.87 | 99.74 | 99.62 | ok | 0.3 | 0.3 |  |
| orbit_light | 0.99 | 99.85 | 99.68 | 99.5 | ok | 1.7 | 4.3 |  |
| adversarial | None | 99.84 | 99.67 | 99.51 | ok | 2.3 | 34.7 |  |
| adversarial | 0.5 | 99.92 | 99.82 | 99.86 | ok | 0.0 | 0.0 |  |
| adversarial | 0.8 | 99.9 | 99.8 | 99.77 | ok | 0.0 | 0.0 |  |
| adversarial | 0.95 | 99.88 | 99.76 | 99.68 | ok | 0.3 | 0.3 |  |
| adversarial | 0.99 | 99.85 | 99.68 | 99.52 | ok | 1.3 | 4.3 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 99.85 | 99.68 | 99.41 | 1000/1000 | 31.1 [28-34] | 32.0 [29-35] | 5.4 [4-7] | 33.6 [31-37] | 66.4 | 8 |
| MPS | 99.82 | 99.63 | 99.38 | 997/1000 | 32.0 [29-35] | 32.0 [29-35] | 25.0 [22-28] | 32.8 [30-36] | 67.0 | 32 |
| LGAT | 99.86 | 99.70 | 99.46 | 1000/1000 | 2.1 [1-3] | 31.2 [28-34] | 31.0 [28-34] | 33.2 [30-36] | 66.8 | 46 |
| MPS+LGAT | 99.83 | 99.64 | 99.40 | 996/1000 | 20.2 [18-23] | 31.7 [29-35] | 26.8 [24-30] | 33.2 [30-36] | 66.5 | 77 |
| Orbit | 99.80 | 99.59 | 99.34 | 999/1000 | 1.4 [1-2] | 1.4 [1-2] | 21.6 [19-24] | 32.3 [30-35] | 67.6 | 22 |
| AdvTrain | 99.84 | 99.66 | 99.49 | 998/1000 | 1.3 [1-2] | 1.6 [1-3] | 10.1 [8-12] | 32.3 [29-35] | 67.6 | 116 |
| Gate (certified) | 99.92 | 99.83 | 99.79 | 999/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 99.9 | 171 |
| V2 | 99.93 | 99.85 | 99.87 | 1000/1000 | 0.1 [0-1] | 0.1 [0-1] | 0.1 [0-1] | 0.1 [0-1] | 99.9 | 179 |
| V2-R | 99.93 | 99.85 | 99.87 | 1000/1000 | 0.1 [0-1] | 0.1 [0-1] | 0.1 [0-1] | 0.1 [0-1] | 99.9 | 179 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 310/0 p=9.6e-94 | 319/0 p=1.9e-96 | 53/0 p=2.2e-16 | 335/0 p=2.9e-101 |
| V2 vs MPS | 319/0 p=1.9e-96 | 319/0 p=1.9e-96 | 249/0 p=2.2e-75 | 327/0 p=7.3e-99 |
| V2 vs LGAT | 20/0 p=1.9e-06 | 311/0 p=4.8e-94 | 309/0 p=1.9e-93 | 331/0 p=4.6e-100 |
| V2 vs MPS+LGAT | 201/0 p=6.2e-61 | 316/0 p=1.5e-95 | 267/0 p=8.4e-81 | 331/0 p=4.6e-100 |
| V2 vs AdvTrain | 12/0 p=0.00049 | 15/0 p=6.1e-05 | 100/0 p=1.6e-30 | 321/0 p=4.7e-97 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |

### CIC-IDS2017, lightgbm

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 99.96, F1 99.93, recall 99.86.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 99.96 | 99.91 | 99.97 | ok | 0.0 | 0.0 |  |
| undefended | 0.8 | 99.97 | 99.94 | 99.94 | ok | 0.0 | 0.0 | yes |
| undefended | 0.95 | 99.97 | 99.95 | 99.93 | ok | 0.3 | 0.3 |  |
| undefended | 0.99 | 99.97 | 99.95 | 99.91 | ok | 1.0 | 1.0 |  |
| orbit | None | 99.95 | 99.9 | 99.84 | ok | 2.7 | 33.0 |  |
| orbit | 0.5 | 99.95 | 99.9 | 99.97 | ok | 0.0 | 0.0 |  |
| orbit | 0.8 | 99.96 | 99.92 | 99.94 | ok | 0.0 | 0.0 |  |
| orbit | 0.95 | 99.97 | 99.93 | 99.93 | ok | 0.3 | 0.3 |  |
| orbit | 0.99 | 99.96 | 99.93 | 99.91 | ok | 0.7 | 1.0 |  |
| orbit_light | None | 99.96 | 99.91 | 99.87 | ok | 7.7 | 33.0 |  |
| orbit_light | 0.5 | 99.95 | 99.91 | 99.97 | ok | 0.0 | 0.0 |  |
| orbit_light | 0.8 | 99.97 | 99.93 | 99.95 | ok | 0.0 | 0.0 |  |
| orbit_light | 0.95 | 99.97 | 99.94 | 99.94 | ok | 0.3 | 0.3 |  |
| orbit_light | 0.99 | 99.97 | 99.94 | 99.92 | ok | 1.0 | 1.0 |  |
| adversarial | None | 99.94 | 99.88 | 99.83 | ok | 2.0 | 17.3 |  |
| adversarial | 0.5 | 99.95 | 99.9 | 99.97 | ok | 0.0 | 0.0 |  |
| adversarial | 0.8 | 99.96 | 99.92 | 99.93 | ok | 0.0 | 0.0 |  |
| adversarial | 0.95 | 99.96 | 99.92 | 99.92 | ok | 0.3 | 0.3 |  |
| adversarial | 0.99 | 99.96 | 99.91 | 99.9 | ok | 0.3 | 1.0 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 99.96 | 99.92 | 99.87 | 1000/1000 | 20.3 [18-23] | 34.6 [32-38] | 57.9 [55-61] | 35.7 [33-39] | 42.1 | 11 |
| MPS | 99.91 | 99.82 | 99.73 | 999/1000 | 34.1 [31-37] | 34.1 [31-37] | 31.6 [29-35] | 35.4 [32-38] | 64.5 | 6 |
| LGAT | 99.96 | 99.92 | 99.87 | 1000/1000 | 22.7 [20-25] | 33.4 [30-36] | 36.3 [33-39] | 35.6 [33-39] | 63.7 | 54 |
| MPS+LGAT | 99.92 | 99.82 | 99.74 | 999/1000 | 33.3 [30-36] | 33.4 [31-36] | 95.9 [94-97] | 35.3 [32-38] | 4.1 | 36 |
| Orbit | 99.94 | 99.88 | 99.81 | 1000/1000 | 2.0 [1-3] | 4.0 [3-5] | 3.1 [2-4] | 35.4 [32-38] | 64.6 | 26 |
| AdvTrain | 99.94 | 99.87 | 99.82 | 1000/1000 | 1.8 [1-3] | 2.8 [2-4] | 2.3 [2-3] | 18.5 [16-21] | 81.5 | 252 |
| Gate (certified) | 99.91 | 99.81 | 99.76 | 999/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 99.9 | 96 |
| V2 | 99.97 | 99.93 | 99.94 | 1000/1000 | 0.4 [0-1] | 0.4 [0-1] | 0.4 [0-1] | 0.4 [0-1] | 99.6 | 107 |
| V2-R | 99.97 | 99.93 | 99.94 | 1000/1000 | 0.4 [0-1] | 0.4 [0-1] | 0.4 [0-1] | 0.4 [0-1] | 99.6 | 107 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 199/0 p=2.5e-60 | 342/0 p=2.2e-103 | 575/0 p=1.6e-173 | 353/0 p=1.1e-106 |
| V2 vs MPS | 337/0 p=7.1e-102 | 337/0 p=7.1e-102 | 312/0 p=2.4e-94 | 350/0 p=8.7e-106 |
| V2 vs LGAT | 223/0 p=1.5e-67 | 330/0 p=9.1e-100 | 359/0 p=1.7e-108 | 352/0 p=2.2e-106 |
| V2 vs MPS+LGAT | 329/0 p=1.8e-99 | 330/0 p=9.1e-100 | 954/0 p=1.3e-287 | 349/0 p=1.7e-105 |
| V2 vs AdvTrain | 15/1 p=0.00052 | 25/1 p=8e-07 | 23/4 p=0.00031 | 181/0 p=6.5e-55 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |

### CIC-IDS2017, xgboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 99.96, F1 99.92, recall 99.85.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 99.96 | 99.91 | 99.98 | ok | 0.7 | 0.7 |  |
| undefended | 0.8 | 99.96 | 99.92 | 99.96 | ok | 0.7 | 0.7 |  |
| undefended | 0.95 | 99.97 | 99.94 | 99.92 | ok | 0.7 | 0.7 |  |
| undefended | 0.99 | 99.97 | 99.94 | 99.9 | ok | 2.0 | 2.0 |  |
| orbit | None | 99.96 | 99.91 | 99.85 | ok | 3.3 | 34.3 |  |
| orbit | 0.5 | 99.95 | 99.91 | 99.98 | ok | 0.7 | 0.7 |  |
| orbit | 0.8 | 99.96 | 99.92 | 99.96 | ok | 0.7 | 0.7 |  |
| orbit | 0.95 | 99.97 | 99.93 | 99.93 | ok | 0.7 | 0.7 |  |
| orbit | 0.99 | 99.97 | 99.93 | 99.9 | ok | 0.3 | 2.0 |  |
| orbit_light | None | 99.96 | 99.91 | 99.86 | ok | 6.7 | 35.0 |  |
| orbit_light | 0.5 | 99.96 | 99.91 | 99.98 | ok | 0.3 | 0.7 |  |
| orbit_light | 0.8 | 99.96 | 99.92 | 99.97 | ok | 0.7 | 0.7 |  |
| orbit_light | 0.95 | 99.97 | 99.93 | 99.93 | ok | 0.3 | 0.7 | yes |
| orbit_light | 0.99 | 99.97 | 99.94 | 99.91 | ok | 1.7 | 2.0 |  |
| adversarial | None | 99.96 | 99.91 | 99.86 | ok | 2.7 | 19.3 |  |
| adversarial | 0.5 | 99.96 | 99.91 | 99.98 | ok | 0.7 | 0.7 |  |
| adversarial | 0.8 | 99.96 | 99.92 | 99.96 | ok | 0.7 | 0.7 |  |
| adversarial | 0.95 | 99.97 | 99.93 | 99.93 | ok | 0.7 | 0.7 |  |
| adversarial | 0.99 | 99.97 | 99.93 | 99.9 | ok | 0.3 | 2.0 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 99.96 | 99.91 | 99.85 | 1000/1000 | 69.7 [67-72] | 99.1 [98-100] | 77.0 [74-80] | 100.0 [100-100] | 0.0 | 38 |
| MPS | 99.91 | 99.82 | 99.73 | 999/1000 | 92.8 [91-94] | 92.8 [91-94] | 47.7 [45-51] | 98.1 [97-99] | 1.9 | 64 |
| LGAT | 99.96 | 99.92 | 99.87 | 1000/1000 | 3.0 [2-4] | 31.9 [29-35] | 52.6 [50-56] | 100.0 [100-100] | 0.0 | 75 |
| MPS+LGAT | 99.92 | 99.83 | 99.73 | 999/1000 | 19.7 [17-22] | 32.5 [30-36] | 57.4 [54-60] | 94.9 [93-96] | 5.1 | 59 |
| Orbit | 99.95 | 99.89 | 99.82 | 1000/1000 | 2.1 [1-3] | 2.8 [2-4] | 7.4 [6-9] | 34.3 [31-37] | 65.7 | 3 |
| AdvTrain | 99.95 | 99.89 | 99.82 | 1000/1000 | 1.6 [1-3] | 2.6 [2-4] | 1.2 [1-2] | 15.2 [13-18] | 84.8 | 144 |
| Gate (certified) | 99.93 | 99.86 | 99.85 | 998/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 99.8 | 182 |
| V2 | 99.97 | 99.93 | 99.94 | 999/1000 | 0.2 [0-1] | 0.7 [0-1] | 0.7 [0-1] | 0.7 [0-1] | 99.2 | 199 |
| V2-R | 99.97 | 99.93 | 99.94 | 999/1000 | 0.2 [0-1] | 0.7 [0-1] | 0.7 [0-1] | 0.7 [0-1] | 99.2 | 199 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 694/0 p=2.4e-209 | 983/0 p=2.4e-296 | 763/1 p=1.6e-227 | 992/0 p=4.8e-299 |
| V2 vs MPS | 924/0 p=1.4e-278 | 919/0 p=4.5e-277 | 470/1 p=1.5e-139 | 972/0 p=5e-293 |
| V2 vs LGAT | 28/0 p=7.5e-09 | 312/0 p=2.4e-94 | 519/1 p=3e-154 | 992/0 p=4.8e-299 |
| V2 vs MPS+LGAT | 195/0 p=4e-59 | 318/0 p=3.7e-96 | 565/0 p=1.7e-170 | 940/0 p=2.2e-283 |
| V2 vs AdvTrain | 14/0 p=0.00012 | 24/5 p=0.00055 | 10/6 p=0.45 | 144/0 p=9e-44 |
| V2 vs V2-R | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 | 0/0 p=1 |


## UNSW-NB15

### UNSW-NB15, catboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 86.45, F1 85.67, recall 90.03.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 86.21 | 85.85 | 93.01 | ok | 4.0 | 4.3 | yes |
| undefended | 0.8 | 86.86 | 86.16 | 91.0 | ok | 27.3 | 30.7 |  |
| undefended | 0.95 | 86.57 | 85.81 | 90.3 | ok | 38.3 | 40.3 |  |
| undefended | 0.99 | 86.47 | 85.69 | 90.06 | ok | 50.0 | 54.7 |  |
| orbit | None | 83.57 | 83.74 | 94.09 | no | 19.6 | 38.5 |  |
| orbit | 0.5 | 84.05 | 84.51 | 96.78 | no | 2.3 | 4.7 |  |
| orbit | 0.8 | 84.1 | 84.36 | 95.34 | no | 6.4 | 27.2 |  |
| orbit | 0.95 | 83.74 | 83.94 | 94.47 | no | 11.1 | 34.5 |  |
| orbit | 0.99 | 83.58 | 83.75 | 94.11 | no | 13.2 | 38.9 |  |
| orbit_light | None | 85.36 | 85.0 | 92.26 | no | 27.8 | 51.2 |  |
| orbit_light | 0.5 | 85.17 | 85.17 | 94.65 | no | 4.0 | 4.7 |  |
| orbit_light | 0.8 | 85.8 | 85.53 | 93.3 | no | 10.7 | 29.0 |  |
| orbit_light | 0.95 | 85.47 | 85.14 | 92.51 | no | 16.4 | 39.8 |  |
| orbit_light | 0.99 | 85.37 | 85.01 | 92.28 | no | 19.7 | 43.8 |  |
| adversarial | None | 83.46 | 83.79 | 95.05 | no | 10.8 | 36.4 |  |
| adversarial | 0.5 | 83.8 | 84.34 | 96.99 | no | 0.7 | 4.0 |  |
| adversarial | 0.8 | 83.92 | 84.32 | 96.11 | no | 2.7 | 24.6 |  |
| adversarial | 0.95 | 83.61 | 83.96 | 95.39 | no | 7.7 | 32.0 |  |
| adversarial | 0.99 | 83.47 | 83.81 | 95.08 | no | 9.4 | 36.4 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 86.67 | 85.81 | 89.58 | 1000/1000 | 62.2 [59-65] | 72.6 [70-75] | 27.5 [25-30] | 60.1 [57-63] | 27.4 | 3 |
| MPS | 84.71 | 84.01 | 89.27 | 974/1000 | 61.6 [58-65] | 61.6 [58-65] | 27.5 [25-30] | 65.2 [62-68] | 33.9 | 5 |
| LGAT | 86.20 | 85.92 | 93.62 | 993/1000 | 17.2 [15-20] | 27.1 [24-30] | 9.0 [7-11] | 45.4 [42-48] | 54.2 | 25 |
| MPS+LGAT | 85.18 | 84.96 | 93.09 | 987/1000 | 20.8 [18-23] | 22.2 [20-25] | 7.1 [6-9] | 45.8 [43-49] | 53.5 | 26 |
| Orbit | 84.08 | 84.13 | 93.82 | 982/1000 | 14.7 [13-17] | 19.0 [17-22] | 10.9 [9-13] | 40.7 [38-44] | 58.2 | 5 |
| AdvTrain | 84.06 | 84.28 | 94.96 | 990/1000 | 8.8 [7-11] | 9.1 [8-11] | 12.4 [10-15] | 37.5 [34-40] | 61.9 | 142 |
| Gate (certified) | 86.59 | 85.43 | 87.39 | 940/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 94.0 | 53 |
| V2 | 86.42 | 85.98 | 92.58 | 1000/1000 | 4.9 [4-6] | 4.9 [4-6] | 1.0 [0-2] | 5.7 [4-7] | 94.3 | 56 |
| V2-R | 84.31 | 84.75 | 96.90 | 998/1000 | 1.0 [0-2] | 1.0 [0-2] | 0.5 [0-1] | 4.7 [4-6] | 95.1 | 195 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 577/4 p=1.2e-165 | 681/4 p=1.1e-196 | 265/0 p=3.4e-80 | 544/0 p=3.5e-164 |
| V2 vs MPS | 565/4 p=4.5e-162 | 565/4 p=4.5e-162 | 263/5 p=4.8e-71 | 589/1 p=2.9e-175 |
| V2 vs LGAT | 147/22 p=6.9e-24 | 245/22 p=8.2e-49 | 87/8 p=6.7e-18 | 407/10 p=2.4e-106 |
| V2 vs MPS+LGAT | 190/28 p=8.9e-31 | 204/28 p=3.4e-34 | 68/8 p=5.6e-13 | 418/17 p=3.4e-101 |
| V2 vs AdvTrain | 76/36 p=0.0002 | 79/36 p=7.5e-05 | 120/7 p=1.1e-27 | 319/3 p=1.3e-90 |
| V2 vs V2-R | 0/37 p=1.5e-11 | 0/37 p=1.5e-11 | 3/8 p=0.23 | 0/8 p=0.0078 |

### UNSW-NB15, lightgbm

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 88.16, F1 87.01, recall 88.17.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 87.02 | 86.35 | 91.27 | no | 3.3 | 3.7 |  |
| undefended | 0.8 | 88.27 | 87.16 | 88.53 | ok | 30.3 | 32.3 |  |
| undefended | 0.95 | 88.22 | 87.08 | 88.3 | ok | 30.7 | 40.7 |  |
| undefended | 0.99 | 88.17 | 87.02 | 88.19 | ok | 38.3 | 45.0 |  |
| orbit | None | 87.15 | 86.41 | 90.87 | no | 12.8 | 32.7 |  |
| orbit | 0.5 | 86.65 | 86.17 | 92.44 | no | 2.0 | 2.0 |  |
| orbit | 0.8 | 87.32 | 86.63 | 91.32 | no | 8.8 | 25.9 |  |
| orbit | 0.95 | 87.21 | 86.5 | 91.05 | no | 8.8 | 27.6 |  |
| orbit | 0.99 | 87.16 | 86.44 | 90.93 | no | 9.8 | 29.6 |  |
| orbit_light | None | 87.61 | 86.81 | 90.67 | no | 30.2 | 43.6 |  |
| orbit_light | 0.5 | 86.82 | 86.29 | 92.22 | no | 2.3 | 3.3 |  |
| orbit_light | 0.8 | 87.74 | 86.98 | 91.09 | ok | 21.4 | 29.8 | yes |
| orbit_light | 0.95 | 87.68 | 86.9 | 90.84 | ok | 21.7 | 30.4 |  |
| orbit_light | 0.99 | 87.63 | 86.84 | 90.73 | no | 22.1 | 32.4 |  |
| adversarial | None | 85.57 | 85.44 | 94.11 | no | 3.7 | 5.1 |  |
| adversarial | 0.5 | 85.46 | 85.48 | 95.17 | no | 1.0 | 1.0 |  |
| adversarial | 0.8 | 85.8 | 85.71 | 94.67 | no | 1.0 | 1.0 |  |
| adversarial | 0.95 | 85.7 | 85.59 | 94.4 | no | 1.0 | 1.0 |  |
| adversarial | 0.99 | 85.61 | 85.48 | 94.2 | no | 1.7 | 2.0 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 88.47 | 87.28 | 87.92 | 1000/1000 | 52.9 [50-56] | 66.4 [63-69] | 32.7 [30-36] | 89.5 [87-91] | 10.5 | 10 |
| MPS | 87.67 | 86.23 | 85.79 | 952/1000 | 29.9 [27-33] | 31.2 [28-34] | 10.7 [9-13] | 48.9 [46-52] | 48.6 | 22 |
| LGAT | 87.77 | 87.09 | 91.71 | 995/1000 | 9.7 [8-12] | 14.5 [12-17] | 6.3 [5-8] | 72.3 [69-75] | 27.6 | 45 |
| MPS+LGAT | 87.27 | 86.53 | 90.91 | 983/1000 | 9.0 [7-11] | 9.4 [8-11] | 4.0 [3-5] | 36.5 [34-40] | 62.4 | 43 |
| Orbit | 87.48 | 86.62 | 90.14 | 985/1000 | 10.1 [8-12] | 12.3 [10-14] | 6.2 [5-8] | 34.4 [32-37] | 64.6 | 24 |
| AdvTrain | 86.09 | 85.84 | 93.71 | 989/1000 | 4.3 [3-6] | 4.6 [3-6] | 3.3 [2-5] | 6.0 [5-8] | 93.0 | 374 |
| Gate (certified) | 86.91 | 85.69 | 87.15 | 951/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 95.1 | 293 |
| V2 | 88.09 | 87.24 | 90.52 | 992/1000 | 19.6 [17-22] | 22.0 [20-25] | 2.1 [1-3] | 31.9 [29-35] | 67.6 | 321 |
| V2-R | 86.31 | 86.10 | 94.27 | 994/1000 | 1.3 [1-2] | 1.3 [1-2] | 0.6 [0-1] | 1.3 [1-2] | 98.1 | 667 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 340/12 p=1.4e-84 | 445/6 p=3.9e-123 | 306/3 p=9.4e-87 | 573/2 p=2.7e-168 |
| V2 vs MPS | 179/61 p=1.2e-14 | 175/69 p=8.4e-12 | 95/6 p=1.1e-21 | 203/19 p=4.6e-40 |
| V2 vs LGAT | 58/157 p=1e-11 | 82/158 p=1.1e-06 | 52/10 p=5.7e-08 | 448/48 p=2e-82 |
| V2 vs MPS+LGAT | 69/171 p=3.6e-11 | 71/193 p=3.3e-14 | 36/15 p=0.0046 | 102/55 p=0.00022 |
| V2 vs AdvTrain | 32/180 p=3.3e-26 | 34/204 p=9.8e-31 | 27/14 p=0.06 | 45/299 p=3.8e-47 |
| V2 vs V2-R | 1/179 p=2.4e-52 | 1/203 p=1.6e-59 | 0/14 p=0.00012 | 0/300 p=9.8e-91 |

### UNSW-NB15, xgboost

Validation tuning (selection frozen before test):

Reference (undefended) validation: acc 88.35, F1 87.25, recall 88.62.

| free classifier | gate tau | val acc | val F1 | val recall | parity | val ASR FFE20 | val ASR white-box | selected |
|---|---|---|---|---|---|---|---|---|
| undefended | 0.5 | 87.15 | 86.48 | 91.36 | no | 4.3 | 4.3 |  |
| undefended | 0.8 | 88.44 | 87.37 | 88.96 | ok | 30.3 | 32.7 |  |
| undefended | 0.95 | 88.39 | 87.3 | 88.71 | ok | 35.0 | 44.7 |  |
| undefended | 0.99 | 88.36 | 87.26 | 88.64 | ok | 47.0 | 51.0 |  |
| orbit | None | 87.37 | 86.7 | 91.57 | no | 12.5 | 18.5 |  |
| orbit | 0.5 | 86.84 | 86.4 | 92.96 | no | 3.0 | 3.3 |  |
| orbit | 0.8 | 87.52 | 86.9 | 92.01 | no | 7.7 | 9.7 |  |
| orbit | 0.95 | 87.42 | 86.77 | 91.72 | no | 8.1 | 11.4 |  |
| orbit | 0.99 | 87.4 | 86.74 | 91.65 | no | 9.8 | 14.1 |  |
| orbit_light | None | 87.92 | 87.09 | 90.6 | ok | 24.1 | 39.5 |  |
| orbit_light | 0.5 | 87.2 | 86.61 | 92.04 | no | 3.7 | 4.0 |  |
| orbit_light | 0.8 | 87.97 | 87.16 | 90.84 | ok | 14.0 | 27.7 | yes |
| orbit_light | 0.95 | 87.95 | 87.13 | 90.69 | ok | 15.0 | 29.3 |  |
| orbit_light | 0.99 | 87.93 | 87.1 | 90.62 | ok | 17.7 | 32.4 |  |
| adversarial | None | 85.91 | 85.74 | 94.22 | no | 5.4 | 16.2 |  |
| adversarial | 0.5 | 85.72 | 85.7 | 95.1 | no | 2.7 | 3.0 |  |
| adversarial | 0.8 | 86.07 | 85.93 | 94.6 | no | 2.7 | 8.1 |  |
| adversarial | 0.95 | 85.99 | 85.84 | 94.42 | no | 2.4 | 8.8 |  |
| adversarial | 0.99 | 85.96 | 85.8 | 94.33 | no | 3.4 | 10.8 |  |

Test cohort (Real ASR %, Wilson 95% CI, flows the model detects; "retained" = share of the cohort still detected after the attack):

| model | acc | F1 | recall | cohort kept | FFE M=5 | FFE M=20 | CAPGD | White-box | retained (min over attacks) | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| Undefended | 88.51 | 87.32 | 87.99 | 1000/1000 | 49.0 [46-52] | 62.2 [59-65] | 24.4 [22-27] | 90.9 [89-92] | 9.1 | 30 |
| MPS | 87.80 | 86.43 | 86.40 | 963/1000 | 33.0 [30-36] | 33.6 [31-37] | 16.3 [14-19] | 45.3 [42-48] | 52.7 | 65 |
| LGAT | 87.75 | 87.15 | 92.34 | 996/1000 | 9.8 [8-12] | 15.0 [13-17] | 6.2 [5-8] | 81.7 [79-84] | 18.2 | 73 |
| MPS+LGAT | 87.44 | 86.58 | 90.06 | 973/1000 | 7.4 [6-9] | 7.4 [6-9] | 7.3 [6-9] | 32.4 [30-35] | 65.8 | 77 |
| Orbit | 87.79 | 87.03 | 91.07 | 992/1000 | 8.6 [7-10] | 12.3 [10-14] | 3.6 [3-5] | 18.4 [16-21] | 80.9 | 2 |
| AdvTrain | 86.60 | 86.28 | 93.67 | 991/1000 | 5.2 [4-7] | 6.5 [5-8] | 3.7 [3-5] | 14.6 [13-17] | 84.6 | 158 |
| Gate (certified) | 86.74 | 85.47 | 86.70 | 955/1000 | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 0.0 [0-0] | 95.5 | 196 |
| V2 | 88.38 | 87.50 | 90.43 | 998/1000 | 9.6 [8-12] | 12.2 [10-14] | 1.6 [1-3] | 25.4 [23-28] | 74.5 | 214 |
| V2-R | 86.38 | 86.23 | 94.83 | 994/1000 | 2.1 [1-3] | 2.1 [1-3] | 0.6 [0-1] | 2.6 [2-4] | 96.8 | 354 |

Exact McNemar, V2 versus other model (flows detected by both; "a/b" = flows evaded only against the other model / only against V2):

| comparison | FFE M=5 | FFE M=20 | CAPGD | White-box |
|---|---|---|---|---|
| V2 vs Undefended | 396/4 p=8.2e-112 | 501/3 p=8.1e-145 | 226/0 p=1.9e-68 | 654/0 p=2.7e-197 |
| V2 vs MPS | 286/43 p=3.5e-45 | 271/48 p=6.7e-39 | 153/2 p=5.3e-43 | 217/9 p=7e-53 |
| V2 vs LGAT | 65/62 p=0.86 | 100/72 p=0.039 | 51/5 p=1.2e-10 | 588/26 p=1.4e-139 |
| V2 vs MPS+LGAT | 55/60 p=0.71 | 53/84 p=0.01 | 68/3 p=5.1e-17 | 105/22 p=3.7e-14 |
| V2 vs AdvTrain | 30/68 p=0.00016 | 37/89 p=4.1e-06 | 25/4 p=0.0001 | 64/167 p=8.7e-12 |
| V2 vs V2-R | 2/71 p=5.7e-19 | 0/95 p=5e-29 | 2/12 p=0.013 | 0/222 p=3e-67 |


## 4. Reading the tables (honest summary)

* **CIC-IDS2017**: V2 is essentially un-evadable (0.1-0.7 %, all attacks, all libraries) with no accuracy cost (recall rises by 0.07-0.46 points). The gate alone reaches the same level.
* **CSE-CIC-IDS2018**: V2 Real ASR is 8.8 / 7.9 / 8.1 % under the white-box attacker (CatBoost / LightGBM / XGBoost), 0.0-0.3 % under CAPGD and 0.0-8.1 % under FFE, versus 99-100 % for the undefended models and for MPS and LGAT under the white-box attacker. The remaining 8 % are cohort flows the monotone gate does not flag and the free classifier loses under attack; V2 retains 91-92 % of the cohort under every attack. Free classifiers alone (Orbit, AdvTrain) stop FFE and CAPGD to 1-50 % but stay about 80 % evadable by the white-box attacker; the certified gate is what closes that gap. LGAT's FFE M=5 figure is its training attack.
* **UNSW-NB15** (hardest: overlapping classes, about 88 % clean accuracy): at strict parity V2 gives 9.6-22 % FFE and 25-32 % white-box Real ASR for XGBoost and LightGBM (about 5 % for CatBoost), against 49-91 % undefended. It is not better than LGAT against FFE for XGBoost (McNemar p = 0.86 at M=5, 0.039 at M=20, with 65 versus 62 and 100 versus 72 discordant flows) and LGAT is clearly better than V2 against FFE for LightGBM (9.7 versus 19.6 % at M=5). V2 is clearly better against the white-box attacker and CAPGD. Retained recall under attack is only 67-94 %. V2-R (accuracy 2.1-2.4 points lower, recall 3-6 points higher than undefended) reaches 0.5-4.7 % against every attack and retains 95-98 %. On this corpus robustness costs accuracy.
* **Across libraries**: the hardening works for all three libraries; none is intrinsically robust (undefended white-box Real ASR 90-100 % for XGBoost and LightGBM on UNSW-NB15 and CSE-CIC-IDS2018, 60 % for CatBoost on UNSW-NB15; LightGBM and CatBoost on CIC-IDS2017 sit near 34 %, consistent with the earlier LightGBM/CIC-IDS2017 observation). Model-level changes that are not attack-specific: hiding the window, rates and mean length (MPS) helps against rate and duration dilation but not uniformly (XGBoost on CIC-IDS2017: MPS is worse than undefended against FFE M=5, 92.8 versus 69.7 %); monotone constraints plus capacity hold across libraries and are the only change that carries a guarantee.

## 5. Limitations

* The certificate is conditional on the schema threat model. Attackers that reduce packets or bytes, change ports, split or merge flows, or exploit the feature extractor are out of scope; Real ASR counts only schema-feasible evasions, and packet-level replay was not re-run for the hardened models. The 12-feature schema limits what any model sees (attack recall is 77-78 % on CSE-CIC-IDS2018 for every model, undefended included).
* The monotone direction is a modelling assumption (more forward packets, bytes or duration never lowers the score). It removes the ability to learn "short and small is malicious" patterns, which is the clean-accuracy cost seen for the stand-alone gate on CSE-CIC-IDS2018 and UNSW-NB15.
* V2 is only as strong as its residual: flows the gate misses fall back to a hardened free classifier with no guarantee. No attack specific to the augmentation or to the fusion rule, beyond the white-box control, was designed.
* Tuning used a 300-flow validation cohort and a small grid. One seed and one split per cell; Wilson intervals exclude seed and split variance. The white-box attacker is greedy single-feature threshold crossing, not an exact search; a stronger search could raise the residual ASR of the free components (not the gate's).

## 6. Not done

* **FFE-v2 evaluation.** Hardened models are not persisted, so evaluating them against FFE-v2 needs the full retraining (hours on this machine), which did not fit the time limit. The slot is ready: `python experiments/run_defenses_v2.py <dataset> --plugin src.defenses.plugin_ffev2 --extra-attacks ffe_v2` registers `ffe_v2` through `register_attack` (FFE-v2 only needs `predict_proba` and `get_split_thresholds`, which every defended model provides). This path has not been exercised. The gate is certified against FFE-v2 by the same argument as long as FFE-v2 stays inside the schema.
* Random forest, the manipulation-signature detector evaluation, and more than one seed.

## 7. Suggested supplementary paragraph

*Certified monotone gating.* The schema lets an attacker only increase forward packets, forward bytes, duration and header bytes, and fixes port, protocol and the backward direction. A tree ensemble that hides the window, rate and mean-length columns and is constrained to be non-decreasing in the four increase-only columns therefore cannot lower its score under any feasible transformation. We train such a gate (300 trees of depth 8) and OR-fuse it with an augmentation-hardened free classifier, choosing the gate threshold on validation so that clean accuracy, F1 and recall stay within 0.5 point of the undefended model. On CIC-IDS2017 the fused model's Real ASR is at most 0.7 % against FFE (M=5 and 20), CAPGD and a white-box threshold-crossing attacker for XGBoost, LightGBM and CatBoost. On CSE-CIC-IDS2018 it is at most 8.8 %, against 99-100 % for the undefended models and for MPS and LGAT under the white-box attacker. On UNSW-NB15 it is 1.0-32 % at strict parity, and a relaxed variant that gives up about 2 accuracy points reaches 0.5-4.7 %. The residual evasions come from malicious flows that the gate does not flag, which fall back to a classifier without a guarantee.

Files: `src/defenses/` (models, augment, attackers registry, whitebox_generic, recipe, anomaly, plugin_ffev2), `experiments/run_defenses_v2.py`, `experiments/render_defense_v2_tables.py`, `tests/test_defenses_v2.py`, `results/defense_v2_<dataset>.json`.
