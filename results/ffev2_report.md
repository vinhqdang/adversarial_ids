# FFE-v2: a stronger score-only attack on tree-ensemble NIDS

Code: `src/attacks/ffe_v2.py` (FFE-v1 in `flow_field_evasion.py` is untouched), `src/validation/packet_replay.py` (new `resegment`, `plan_from_ledger`, `replay_plan`), `src/attacks/whitebox_tree.py` (new `RawMarginModel`: the project-in-loop control for LightGBM and CatBoost), `experiments/{diagnose_ffev2,tune_ffev2,run_ffev2,run_ffev2_replay,report_ffev2}.py`, tests `tests/test_ffe_v2.py` (110 tests). Raw numbers: `results/ffev2_<dataset>.json` (per-flow outcomes, same per-attack schema as `results/full_<dataset>.json`, under `models.xgboost|lightgbm|catboost`), `results/ffev2_diagnosis.json`, `results/ffev2_tuning.json`, `results/ffev2_config.json` (the frozen configuration), `results/ffev2_replay.json`, `results/ffev2_tables.md`.

## Summary

* FFE-v2 (score access only, no tree access) has a Real ASR at least that of the projected white-box control and of FFE-v1 on every one of the nine (corpus, target) cells; it is significantly higher than the control on four (UNSW-NB15 x3, CIC-IDS2017 LightGBM) and than v1 on seven (exact McNemar): UNSW-NB15 XGBoost 98.3 vs control 90.2 vs v1 57.6; UNSW-NB15 LightGBM 96.7 (99.2 with snapping) vs 88.1 vs 62.4; UNSW-NB15 CatBoost 97.5 vs 56.8 vs 70.1; CIC-IDS2017 LightGBM 96.0 vs 36.6 vs 35.1; CSE-CIC-IDS2018 CatBoost 99.5 vs 99.5 vs 59.3. Ties with the control (identical flows): CSE-CIC-IDS2018 x3 (99.5; the remaining 5 DDoS flows are unreachable), CIC-IDS2017 XGBoost (100.0) and CatBoost (36.4).
* One cell is a ceiling, not a win: CIC-IDS2017 CatBoost (36.4% for FFE-v2 and the control, 34.8% for v1). The decision for DDoS/DoS flows is driven by attacker-immutable features (tot_bwd_pkts, totlen_bwd_pkts, dst_port: 77% of CatBoost's importance); a 20,000-sample multi-start search over all generators jointly crosses the boundary for 0% of DDoS and about 5% of DoS flows. No method restricted to the schema's mutable features can do much better; the number describes CatBoost's reliance on immutable fields, not attack strength.
* Cost is not uniformly lower. FFE-v2 uses about 2x the queries of FFE-v1 on the easy corpora (30 vs 16.5) and, on UNSW-NB15, a larger median cost than FFE-v1 on the flows both evade (6.7 vs 5.8 XGBoost, 14.6 vs 7.5 LightGBM, 29.1 vs 9.8 CatBoost), because v1 only succeeds on the cheap flows. Against the control it is cheaper on the flows both evade on UNSW-NB15 XGBoost and LightGBM (paired medians 32.5 vs 213 and 35.8 vs 49) but not on CatBoost (22.1 vs 10.5). The control reads the trees and needs 2 queries; FFE-v2 needs about 30 to 40 score queries (109 on CIC-IDS2017 CatBoost, where it exhausts its budget on unreachable flows).
* Snapping (Tier 1, reads thresholds) adds up to 2.5 points on UNSW-NB15 (LightGBM 96.7 to 99.2) and nothing elsewhere; the score-only variant is the headline.

## 1. Diagnosis of FFE-v1 (validation split, `results/ffev2_diagnosis.json`)

FFE-v1 on the validation cohorts (N about 1000): CSE-CIC-IDS2018 99.7 / 99.7 / 58.3 (XGBoost / LightGBM / CatBoost), CIC-IDS2017 99.2 / 35.1 / 34.2, UNSW-NB15 60.9 / 64.4 / 72.8.

**UNSW-NB15.** On the 391 flows where v1 fails against XGBoost, the projected control evades 326. Its successful moves: 99% raise `tot_fwd_pkts` (median +213 packets), 82% do so without changing `totlen_fwd` (re-segmentation of the same payload; the projection fills the header sum at 20 B per extra packet), only 7% change the duration, none the window (every TCP flow in UNSW-NB15 advertises 0 or 255 B, so the window never matters there). The four v1 generators cannot make this move: chaff packets add 28 to 52 B of payload each, so the forward mean is pulled towards 40 B instead of down to 3 to 5 B, and the chaff step is capped at 50% of the packet count per iteration within a 20-iteration budget; padding raises the mean; dilation only lowers the rates. The score then plateaus (many splits on `tot_fwd_pkts`, `fwd_pkt_len_mean` and `flow_pkts_s` must be crossed jointly) and v1 spends its whole budget (89 to 94 queries). Reachability (20,000 random joint draws over time, pad, chaff, seg and window per failed flow; 80 failed flows per cell): 100% of the failed XGBoost and LightGBM flows and 97.5% of the CatBoost ones have a combination that crosses the boundary, so these failures are algorithmic, not structural. (For LightGBM and CatBoost the control's moves are mixed: 30% and 19% pure segmentation-like, 69% and 81% with added payload.)

**CIC-IDS2017.** v1's window target 64240 B is out of support (0.08% of benign and 0.015% of attack TCP flows have it; 11.5% of benign flows advertise 65535). XGBoost: the 8 flows v1 misses are all fixed by a window move. LightGBM: 90% of the sampled failed flows are reachable (the unreachable ones are DoS flows) and the control wins only 17 of 649 failures, all through window moves, so the control is itself limited to 36.6%. CatBoost: only 3.75% of the sampled failed flows are reachable (DDoS 0 of 37, DoS 0 of 40 sampled): the immutable-feature ceiling. CatBoost's 22 window borders stop at 32,851.5, so 45,000, 64,240 and 65,535 fall in one bin and the window generator is inert for it on this corpus.

**CSE-CIC-IDS2018 CatBoost.** v1 evades 58.3%; the control evades 411 of the 414 failures with a pure window move (every one of its moves is a window change). v1's single target 64240 B lies on the wrong side of CatBoost's borders; the in-support benign values (8192, 65535, 219, 237, 279, 256) do not.

**Where v1 stalls:** (i) one hard-coded window target; (ii) one fixed magnitude per generator and iteration, too small to cross plateaus or too large for cheap flows; (iii) no generator that lowers the forward mean or raises the packet count without adding payload; (iv) progress measured on the saturating probability (float32 probabilities near 1 give tied probes, which sends v1 to its fallback); (v) the unit cost applied twice in the generator ranking.

## 2. FFE-v2 design

Every move is a packet-level action under the same schema bounds; every candidate is projected with the closed-form operator (`project_batch_with_origin` is tested equal to `project_network_invariants`) and every returned flow passes `verify_physical_feasibility`.

1. **Sixth generator `seg` (MSS clamp / re-segmentation).** The same forward payload is carried by K more, smaller packets: `tot_fwd_pkts += K`, `fwd_header_len += K*h0`, `totlen_fwd` and the duration unchanged (new packets follow their parent back to back; at least 1 B of payload per packet, K <= totlen_fwd - tot_fwd_pkts). The cost model prices each added packet as a chaff packet (`cost.py` unchanged). Packet-level semantics: `packet_replay.resegment` halves the largest forward packet K times; tested to leave payload, duration, first forward packet and window unchanged. An optional `hdr` generator (TCP option padding) exists but was not selected (no validation gain; the cost model does not price header bytes).
2. **Multi-scale probing.** Each generator is probed on a geometric ladder (ratio 6) spanning its whole feasible range: delay 1 ms to the 110 s cap, padding 16 B to the MTU headroom, 1 to 4096 chaff or segmented packets. Identical projected candidates are probed once.
3. **Data-driven window targets.** The window generator tries the 6 most frequent benign TCP windows of the attacker's benign sample (training split), in both directions, instead of 64240. A variant with generic OS defaults (8192, 14600, 29200, 64240, 65535; no training data) is reported.
4. **Log-odds progress.** Progress is the log-odds of the returned score (a monotone transform of the same score), not the saturating probability.
5. **Selection.** If any probe crosses 0.5, the cheapest crossing probe under the paper's operational cost is taken and the search ends. Otherwise the probe with the largest score decrease per unit of marginal operational cost, with a relax-cost fallback (pure score minimisation after 3 iterations or when no probe has a cost-efficient decrease). Budget: 8 iterations.
6. **Cost refinement.** After a crossing, 4 bisection probes shrink the magnitude of the crossing move; each probe is a scored flow, so success is never assumed. This roughly halves the validation geometric-mean cost (1.07e-2 to 6.3e-3) for about 2 queries.
7. **Tier 1 snapping (optional).** When no probe helps, threshold crossings of all mutable features (8 per feature and direction, from the nearest to the farthest threshold) are converted to closed-form generator magnitudes (including exact inversions of the rate and mean thresholds through delay, padding, chaff and segmentation), projected, scored, and the cheapest crossing or best candidate is taken. This is the white-box ingredient, the only use of the trees' thresholds.
8. **Tried and not kept:** beam search (width 2 or 4: +0.1 mean ASR for 15 to 45 more queries), restarts (subsumed by the relax fallback), compound pair moves (iteration over the ladder already composes moves), `hdr`.

Query accounting: every scored flow counts as one query, including probes scored in a batch. Only snapping reads thresholds.

## 3. Protocol (no leakage)

Models are trained as in `run_full_benchmark.py` (60 trees, depth 6); CatBoost as in `run_transferability.py` (60 iterations, depth 6). All design choices and hyper-parameters were tuned on cohorts of up to 1000 detected malicious flows drawn from the 10% VALIDATION split with the benchmark's cohort procedure (detected by XGBoost and LightGBM; for CatBoost the part CatBoost itself detects). The configuration was then frozen (`results/ffev2_config.json`) and evaluated once on the benchmark's 1000-flow TEST cohorts (the identical flows of `results/full_*.json`, verified by the runner). FFE-v1 rows for XGBoost and LightGBM and CAPGD (XGBoost only) are the saved benchmark outcomes; the XGBoost control is `results/whitebox_*.json`; v1 on CatBoost and the control on LightGBM and CatBoost were run for this report. The CatBoost cohort is the benchmark cohort restricted to flows CatBoost detects (N = 1000 / 997 / 983), so every success is a genuine evasion of that model.

Frozen configuration: generators time, pad, chaff, seg, win; ladder ratio 6; 6 benign window values; 8 iterations; relax after 3; 4 refinement probes; the snapping variant adds `snap_per_feature = 8`.

### Validation-split tuning table (Real ASR %)

Columns: corpus (cse2018, cic2017, unsw) by target (xgb, lgb, cat). Rows from `base` to `plus_hdr` come from the first stage (ladder of 6 levels from a flow-relative first step, ratio 4; each row varies one factor: `ratio=`, `levels=`, `beam=`, `lambda=`, `relax=`, `win_k=`, `iter=`, `snap_pf=`, `no_*`, `plus_hdr`); `span ...` rows use the final span-based ladder; `refine=` and `A ratio6` rows the final design; the `frozen` rows are the two configurations evaluated on test, next to FFE-v1 and the control on the same validation cohorts.

| Configuration | cse2 xgb | cse2 lig | cse2 cat | cic2 xgb | cic2 lig | cic2 cat | unsw xgb | unsw lig | unsw cat | mean | queries | geo-mean median cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.4 | 91.3 | 39.0 | 8.98e-03 |
| ratio=2 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 68.7 | 70.5 | 69.2 | 82.1 | 57.0 | 5.71e-03 |
| ratio=3 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 93.4 | 93.1 | 94.5 | 90.1 | 42.4 | 6.80e-03 |
| ratio=6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.9 | 97.8 | 98.1 | 91.6 | 36.4 | 1.14e-02 |
| ratio=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.5 | 96.3 | 98.8 | 91.4 | 34.8 | 9.68e-03 |
| levels=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 76.8 | 74.8 | 81.5 | 84.8 | 38.7 | 5.86e-03 |
| levels=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.3 | 98.1 | 98.7 | 91.6 | 43.2 | 8.98e-03 |
| levels=10 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.3 | 98.3 | 98.6 | 91.6 | 46.7 | 8.98e-03 |
| beam=2 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.0 | 98.4 | 98.1 | 91.5 | 54.2 | 8.98e-03 |
| beam=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.5 | 99.0 | 98.6 | 91.7 | 83.8 | 8.98e-03 |
| lambda=0.0 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 96.7 | 97.7 | 98.9 | 91.5 | 36.9 | 8.98e-03 |
| lambda=0.5 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.8 | 97.6 | 97.7 | 91.4 | 37.7 | 8.98e-03 |
| lambda=2.0 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.6 | 97.5 | 97.5 | 91.3 | 39.6 | 8.98e-03 |
| relax=0 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 96.7 | 97.7 | 98.9 | 91.5 | 36.9 | 8.98e-03 |
| relax=1 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 96.6 | 97.4 | 97.9 | 91.3 | 37.1 | 8.98e-03 |
| relax=6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.5 | 91.3 | 41.5 | 8.98e-03 |
| relax=99 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.5 | 91.3 | 41.5 | 8.98e-03 |
| win_k=0 | 60.8 | 52.2 | 35.3 | 35.0 | 35.0 | 35.4 | 96.7 | 97.5 | 97.4 | 60.6 | 56.5 | 1.82e-02 |
| win_k=3 | 99.7 | 99.7 | 99.7 | 99.6 | 94.9 | 36.1 | 96.7 | 97.5 | 97.4 | 91.2 | 34.0 | 1.04e-02 |
| win_k=12 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.4 | 91.3 | 47.5 | 7.05e-03 |
| iter=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 95.0 | 97.4 | 96.8 | 91.0 | 34.7 | 8.98e-03 |
| iter=12 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.6 | 91.3 | 39.0 | 8.98e-03 |
| iter=20 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.5 | 97.6 | 91.3 | 39.0 | 8.98e-03 |
| snap_pf=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.2 | 98.6 | 98.5 | 91.6 | 41.6 | 8.98e-03 |
| snap_pf=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.4 | 97.4 | 98.7 | 98.5 | 91.7 | 43.9 | 8.98e-03 |
| snap_pf=16 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.4 | 98.1 | 98.8 | 98.5 | 91.7 | 54.1 | 8.98e-03 |
| no_seg | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 96.6 | 96.9 | 97.2 | 91.2 | 32.2 | 9.10e-03 |
| no_chaff | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 95.4 | 59.9 | 63.9 | 83.3 | 39.5 | 6.27e-03 |
| no_time | 99.7 | 99.7 | 99.7 | 100.0 | 94.8 | 36.3 | 95.3 | 93.3 | 98.7 | 90.8 | 30.4 | 1.51e-02 |
| no_pad | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 94.5 | 97.0 | 97.0 | 91.0 | 30.5 | 9.10e-03 |
| plus_hdr | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.9 | 97.6 | 97.5 | 91.4 | 45.0 | 8.98e-03 |
| span ratio=3 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.2 | 98.6 | 98.8 | 91.6 | 55.2 | 9.28e-03 |
| span ratio=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 98.0 | 98.4 | 91.6 | 46.4 | 9.22e-03 |
| span ratio=6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 39.4 | 1.07e-02 |
| span ratio=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 95.0 | 98.6 | 91.2 | 36.3 | 9.34e-03 |
| span ratio=12 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 94.7 | 98.0 | 91.1 | 33.1 | 1.73e-02 |
| span ratio=16 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.2 | 96.1 | 98.0 | 91.3 | 31.3 | 1.81e-02 |
| span pad_min=1 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.4 | 97.3 | 98.4 | 91.6 | 40.6 | 1.07e-02 |
| span pad_min=16 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 39.4 | 1.07e-02 |
| span pad_min=100 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.3 | 91.6 | 38.0 | 1.07e-02 |
| span time_min=100 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.7 | 97.2 | 98.0 | 91.5 | 41.4 | 1.10e-02 |
| span time_min=10000 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.4 | 97.9 | 91.5 | 39.1 | 8.27e-03 |
| span pkt_max=512 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 96.7 | 97.4 | 97.3 | 91.3 | 38.5 | 1.07e-02 |
| span pkt_max=16384 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 98.0 | 98.4 | 91.6 | 41.0 | 1.07e-02 |
| A ratio6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 39.4 | 1.07e-02 |
| refine=3 ratio=3 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.2 | 98.6 | 98.8 | 91.6 | 56.7 | 5.86e-03 |
| refine=6 ratio=3 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.2 | 98.6 | 98.8 | 91.6 | 58.1 | 5.62e-03 |
| refine=3 ratio=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 98.0 | 98.4 | 91.6 | 47.9 | 6.29e-03 |
| refine=6 ratio=4 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 98.0 | 98.4 | 91.6 | 49.2 | 6.04e-03 |
| refine=3 ratio=6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 40.8 | 6.59e-03 |
| refine=6 ratio=6 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 42.1 | 6.05e-03 |
| refine=3 ratio=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 95.0 | 98.6 | 91.2 | 37.7 | 6.51e-03 |
| refine=6 ratio=8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 97.4 | 95.0 | 98.6 | 91.2 | 39.1 | 6.35e-03 |
| refine=4 pkt16k | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 98.0 | 98.4 | 91.6 | 42.9 | 6.34e-03 |
| refine=4 beam2 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.9 | 98.6 | 97.9 | 91.7 | 56.5 | 6.19e-03 |
| refine=4 snap8 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.7 | 98.6 | 98.7 | 91.8 | 46.1 | 6.33e-03 |
| refine=4 snap16 beam2 | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.4 | 99.1 | 99.6 | 98.5 | 91.9 | 80.6 | 6.27e-03 |
| refine=4 plus_hdr | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.8 | 98.1 | 91.6 | 47.2 | 6.19e-03 |
| FFE-v2 frozen (score-only) | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.6 | 41.3 | 6.27e-03 |
| FFE-v2 frozen (+snapping) | 99.7 | 99.7 | 99.7 | 100.0 | 94.9 | 36.3 | 98.7 | 98.6 | 98.7 | 91.8 | 46.1 | 6.33e-03 |
| FFE-v1 | 99.7 | 99.7 | 58.3 | 99.2 | 35.1 | 34.2 | 60.9 | 64.4 | 72.8 | 69.4 | 59.7 | 8.21e-02 |
| White-box control (project in loop) | 99.7 | 99.7 | 99.7 | 100.0 | 36.8 | 36.3 | 91.4 | 87.3 | 57.8 | 78.7 | 2.0 | 1.84e-02 |
| FFE-v2 generic windows | 99.7 | 99.7 | 99.7 | 99.6 | 94.9 | 36.3 | 98.2 | 97.7 | 98.2 | 91.5 | 40.6 | 8.69e-03 |

Reading the table: the window generator is decisive (`win_k=0` drops CSE-CIC-IDS2018 to 35 to 61% and CIC-IDS2017 XGBoost to 35%); chaff matters on UNSW-NB15 for LightGBM and CatBoost (`no_chaff` 60 and 64%); a ladder that is too short or too fine fails on UNSW-NB15 (`ratio=2`, `levels=4`: 69 to 82%); `seg` alone is worth little on validation once chaff and padding exist (`no_seg` 96.6 / 96.9 / 97.2 vs 96.7 / 97.5 / 97.4 on UNSW-NB15) but the white-box control relies on it there and 16 to 39% of FFE-v2's successful UNSW-NB15 flows use it; CIC-IDS2017 LightGBM (94.9) and CatBoost (36.3) sit at the reachability ceiling in every configuration.

## 4. Final test results (frozen configuration, single evaluation)

Real ASR (evades and schema-feasible), %, queries, cost, McNemar, generator usage and per-family results:

| Corpus | Target | N | FFE-v1 | Control | CAPGD | FFE-v2 score-only | FFE-v2 + snapping |
|---|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 1000 | 99.5 | 99.5 | 88.5 | 99.5 | 99.5 |
| CSE-CIC-IDS2018 | lightgbm | 1000 | 99.5 | 99.5 | n/a | 99.5 | 99.5 |
| CSE-CIC-IDS2018 | catboost | 1000 | 59.3 | 99.5 | n/a | 99.5 | 99.5 |
| CIC-IDS2017 | xgboost | 1000 | 99.2 | 100.0 | 89.5 | 100.0 | 100.0 |
| CIC-IDS2017 | lightgbm | 1000 | 35.1 | 36.6 | n/a | 96.0 | 96.0 |
| CIC-IDS2017 | catboost | 997 | 34.8 | 36.4 | n/a | 36.4 | 36.4 |
| UNSW-NB15 | xgboost | 1000 | 57.6 | 90.2 | 21.1 | 98.3 | 98.5 |
| UNSW-NB15 | lightgbm | 1000 | 62.4 | 88.1 | n/a | 96.7 | 99.2 |
| UNSW-NB15 | catboost | 983 | 70.1 | 56.8 | n/a | 97.5 | 97.7 |

Queries (mean per flow) and median operational cost of the successful flows, and median cost on the flows both methods evade (paired):

| Corpus | Target | q v1 | q control | q v2 | q v2+snap | med cost v1 | med cost control | med cost v2 | paired med v2 / v1 (n) | paired med v2 / control (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 16.5 | 2.0 | 30.3 | 30.8 | 0.00057 | 0.000495 | 3.02e-05 | 3.02e-05 / 0.00057 (995) | 3.02e-05 / 0.000495 (995) |
| CSE-CIC-IDS2018 | lightgbm | 20.2 | 2.0 | 31.1 | 31.6 | 0.000977 | 0.000496 | 2.42e-05 | 2.42e-05 / 0.000977 (995) | 2.42e-05 / 0.000496 (995) |
| CSE-CIC-IDS2018 | catboost | 52.5 | 2.0 | 29.1 | 29.2 | 0.000977 | 0.000285 | 0.000285 | 0.000121 / 0.000977 (593) | 0.000285 / 0.000285 (995) |
| CIC-IDS2017 | xgboost | 14.6 | 2.0 | 28.3 | 28.3 | 0.000855 | 0.000376 | 0.000554 | 0.000554 / 0.000855 (992) | 0.000554 / 0.000376 (1000) |
| CIC-IDS2017 | lightgbm | 120.1 | 2.0 | 32.4 | 36.5 | 1.62 | 1.56e-05 | 0.000554 | 1.17e-05 / 1.62 (351) | 1.17e-05 / 1.56e-05 (366) |
| CIC-IDS2017 | catboost | 76.3 | 2.0 | 109.0 | 140.4 | 0.5 | 0.00671 | 3.22e-06 | 3.22e-06 / 0.5 (347) | 3.22e-06 / 0.00671 (363) |
| UNSW-NB15 | xgboost | 93.7 | 2.0 | 36.2 | 37.9 | 5.61 | 213 | 40 | 6.72 / 5.77 (571) | 32.5 / 213 (894) |
| UNSW-NB15 | lightgbm | 89.0 | 2.0 | 38.3 | 41.5 | 7.49 | 49 | 50.4 | 14.6 / 7.49 (618) | 35.8 / 49 (860) |
| UNSW-NB15 | catboost | 67.9 | 2.0 | 38.0 | 42.3 | 9.8 | 10.5 | 56 | 29.1 / 9.78 (688) | 22.1 / 10.5 (558) |

Exact McNemar (Real ASR, same flows), FFE-v2 score-only versus each method: v2-only / other-only / p.

| Corpus | Target | vs v1 | vs control | vs CAPGD | vs v2+snap |
|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 0/0/1 | 0/0/1 | 110/0/1.5e-33 | 0/0/1 |
| CSE-CIC-IDS2018 | lightgbm | 0/0/1 | 0/0/1 | n/a | 0/0/1 |
| CSE-CIC-IDS2018 | catboost | 402/0/1.9e-121 | 0/0/1 | n/a | 0/0/1 |
| CIC-IDS2017 | xgboost | 8/0/0.0078 | 0/0/1 | 105/0/4.9e-32 | 0/0/1 |
| CIC-IDS2017 | lightgbm | 609/0/9.4e-184 | 594/0/3.1e-179 | n/a | 0/0/1 |
| CIC-IDS2017 | catboost | 16/0/3.1e-05 | 0/0/1 | n/a | 0/0/1 |
| UNSW-NB15 | xgboost | 412/5/6.1e-115 | 89/8/2e-18 | 772/0/8.1e-233 | 0/2/0.5 |
| UNSW-NB15 | lightgbm | 349/6/7.4e-95 | 107/21/4.5e-15 | n/a | 0/25/6e-08 |
| UNSW-NB15 | catboost | 270/1/1.4e-79 | 400/0/7.7e-121 | n/a | 0/2/0.5 |

Clean accuracy: CSE-CIC-IDS2018 97.30/97.28/97.26; CIC-IDS2017 99.96/99.96/99.85; UNSW-NB15 88.51/88.47/86.67 (xgboost/lightgbm/catboost).

Generator usage of FFE-v2 successful flows (share of flows using each; score-only):

| Corpus | Target | time | pad | chaff | seg | win |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 31% | 0% | 0% | 0% | 69% |
| CSE-CIC-IDS2018 | lightgbm | 49% | 0% | 2% | 0% | 49% |
| CSE-CIC-IDS2018 | catboost | 2% | 0% | 0% | 0% | 98% |
| CIC-IDS2017 | xgboost | 12% | 0% | 0% | 0% | 87% |
| CIC-IDS2017 | lightgbm | 14% | 1% | 0% | 1% | 87% |
| CIC-IDS2017 | catboost | 79% | 0% | 6% | 0% | 15% |
| UNSW-NB15 | xgboost | 14% | 13% | 37% | 39% | 0% |
| UNSW-NB15 | lightgbm | 12% | 14% | 64% | 16% | 0% |
| UNSW-NB15 | catboost | 11% | 7% | 61% | 24% | 0% |

Per-family Real ASR of FFE-v2 (score-only):

* CSE-CIC-IDS2018 / xgboost: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CSE-CIC-IDS2018 / lightgbm: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CSE-CIC-IDS2018 / catboost: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CIC-IDS2017 / xgboost: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 100.0 (n=235), DoS 100.0 (n=422), PortScan 100.0 (n=322)
* CIC-IDS2017 / lightgbm: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 100.0 (n=235), DoS 90.5 (n=422), PortScan 100.0 (n=322)
* CIC-IDS2017 / catboost: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 0.0 (n=235), DoS 5.0 (n=420), PortScan 100.0 (n=321)
* UNSW-NB15 / xgboost: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 93.8 (n=80), Exploits 99.8 (n=504), Fuzzers 95.9 (n=219), Generic 90.9 (n=22), Reconnaissance 100.0 (n=143), Shellcode 100.0 (n=19), Worms 100.0 (n=6)
* UNSW-NB15 / lightgbm: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 95.0 (n=80), Exploits 98.0 (n=504), Fuzzers 94.1 (n=219), Generic 100.0 (n=22), Reconnaissance 96.5 (n=143), Shellcode 100.0 (n=19), Worms 83.3 (n=6)
* UNSW-NB15 / catboost: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 97.4 (n=77), Exploits 97.2 (n=497), Fuzzers 98.1 (n=213), Generic 100.0 (n=21), Reconnaissance 96.5 (n=143), Shellcode 100.0 (n=19), Worms 100.0 (n=6)

FFE-v2 with snapping equals the score-only variant within 2 flows on eight cells and is better only on UNSW-NB15 LightGBM (+25 flows, p = 6e-8); queries rise by 0 to 5 on most cells and by 31 on CIC-IDS2017 CatBoost.

Without training data (generic window set, validation): CSE 99.7 / 99.7 / 99.7, CIC 99.6 / 94.9 / 36.3, UNSW 98.2 / 97.7 / 98.2 (`FFE-v2 generic windows`), so the data-driven window set is worth at most 0.4 points and the attack does not need training data.

## 5. CatBoost as a first-class target

CatBoost (60 oblivious trees, depth 6) is attacked directly with score probes; snapping reads its borders through `get_split_thresholds`. Transfer of FFE crafted on other models reached 24.0% earlier; the direct attack gives 99.5 / 36.4 / 97.5% (CSE-CIC-IDS2018 / CIC-IDS2017 / UNSW-NB15) against 59.3 / 34.8 / 70.1 for FFE-v1 and 99.5 / 36.4 / 56.8 for the control. Why v1 and transfer fail: (i) CatBoost's window borders differ from XGBoost's, so 64240 B lands in a different leaf (CSE-CIC-IDS2018: the control evades 411 of 414 v1 failures with a window move alone); on CIC-IDS2017 its largest window border is 32,851.5, so the window is inert there; (ii) on CIC-IDS2017 DDoS/DoS CatBoost leans on immutable fields (gain importance: totlen_bwd_pkts 49.7, tot_bwd_pkts 16.6, dst_port 11.1, window 3.9) and those flows are unreachable; (iii) on UNSW-NB15 it needs packet-count moves that only chaff or segmentation provide. The control for CatBoost (and LightGBM) is a generic greedy threshold-crossing search scoring candidates with the model's free raw margin (`RawMarginModel`), not a port of LT-Attack; its 56.8% on UNSW-NB15 CatBoost may reflect single-feature greedy steps stalling on oblivious trees (one shared split per level), which we did not test.

## 6. Packet-level replay of FFE-v2 (`results/ffev2_replay.json`; XGBoost, first 300 test flows per corpus)

The integer action ledger of each successful flow (delay, chaff, re-segmented packets, padding, window) is applied to a synthetic packet trace, the 12 features are re-extracted, feasibility is re-checked and the target re-scored. Under the `schema` and `empty_compensated` conventions (chaff counted as in the schema, see `results/packet_replay.md`) 100% of the replayed flows reproduce the schema vector within 1% on all 12 features (including the 118 UNSW-NB15 flows that use segmentation, with no unmet splits or padding), 100% stay feasible and 100% stay evasive (replayed ASR of all 300: 99.3 / 100 / 98.3%). Under `empty_literal` (chaff packets carry no payload, so the bytes the schema credits to chaff are not sent): CSE-CIC-IDS2018 and CIC-IDS2017 unchanged, UNSW-NB15 retains 69.2% of its evasions (replayed ASR 68.0%), the same accounting sensitivity found for v1. The header-length mismatch of v1's snapping does not arise because the ledger, not the vector difference, drives the replay. The limits of `packet_replay.md` apply (synthetic traces, no TCP state machine). Segmentation also assumes the payload can be split into pieces of at least one byte; zero-payload packets cannot be split, which the schema cannot verify from aggregates (the synthetic replay satisfied it).

## 7. Limitations

* Feasibility is verified at the 12-feature schema level and by a synthetic replay. Segmentation semantics (sender MSS/TSO, retransmission, the server's reaction) are not tested on a real stack, and the cost model prices a re-segmented packet like a chaff packet.
* The window set uses the attacker's benign sample (training split); the generic variant needs none and loses at most 0.4 points on validation.
* FFE-v2 uses about 2x v1's queries on the easy corpora and 36 to 38 on UNSW-NB15 (v1: 68 to 94); on unreachable flows it uses the whole 8-iteration budget (109 queries on CIC-IDS2017 CatBoost). CAPGD (3 to 16 queries) and the white-box control (2) are far cheaper in queries.
* On UNSW-NB15 FFE-v2's cost is higher than v1's on v1's easy successes (paired median 6.7 / 14.6 / 29.1 vs 5.8 / 7.5 / 9.8) and its absolute median (40 to 56) is dominated by tens to hundreds of re-segmented or chaff packets priced at 1 each; the advantage over the control is on the flows both evade, not uniform (CatBoost: 22.1 vs 10.5).
* CIC-IDS2017 CatBoost, about 10% of CIC-IDS2017 DoS on LightGBM and 5 CSE-CIC-IDS2018 DDoS flows on every model are not reachable with the four mutable controls; reachability was measured by random multi-start search, which cannot prove unreachability.
* The LightGBM and CatBoost controls are our greedy variant, not LT-Attack; CAPGD was run on XGBoost only. Tuning used one validation cohort per corpus and one seed; the test cohorts are single draws, so differences of a few flows carry no information. Models were trained once (seed 42); there is no retraining variance.
