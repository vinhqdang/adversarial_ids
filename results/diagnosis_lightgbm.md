# Why FFE evades LightGBM in 35.1% but XGBoost in 99.2% on CIC-IDS2017

## Verdict
Not a bug in LightGBM handling. It is a property of the attack's fixed TCP-window target (64240 B) interacting with an
empty region of the training data. Reproduced exactly (same cohort, seed): LightGBM 35.1%, XGBoost 99.2%.
results/*.json do not need regenerating (they are correct for the attack as specified); the paper needs one explanatory
sentence and, optionally, a variant row.

## Evidence
1. Failures are family-specific (cohort of 1000; LightGBM ASR): PortScan 100, Botnet 100, BruteForce 100, DoS 1.9, DDoS 0.0.
   XGBoost: DoS 98.1, DDoS 100. DoS+DDoS are ~66% of the cohort, which gives the aggregate 35.1%.
2. Thresholds are extracted correctly (LightGBM dump parsed; 13 distinct thresholds on init_fwd_win_byts); the output is
   predict_proba in both models; start scores are identical (median 0.998). Feature importance (gain) is nearly identical
   (totlen_bwd 0.31/0.31, flow_duration 0.29/0.29, tot_bwd 0.16/0.17, init_fwd_win 0.11/0.11). So reliance on
   immutable features does not explain the gap: replacing tot_bwd_pkts+totlen_bwd_pkts by benign medians flips 97% (DoS) of flows
   for both models.
3. Decisive probe (all 33,857 detected DoS test flows, only init_fwd_win_byts set to 64240, which is FFE's window morph):
   XGBoost flips 97.2% to benign, LightGBM 0.9%. DDoS (18,951): 99.96% vs 0.0%. Sweeping the window value: both models
   are malicious at 29200 (the attack-tool value); for 45000-64240 XGBoost says benign (97.2%) and LightGBM malicious
   (0.9%); at 65535 LightGBM says benign 84.4%.
4. Mechanism: in CIC-IDS2017 training data the window takes only a few values: attack flows 29200 (55%), 8192, 1024;
   benign 0 (57%), 8192, 29200, 65535 (5%). Nothing lies in (29200, 65535), so tree thresholds placed in that gap are
   arbitrary. XGBoost has thresholds at 32831 and 65535 (a 64240 flow lands on the 65535-benign side); LightGBM has
   thresholds at 63586 and 65435 and 64240 lands between them, on the malicious side. 64240 is out-of-support for this
   dataset (it is the common Linux/Windows value in other traces) and each model extrapolates differently.
5. Not capacity or binning: LightGBM with num_leaves=64, max_bin=1023, min_child_samples=5, min_data_in_bin=1 (max_bin 4095):
   win=64240 still flips only 0.9-1.2% of DoS (n_estimators=300: 3.1%).
6. Causal test of the hypothesis (experiments/diagnose_lightgbm_target.py, full FFE, same 1000-flow cohort, only the window
   target changed 64240 -> 65535, an in-support benign value):
   LightGBM 35.1% -> 93.7% (DDoS 100, DoS 85.1, queries 120 -> 21.8); XGBoost 99.2% -> 99.2% (queries 14.6 -> 11.0).
   The gap shrinks from 64 points to 5.5 points.

## Residual
LightGBM at 93.7%: remaining DoS failures (14.9% of DoS) not examined further; likely flows needing other thresholds.

## Suggested paper sentences
"FFE evades XGBoost on 99.2% of the detected CIC-IDS2017 cohort but LightGBM on 35.1%. The gap is not due to model
quality or to reliance on immutable features (both models have near-identical gain profiles) but to the window generator:
FFE morphs init_fwd_win_byts towards 64240 B, a value that never occurs in the CIC-IDS2017 training flows, so
the response of each tree ensemble there is determined by where its splits happen to fall in an empty interval
(XGBoost routes 64240 to the benign side for 97% of DoS flows; LightGBM for 0.9%). Using an in-support benign window
(65535 B) raises LightGBM's evasion rate to 93.7% (XGBoost unchanged at 99.2%). Evasion success against a given tree
ensemble therefore depends on whether the attacker's target value is within the training support."

## Scripts (experiments/)
diagnose_lightgbm_quick.py (importance, per-family, trajectories), diagnose_lightgbm_probe.py (single-move probes),
diagnose_lightgbm_win.py (window thresholds, sweep, capacity ablations), diagnose_lightgbm_target.py (causal rerun).
src/ was not modified.
