# HANDOFF: "Steering Through the Trees" (FFE), state of work

Repo: `vinhqdang/adversarial_ids`, branch `main` only (owner's rules: push to `main`, no other branch, no AI attribution anywhere, git identity `Vinh <dqvinh87@gmail.com>`).
Paper: `paper/` (LaTeX, `elsarticle`), author Quang-Vinh Dang. Python: conda env `py313` (call `.../envs/py313/python.exe` directly; `conda run` is broken on this machine). GPU: RTX 5000 Ada, 16 GB, 32 CPU cores. **Import xgboost/lightgbm before torch** (reverse order segfaults).

## 1. What changed and why

The earlier results came from **12,000-flow samples** (byte-range slices of a few raw files, subsampled, silent fallback to synthetic data). They were not full-data results and several headline claims did not survive the full-data rerun. Everything was rebuilt:

* `src/datasets/full_data.py`: reads every raw file (`data/raw/`, git-ignored; `download_raw()` fetches them), maps to the 12-feature schema, drops invalid records, **projects onto the flow manifold, then** collapses exact duplicates (same 12 features and label). Records with identical features but different labels are **kept** and counted. UDP flows are kept (CICFlowMeter window -1 becomes 0). CIC-IDS2017 "Attempted" probes are dropped and "Infiltration - Portscan" is mapped to PortScan. UNSW-NB15 zero-duration flows are dropped. Cache in `data/full/*.parquet` (git-ignored; `*_stats.json` tracked).
* Kept flows: CSE-CIC-IDS2018 4,968,389; CIC-IDS2017 1,658,869; UNSW-NB15 123,508 (official train+test partition only, 257,673 records, not the 2.54 M capture). The CSE-CIC-IDS2018 distribution is capped at 1,048,575 rows per file.
* Code review by independent agents found and we fixed: UDP flows were silently dropped from CSE-CIC-IDS2018; FFE's cost was normalised L1 while baselines used schema-weighted cost (the old "7,500x lower cost" was a unit mix-up); FFE silently had no TSBS against CatBoost; transfer was not conditioned on all models detecting the clean flow; the MPS defense zeroed columns (degenerate attack start state) instead of hiding them from the model; cohorts differed between XGBoost and LightGBM.
* `src/attacks/cost.py`: the paper's operational cost (1e-4/s delay, 1e-2/byte padding, 1/chaff packet, 1e-3 window) applied to every attack's output. `src/attacks/modern_baselines.py`: CAPGD and NIDSGAN-style generator (our re-implementations).
* `experiments/`: `run_full_benchmark.py`, `run_seeds.py` (5 seeds), `run_ablations.py`, `run_transferability.py`, `run_defenses.py`, `summarize_results.py` (generates every table in `paper/tables/` and `results/summary.json`: exact McNemar with Holm correction, XGBoost root splits, efficiency ratios, seed wins).
* `paper/references.bib`: 93 entries (five uncited arXiv-only ones were dropped), 34 new 2021-2026 references verified against Crossref/arXiv; four wrong existing entries corrected; published versions with DOI preferred over arXiv. Entries without DOI because none exists: `ke2017lightgbm`, `ring2017flow` (original CIDDS-001 paper), `caratheodory1918vorlesungen`.

## 2. Results (XGBoost, Real ASR = evades and schema-feasible)

FFE 99.5 / 99.2 / 57.6 % on CSE-CIC-IDS2018 / CIC-IDS2017 / UNSW-NB15 with 16.5 / 14.6 / 93.7 queries; clean accuracy 97.30 / 99.96 / 88.51 %.

* Significantly better than all seven baselines on every corpus (Holm-adjusted McNemar, largest p 3.7e-11). CAPGD: 88.5 / 89.5 / 21.1; FFE ahead in 14 of 15 retrained runs (one tie).
* The full FFE reads split thresholds only for TSBS; **FFE without TSBS needs only the target's soft score** and still beats every baseline (99.5 / 97.5 / 54.6). Surrogate attacks need far fewer target queries (CAPGD 3.2-16.2, generator 2.4-4.5): do not claim query efficiency against them. Five-seed means: FFE 99.7 / 97.3 / 58.3 vs CAPGD 97.7 / 82.4 / 31.5 (the CSE-CIC-IDS2018 gap shrinks to about 2 points).
* **TSBS adds at most 3 points of Real ASR and up to 1.8x queries** (ablation). The paper no longer claims snapping is what makes FFE efficient.
* Cost: FFE lowest under the paper's cost model (median 0.0006 / 0.0009 / 5.6 vs 15-53), but FFE optimises that cost; in normalised L1 it is not smaller than PSO on two corpora.
* Naive and ZOO have many feasible evasions at ~zero cost on CSE-CIC-IDS2018 (feasible share 88% / 83%): consistent with moving only the coupled rates inside the 5% tolerance of the feasibility check. The check is not exact.
* LightGBM resists FFE on CIC-IDS2017 (35.1% vs 99.2%); not explained.
* Ablation (CSE-CIC-IDS2018, N=283): saturated at 98.9% for the full cone; cost regime and snapping offset uninformative.
* Transfer (N=221 flows detected by all four architectures): 55.7-78.7% to non-CatBoost targets, 24.0% to CatBoost.
* Defenses (CSE-CIC-IDS2018, FFE M=5 / M=20 / CAPGD): undefended 84.1 / 98.9 / 88.3; MPS 78.8 / 78.8 / 71.9; LGAT 46.5 / 80.3 / 74.3; MPS+LGAT 62.0 / 62.0 / 55.1. LGAT mainly protects against the attack it trained on. No clean-accuracy cost.

### Additional corpora (supplementary S7; XGBoost; single seed; no five-seed repetition)

FFE is **not uniformly best**: InSDN 81.0 (PSO 85.5 on 200 flows, not significantly different, CAPGD 79.9, TSBS off 46.7); USB-IDS-1 **0.0** for FFE and PSO vs 95-99 for CAPGD/generator/unconstrained/ZOO/MOEVA (the target's score is constant over the attack region; FFE's moves apparently do not reach a split; not fully diagnosed); CIDDS-001 100.0 (PSO 100.0, generator 99.1). Per-dataset choices: InSDN window = 0 (exporter wrote -1), DDoS family lost (protocol 0); USB-IDS-1 92% attacks; CIDDS-001 unidirectional (no backward traffic or window), only `attacker` counted as attack (5,560 flows, 0.08%), `victim`/`suspicious`/`unknown` dropped. The main-text claim "highest Real ASR" is restricted to the three main corpora.

## 3. Manuscript (`paper/`)

Target: *Information Sciences*; review layout; **exactly 40 pages** including references (limit 40, no slack: any addition must be paid for by cutting text or moving material to the supplementary). Abstract 196 words (limit 200); highlights in `highlights.tex`.
`supplementary.tex` (12 pages): proofs, auxiliary algorithms, plus `supplementary_extra.tex` (per-family, TSBS, ablation, transfer and corpus-statistics tables S1-S6, section S7). Discussion and conclusion are in `sec_discussion.tex`.
Build: `bash paper/build.sh`. Tests: `pytest` (all pass; new modules 93-98% covered).

## 4. Open items

1. **More datasets.** InSDN, USB-IDS-1 and CIDDS-001 were downloaded from their public sites and added (supplementary S6/S7, `summarize_results.EXTRA`). Further corpora: add an alias table in `ALIASES`, family mapping and builder branch in `full_data.py`, a row in `EXTRA`, rerun `run_full_benchmark.py --datasets <name>`; results must go to the supplementary because of the page limit.
2. **Defenses** are evaluated on one corpus, one seed, FFE and CAPGD only; the attacker is not adaptive to the training procedure.
3. **White-box tree attacks** (LT-Attack, Cascioli et al.) are cited, not run; the novelty claim against them is untested.
4. **Packet-level validation**: feasibility is schema conformance only; replay with feature re-extraction is not done.
5. **Baseline tuning**: default budgets, one small surrogate; no sensitivity study. Repaired variants of ZOO/HSJ not run.
6. LightGBM/CIC-IDS2017 gap, and why CatBoost resists transfer, are unexplained.
7. Random IID split: near-duplicate flows can sit on both sides; a time- or host-grouped split would give lower, more realistic accuracies.
