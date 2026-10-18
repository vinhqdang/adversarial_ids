# HANDOFF: "Steering Through the Trees" (FFE), state of work

Repo: `vinhqdang/adversarial_ids`, branch `main` only (owner's rules: push to `main`, no other branch, no AI attribution anywhere, git identity `Vinh <dqvinh87@gmail.com>`).
Paper: `paper/` (LaTeX, `elsarticle`), author Quang-Vinh Dang. Python: conda env `py313` (call `.../envs/py313/python.exe` directly; `conda run` is broken on this machine). GPU: RTX 5000 Ada, 16 GB.

## 1. What changed in this round (and why)

The previous results came from **12,000-flow samples** built by `real_data_loader.py` (byte-range slices of a few raw files, then random subsampling, silent fallback to synthetic data on error). They were not full-data results. This round replaced them:

* `src/datasets/full_data.py` reads every raw file (`data/raw/`, git-ignored; `download_raw()` fetches them), maps to the 12-feature schema, drops invalid records, removes exact duplicates and label-conflicting records, and writes `data/full/*.parquet` (git-ignored; `*_stats.json` are tracked). `stratified_split` makes the 70/10/20 split after dedup.
* Datasets: CSE-CIC-IDS2018 (4,300,373 flows kept), CIC-IDS2017 (1,692,949), UNSW-NB15 (120,317; official train+test partition only, 257,673 records, not the full 2.54 M capture). CSE-CIC-IDS2018 files are capped at 1,048,575 rows each in the official distribution.
* `project_network_invariants_batch` (vectorised P_M, tested equal to the scalar one).
* New baselines in `src/attacks/modern_baselines.py`: CAPGD (surrogate transfer) and a NIDSGAN-style generator. Both are **our re-implementations**, one MLP surrogate, one hyperparameter setting.
* `experiments/run_full_benchmark.py` (1,000-flow cohort for fast methods, first 200 for MOEVA/PSO), `run_ablations.py`, `run_transferability.py`, `run_defenses.py` now use the full data (cohort 300 balanced by family; LGAT augmentation capped at 5,000 flows).
* `experiments/summarize_results.py` generates every manuscript table (`paper/tables/*.tex`) and `results/summary.json` (exact McNemar, XGBoost root-split counts). Nothing in the tables is typed by hand.
* `paper/references.bib`: 98 entries, 39 new 2021-2026 references verified against Crossref/arXiv; four wrong existing entries corrected (verkerken2026problem, gerlach2025efficiency, lindqvist2025adversarial, simonetto2024constrained); published versions with DOI preferred over arXiv. Entries without a DOI because none exists: `ke2017lightgbm` (NeurIPS), `ring2017flow` (ECCWS, original CIDDS-001 paper), `caratheodory1918vorlesungen` (1918 book).

## 2. Results (all from `results/full_*.json`, `ablation_results.json`, `transferability_results.json`, `defense_results.json`)

XGBoost, Real ASR (evades and physically feasible): FFE 100.0 / 95.9 / 61.5 % on CSE-CIC-IDS2018 / CIC-IDS2017 / UNSW-NB15 with 13.0 / 16.7 / 91.4 queries and 100 % PFR.

* Significantly better than PSO and CAA/MOEVA on all three corpora.
* CAPGD (surrogate): 99.5 / 55.2 / 26.8 %. **On CSE-CIC-IDS2018 FFE and CAPGD are statistically tied** (McNemar p = 0.0625) and CAPGD needs 1.1 queries vs 13.0. The paper says so; do not overstate.
* LightGBM: FFE 100.0 / 34.6 / 55.2 %. CIC-IDS2017 LightGBM resists far better than XGBoost; no mechanism was analysed.
* The "cost" column favours FFE partly by construction (FFE steers by it); the paper states this.
* Clean accuracy: 98.04 / 99.81 / 88.80 % (XGBoost). UNSW-NB15 is weak because the 12-feature mapping is lossy.
* Ablation (CSE-CIC-IDS2018, N=300): saturated at 100 % for the full cone, so cost regime and snapping offset are uninformative.
* Transfer (N=300): XGBoost to LightGBM 69.0, Random Forest 81.3, CatBoost 37.7 %; CatBoost is the hardest target (34-38 %).
* Defenses (M=5, N=300): undefended 96.3 % ASR, MPS 67.0, LGAT 33.0, MPS+LGAT 38.0 (LGAT vs dual not distinguishable). The old "malleability penalty trap" (MPS 98.9 %) and "5.3 % dual" claims are **not reproduced** at full scale and were removed from the paper. Old numbers live only in `results/legacy_12k_benchmark_results.json`.

## 3. Manuscript (`paper/`)

Target: *Information Sciences*; `elsarticle` review layout; **40 pages** including references (limit 40, currently exactly at the limit; any addition must be paid for by cutting or moving text to the supplementary). Abstract 172 words; highlights in `highlights.tex` (5 bullets, <= 85 characters).
Supplementary (`supplementary.tex`, 10 pages): proofs, auxiliary algorithms, plus per-family table, full ablation table and transferability table (`supplementary_extra.tex`).
Build everything: `bash paper/build.sh`. Tests: `pytest` (68 tests pass; coverage of the new modules 93-98 %).

Section map: `sec1` intro, `sec2` related work + taxonomy table, `sec3`-`sec5` formulation/geometry/algorithm, `sec6` setup, `sec7` master results, `sec8` ablation/transfer summary, `sec9` defenses, discussion and conclusion inside `elsevier_manuscript.tex`.

## 4. Open items

1. **Further datasets.** The owner has CIDDS-001, USB-IDS-1, InSDN and others; not found on this machine and no path was given. To add one: add an alias table to `ALIASES` in `full_data.py`, a family mapping, a builder branch, and a dataset row in `experiments/summarize_results.py::DATASETS`; then rerun the benchmark. The page limit means the new rows must go to the supplementary.
2. **Seeds.** One training seed per experiment; CIs capture cohort sampling only. A multi-seed run (retrain + new cohort) is the main remaining statistical gap.
3. **Defenses** were only attacked with FFE (the attack that generated the augmentation) and only on CSE-CIC-IDS2018. Evaluate against CAPGD and on the other corpora.
4. **LightGBM/CIC-IDS2017** gap is unexplained; inspect LightGBM splits.
5. White-box tree attacks (LT-Attack, Cascioli et al.) are cited but not run (need tree structure, no flow constraints).
6. Hardware/line-rate replay of the packet realisation is still future work.
