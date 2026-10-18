# HANDOFF: "Steering Through the Trees" (FFE) – State of Work

Repo: `vinhqdang/adversarial_ids`, branch `main` only (owner's rule: push to `main`, no other branches, no AI attribution in commits, git identity `Vinh <dqvinh87@gmail.com>`).  
Paper: `paper/` (LaTeX, `elsarticle`). Author: Quang-Vinh Dang (`vinh.dq4@buv.edu.vn`).  
Environment: Conda `py313` (`Python 3.13.11`), PyTorch CUDA True, NVIDIA RTX 5000 Ada GPU (16 GB VRAM).

---

## 1. Submission Target and Author Guidelines

- **Target Journal:** *Information Sciences* (Elsevier, IF 6.0). Subscription route (no APC).
- **Format Constraints:**
  - `\documentclass[review,12pt]{elsarticle}`.
  - Page limit: **$\le 40$ pages** double-spaced including figures, tables, and references (current compiled length: **exactly 40 pages**).
  - Abstract: $\le 200$ words (current length: **187 words**).
  - Highlights: 3–5 bullets, each $\le 85$ characters including spaces (maintained in standalone `paper/highlights.tex` and `paper/highlights.pdf`).
  - Keywords: 1–7 keywords.
  - Structure: Consolidated into **6 logical sections**.
  - Statements: CRediT authorship, Competing Interests, Funding, and Data Availability included.
  - Supplementary Material: Theoretical proofs (Proposition 1, Corollary S1, Theorem 1) and auxiliary algorithms (Algorithms S1–S4) moved to `paper/supplementary.tex` (8 pages).

---

## 2. Completed Status & Current Implementation

### 2.1. Real Benchmark Datasets (No Synthetic Data)
All experiments use real network flow traffic extracted from official public repositories and mapped onto our 12-feature schema (`data/`):
1. **CSE-CIC-IDS2018** (`data/cse_cic_ids2018.csv`): 12,000 real flows (9,000 benign, 3,000 malicious across 5 attack families: Volumetric DoS SYN, UDP Flood, Brute Force Auth, Web Exploits, Infiltration) extracted from official AWS S3 captures.
2. **UNSW-NB15** (`data/unsw_nb15.csv`): 12,000 real flows extracted from official UNSW repository.
3. **CIC-IDS2017** (`data/cic_ids2017.csv`): 12,000 real flows extracted from Canadian Institute for Cybersecurity archives.

### 2.2. Master Evasion Benchmark Against Baselines
All baseline runs are executed on real data and archived in `results/benchmark_results.json`:
- **Unconstrained Feature-Space (Naive):** 39.0% ASR, 5.0% PFR, 4.0% Real ASR, 77.8 queries, $\bar{D}_{L_1} = 9,673.78$, 6.62s.
- **ZOO-NIDS (Chen et al., 2017):** 8.0% ASR, 6.0% PFR, 0.0% Real ASR, 144.0 queries, $\bar{D}_{L_1} = 9,323.43$, 30.04s.
- **HopSkipJump-NIDS (Chen et al., 2020):** 100.0% ASR, 0.0% PFR, 0.0% Real ASR, 47.0 queries, $\bar{D}_{L_1} = 1.64\times 10^6$, 10.55s.
- **CAA / MOEVA (Simonetto / Ghamizi):** 39.0% ASR, 15.0% PFR, 1.0% Real ASR, 336.2 queries, $\bar{D}_{L_1} = 111.85$, 103.31s.
- **Traffic Manipulator PSO (Han et al., 2021):** 54.0% ASR, 100.0% PFR, 54.0% Real ASR, 182.6 queries, $\bar{D}_{L_1} = 2,946.18$, 39.85s.
- **Flow-Field Evasion (FFE - Ours):** **59.0% ASR, 100.0% PFR, 59.0% Real ASR, 79.3 queries, $\bar{D}_{L_1} = 0.392$, 28.66s.**
  - *Key takeaway:* FFE matches PSO in evasion success (59.0% vs. 54.0%) while slashing perturbation cost by **$7,500\times$** ($\bar{D}_{L_1} = 0.392$ vs. $2,946.18$) and query complexity by **$2.3\times$** (79.3 vs. 182.6 queries) through Tree-Split Boundary Snapping (TSBS).

### 2.3. Multi-Dataset & Multi-Architecture Validation
Evaluated across XGBoost (CUDA) and LightGBM across all three datasets (`results/benchmark_results.json`):
- **CSE-CIC-IDS2018:**
  - XGBoost: Clean Acc 98.60%, Real ASR 59.0%, PFR 100.0%, 79.3 queries, 417.0 ms.
  - LightGBM: Clean Acc 98.67%, Real ASR 97.0%, PFR 98.0%, 10.4 queries, 15.5 ms.
- **UNSW-NB15:**
  - XGBoost: Clean Acc 94.60%, Real ASR 98.0%, PFR 100.0%, 32.2 queries, 194.9 ms.
  - LightGBM: Clean Acc 94.77%, Real ASR 94.0%, PFR 100.0%, 34.3 queries, 61.5 ms.
- **CIC-IDS2017:**
  - XGBoost: Clean Acc 99.87%, Real ASR 11.0%, PFR 93.0%, 170.3 queries, 986.1 ms.
  - LightGBM: Clean Acc 99.90%, Real ASR 3.0%, PFR 93.0%, 213.7 queries, 381.3 ms.
  - *Topology explanation:* Benign and malicious flows in CIC-IDS2017 are cleanly separated by deep immutable splits (destination port, backward packet statistics), restricting forward Lie cone traversal to 3.0%–11.0% Real ASR.

### 2.4. Defensive Countermeasures & The "Malleability Penalty Trap"
Evaluated on the detected malicious test cohort ($N=95$ flows) at attack budget $M=5$ (`results/defense_results.json`):
- **Undefended Baseline:** Clean Acc 98.62%, Clean AUC 99.88%, Robust Recall 48.4%, FFE ASR 51.6%.
- **MPS-Hardened Alone:** Clean Acc 98.67%, Clean AUC 99.85%, Robust Recall 1.1%, **FFE ASR 98.9%**.
  - *Mechanism:* Penalizing malleable rate features ($x_6, x_8, x_9$) forces trees to split on unpenalized mutable discrete features ($x_2$ forward packets, $x_4$ payload bytes, $x_7$ window, $x_{11}$ header length). FFE exploits these unhardened boundaries via TSBS with near-zero cost in 1–3 queries.
- **LGAT-Hardened Alone:** Clean Acc 98.72%, Clean AUC 99.85%, Robust Recall 74.7%, FFE ASR 25.3%.
- **Dual-Hardened (MPS + LGAT):** Clean Acc 98.67%, Clean AUC 99.84%, **Robust Recall 94.7%, FFE ASR 5.3%**.
  - Slashes evasion to 5.3% with zero degradation of clean detection accuracy.

### 2.5. Ablation Studies & Black-Box Transferability
- **Ablation (`results/ablation_results.json`):**
  - Full FFE: 59.0% Real ASR, 100.0% PFR.
  - No Lie Projection: 59.0% ASR, but PFR collapses to 0.0% (0.0% Real ASR).
  - No TSBS (continuous relaxation only): Real ASR drops to 41.0%, queries rise to 124.8.
  - Single generator restrictions: Evasion drops to 0.0%–32.0%.
- **Black-box Transferability (`results/transferability_results.json`):**
  - Surrogate XGBoost transfers at 95.8% to LightGBM, 77.9% to Random Forest, and 56.8% to CatBoost.

---

## 3. Manuscript Structure (6 Sections)

1. `sec1_introduction.tex`: Motivation, problem-space challenges, four core contributions.
2. `sec2_related_work.tex`: State of the art, feature-space vs. problem-space evasion, formal taxonomy table.
3. `sec3_problem_formulation.tex`: Threat model (Tier 1 & Tier 2), 12-feature physical flow manifold, conservation and rate invariants.
4. `sec4_differential_geometry.tex` & `sec5_algorithm.tex`: Lie algebra formulation, four infinitesimal generators, TSBS algorithm.
5. `sec6_experiments.tex`, `sec7_results.tex`, `sec8_ablations.tex`, `sec9_defenses.tex`: Unified empirical evaluation, master benchmark, multi-dataset, ablations, transferability, defenses.
6. Section 6 (`elsevier_manuscript.tex`): Operational feasibility, query efficiency, limitations, and concluding remarks.
- Back matter: CRediT authorship, Competing Interests, Funding, Data Availability, References.

---

## 4. Verification and Build Instructions

To build the manuscript and supplementary PDFs:
```bash
cd paper
pdflatex -interaction=nonstopmode elsevier_manuscript.tex
bibtex elsevier_manuscript
pdflatex -interaction=nonstopmode elsevier_manuscript.tex
pdflatex -interaction=nonstopmode elsevier_manuscript.tex
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode highlights.tex
```

To run the Python test suite:
```bash
conda run -n py313 pytest
```
Currently 55/55 tests pass in ~15 seconds.

---

## 5. Known Limitations & Threats to Validity

1. **Stratified Sampling:** 12,000-flow samples per dataset enable exhaustive multi-step heuristic baselines (e.g., MOEVA and PSO take >100s per flow). Scaling to full multi-gigabyte wire traces remains future work.
2. **Fixed Random Seeds:** A single fixed seed ensures deterministic fairness across baselines. Multi-seed confidence intervals are identified as threats to validity.
3. **Hardware Testbed:** Packet realization via Scapy/NFQueue proxy enforces protocol compliance at flow level; hardware line-rate FPGA/DPDK replay remains future work.
