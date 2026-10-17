# HANDOFF: "Steering Through the Trees" (FFE) – state of work

Repo: `vinhqdang/adversarial_ids`, branch `main` only (owner's rule: push to `main`, no other branches, no AI attribution in commits, git identity `Vinh <dqvinh87@gmail.com>`).
Paper: `paper/` (LaTeX, `elsarticle`). Owner: Quang-Vinh Dang.

This file lists what was done, what is NOT done, and what is NOT verified. Read section 3 before submitting anything.

## 1. History

1. Submitted to *Computers & Security*: returned without review (blanket moratorium on AI/ML papers).
2. Resubmitted to *Computer Networks* (COMNET-D-26-08135): rejected, "lacks conciseness". Elsevier offered a transfer to 5 journals, 3 of them open access only (APC), which the owner refuses.
3. Chosen target: *Information Sciences* (Elsevier, IF 6.0). It has a subscription route with no publishing charge. Its Guide for Authors: experimental papers max 40 double-spaced pages and 8 figures+tables; abstract <=200 words (Aims) / 250 (guide); 1-7 keywords; 3-5 highlights <=85 characters; Conclusions section; CRediT, funding, competing-interest, data-availability statements; generative-AI declaration if AI tools were used; supplementary material allowed.

## 2. What was done (commit `fc87294` on `main`; previous good state `c7cbd0a`)

- Fixed an algorithm that overflowed its page (TSBS split into two algorithms; stray TAB that printed `\text` as `ext`).
- Condensed the paper from 80 to 40 pages (`\documentclass[review,12pt]{elsarticle}`, refs single-spaced, floats at end by the class): 1 figure + 6 tables, abstract 193 words, 7 keywords, 5 highlights (<=83 chars), new Discussion/Conclusions/statements. Proofs (Prop. 1, Cor. S1, Thm. 1) and auxiliary algorithms S1-S4 moved to `paper/supplementary.tex` (+ `supplementary_body.tex`, 8 pages).
- Journal set to Information Sciences; cover letter for C&S deleted; README header/journal lines updated (README results tables are STALE, see 3.10).
- Corrected statements that contradicted the result files: defenses (95.0 -> 56.0 %, attack budget M=5, non-adaptive), latency range, transferability range (69-94 %), L1 cost, "under 70 ms", CIDDS-002.
- Qualified overclaims: feasibility "by construction" -> continuous relaxation + projection; PFR = schema conformance; MPS/LGAT described as implemented.
- Build: `cd paper && latexmk -pdf elsevier_manuscript.tex` and `pdflatex supplementary.tex` (twice). TeX Live was installed with apt (texlive-latex-extra, texlive-publishers, texlive-science, poppler-utils); if missing: `apt-get update && apt-get install -y texlive-latex-extra texlive-publishers texlive-science texlive-fonts-recommended texlive-bibtex-extra latexmk poppler-utils`.
- Process used: planner agent -> 9 parallel implementers (one per file) -> independent reviewer. A Sonnet review report drove the corrections above.

## 3. NOT done / NOT verified (most important first)

1. **All data is synthetic, not the public datasets.** `src/datasets/generator.py` ("Synthetic Flow Generator ... CIC-IDS2018 Characteristics") draws flows from RNG distributions for 5 attack families + benign, then enforces the manifold invariants. grep found no `read_csv`/download anywhere in `src/` or `experiments/`. The "UNSW-NB15" and "CIDDS-001" corpora are produced by the same generator, "calibrated" to published marginals. The owner expects real data: CSE-CIC-IDS2018, UNSW-NB15, CIDDS-001 (also CIC-IDS2017) are all public. The condensed paper now says "synthetic, calibrated" but a reviewer will ask for real data. **Recommended next step:** run the pipeline on real data.
   - Reachable from the sandbox (HTTP 200): `https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/` (folder `Processed Traffic Data for ML Algorithms/`, 10 CSVs, ~7 GB, one file is 4 GB), UNSW and unb.ca pages, huggingface.co. CIDDS page returns a 301 redirect. kaggle.com returned 404. Sandbox: 4 CPU, 15 GB RAM, ~29 GB disk free, no GPU.
   - First task: map the 12-feature schema (`src/datasets/schema.py`) to CICFlowMeter columns and check whether the real flows satisfy the invariants (x8 = 1e6(x4+x5)/x6, x10 = x4/max(1,x2), header/MTU relations). CICFlowMeter computes some columns differently, so the manifold constraints and the "100 % PFR" claim may need rework. Then re-run everything and rewrite results; expect lower accuracy than 99.9 % and ASR below 100 %.
2. **Baseline rows of Table 2 are not backed by any archived run.** ZOO / HopSkipJump / CAA-MOEVA / PSO numbers (queries 142.5 / 680 / 1240 / 1850; ASR 12/45/78/82; PFR 25.0/17.8/61.5/100; Real ASR 3/8/48/82; times 34.12/145.8/310.5/420.15 s; L1 0.382/0.512/0.342/0.485) appear only as hard-coded lists in `paper/generate_figures.py` (~lines 380-412). `results/benchmark_results.json` has only the Naive and FFE rows; `experiments/run_benchmark.py` runs only those two. `src/attacks/baselines.py` has `ZOONIDSAttack`, `HopSkipJumpNIDSAttack`, `TrafficManipulatorPSOAttack`, but `MOEVAAttack` is just `class MOEVAAttack(TrafficManipulatorPSOAttack): pass` (not MOEVA/R-NSGA-III, not CAA's CAPGD), and the PSO applies `project_network_invariants`, which is why its PFR is 100 %. The abstract, highlight 3, intro and conclusions rest on these numbers. Data-availability statement has an `[AUTHOR: ...]` placeholder about this. **Fix:** run all baselines (ideally real implementations, e.g. the CAA/MOEVA code from the authors' repos) and save JSON, or remove the comparison claims.
3. **Defenses as implemented differ from what the method name suggests** (`experiments/run_defenses.py`). "MPS" = hand-picked zeroing of features x6, x8, x9 in train/test/attack input (not malleability-penalized split gain, which is not implemented). "LGAT" = one-shot augmentation: 50 % of malicious training flows get an FFE adversarial copy (M=5, against the undefended model), appended once; the per-round algorithm in supplementary Algorithm S4 is NOT what was run. Attack budget M=5 (explains 95 % undefended vs 100 % at M=20). Non-adaptive attacker only. The paper text says this; if the owner wants the real methods, re-implement and re-run.
4. **PFR is circular.** FFE projects with `project_network_invariants` after every step and `verify_physical_feasibility` checks the same relations, so 100 % PFR measures the projection, not an independent test. Theorem 1 is about the continuous relaxation, not the discrete lattice. Nothing tests that real packets produce the claimed features (the Scapy/NFQueue proxy was never run). The paper now states this.
5. **Semantic preservation S(P*)=S(P) is never tested.** Temporal dilation dominates the SYN/UDP floods in Table 3 (those "evasions" work by slowing the flood). Stated as a limitation only.
6. **Cohorts and seeds.** Master cohort = 100 random test flows (12.9 queries, L1 0.251). Ablation/defense cohorts = 20 flows per family (11.6 queries at M=20). One seed everywhere, no variance, N=100. Clean accuracy of "the same" XGBoost model is 99.70 (Table 3), 99.72 (Table 6), 99.725 (transferability JSON) from separate runs. Latency for the same config: 43.7 ms vs 57.4 ms. LightGBM diagonal in Table 5 = 99.0 vs 100.0 elsewhere. Add seeds + confidence intervals when re-running.
7. **Naive-baseline PFR:** `benchmark_results.json` = 94.0 %, Table 2 shows N/A (footnote says PFR over evasive candidates only, code computes over all samples). Pick one definition.
8. **Placeholders the owner must fill:** `[AUTHOR: confirm]` in CRediT, competing interests, Funding; `[AUTHOR]` acknowledgements; `[AUTHOR: Zenodo DOI]`; baseline-provenance note. The manuscript lists affiliation BUV / `vinh.dq4@buv.edu.vn`, but the account email is `vinh.dang3@rmit.edu.vn`: confirm. No generative-AI declaration was written; that is the owner's decision (the journal requires one if AI tools were used in manuscript preparation).
9. **Page limit interpretation.** 40 pages is in elsarticle `review` (1.5 spacing, 12 pt) INCLUDING references and floats-at-end; body is 31 pages; references are single-spaced via `\baselinestretch{1}`. True double spacing would be longer. Supplementary holds proofs + auxiliary algorithms. If the journal counts differently, adjust.
10. **Stale material in the repo:** `README.md` results tables/claims (e.g. UNSW-NB15 98.42 %, "100 → 32/18/11" style numbers, 81 pages) were NOT updated; `paper/generate_figures.py` (~line 710) still holds stale fallback defense numbers 100/32/18/11; `experiments/render_ablation_tex.py` no longer matches `sec8_ablations.tex` (tables were merged); `paper/figures/fig2..fig5` are no longer included in the paper; ~10 entries in `references.bib` are uncited (harmless); several `dang*` self-citations are only weakly relevant (e.g. `dang2026stfedxids`).
11. **Possible reasons for rejection even after fixes:** synthetic data only; one seed/100 flows; FFE saturates at 100 % everywhere; theory is routine (commuting additive generators, oversold relative to the projection); thin defense section; attack-heavy framing for the journal. Table 1 taxonomy and Section 3 schema detail could be trimmed further.
12. Not re-verified by a second pass after the last edits: numbers in Discussion/Conclusions vs tables were checked by the reviewer before the final edits only. Re-run a number audit (every %, ms, s, "queries" in the text vs `results/*.json` and tables) after any change.

## 4. Truth sources for numbers

`results/benchmark_results.json` (Naive + FFE only, families, datasets), `results/ablation_results.json`, `results/transferability_results.json`, `results/defense_results.json`. Scripts: `experiments/run_benchmark.py`, `run_ablations.py`, `run_transferability.py`, `run_defenses.py`; models in `src/models/tree_models.py`; attacks in `src/attacks/`.

## 5. File map (paper/)

`elsevier_manuscript.tex` (front matter, Discussion, Conclusions, statements), `sec1..sec9_*.tex` (sections; sec4 framework, sec5 algorithms, sec6 setup, sec7 results, sec8 ablations+transferability, sec9 defenses), `supplementary.tex` + `supplementary_body.tex`, `references.bib`, `figures/`, compiled `elsevier_manuscript.pdf` and `supplementary.pdf`.

## 6. Suggested order for the next agent

1. Decide with the owner: real datasets (recommended) vs keeping synthetic. If real: do 3.1 first (feature mapping + invariant check on one small CSV before downloading everything).
2. Re-run FFE, baselines (real implementations, archived JSON), ablations, transferability, defenses on real data with several seeds; save JSON.
3. Update tables/abstract/highlights/conclusions from the JSON; delete the stale numbers listed in 3.10; keep abstract <=200 words and highlights <=85 chars.
4. Rebuild, check page count (<=40 incl. refs), no undefined refs, then commit and push to `main` only.
5. Fill author placeholders with the owner.

If the owner wants to undo the condensation commit: `git revert fc87294` (previous state `c7cbd0a` is the 80-page Computer Networks version).
