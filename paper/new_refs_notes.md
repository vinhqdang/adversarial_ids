# New reference candidates (verified 2026-10-18)

Verification sources: Crossref API (DOI, volume, pages), arXiv API (IDs, authors), PMLR pages (ICML 2021), GitHub API (repo licenses). Search-engine summaries were not trusted on their own.

## CORRECTIONS NEEDED in existing references.bib (found during verification)
- `verkerken2026problem`: arXiv ID 2404.14876 is a physics paper, not this one. Correct record: Proc. ACM ASIA CCS 2026, pp. 1043-1059, DOI 10.1145/3779208.3807482.
- `gerlach2025efficiency`: arXiv ID is 2507.10048, not 2507.03921. The SAIV venue was not verified.
- `lindqvist2025adversarial`: IEEE Access 13, pp. 16120-16132, DOI 10.1109/ACCESS.2025.3530695. The current bib has pages 12450-12465 and DOI ...3526112, both wrong.
- `simonetto2024constrained`: NeurIPS 37, pp. 27817-27849, DOI 10.52202/079017-0873. The current bib has pages 8910-8932 and an arXiv DOI.
- Not re-checked (still worth a pass): `simonetto2024tabularbench`, `apruzzese2023real`, `dang2026stfedxids`, `he2023adversarial`.

## Entries by topic (one-line relevance)

### (1) Constraint-aware / problem-space attacks on NIDS and tabular data
- debicha2023advbot: realistic botnet-flow evasion (C&S 2023), the closest flow-level "valid traffic" attack after Han 2021.
- debicha2026targeted: black-box targeted adversarial traffic generation against IoT IDS (IJMLC 2026), a recent black-box NIDS baseline for related work.
- zhang2025explainable: TDSC 2025 explainable and transferable NIDS attack, the current top-venue NIDS evasion comparison point.
- merzouk2022practicality: argues many NIDS evasion attacks are impractical, which supports our constraint-preservation motivation.
- sheatsley2022adversarial: constraint-aware (domain-constraint) attack formulation for NIDS.
- elshehaby2026perturbability: perturb-ability score for flow-based ML-NIDS, formalizes mutable vs immutable flow features.
- grini2025constrained: validity and transferability of constrained network attacks.
- vitorino2023realism: constrained adversarial examples vs RF/XGB/LGBM for IoT intrusion detection, with adversarial training.
- vitorino2024benchmark: robustness benchmark on CICIDS2017 and other datasets (tree models included), FPS 2023, LNCS.
- kireev2023cost: NDSS 2023 cost and utility awareness for tabular robustness (realistic threat model).
- he2026tabattackbench / he2025imperceptibility: tabular attack benchmark and imperceptibility analysis (ESWA 2026 / ISWA 2025).
- dyrmishi2025insights: systematic literature review of tabular adversarial attacks (arXiv, 2025).
- dyrmishi2024deep: generative models as constrained tabular attackers (ICMLC 2024).
- simonetto2023constrainedarxiv: arXiv precursor of CAA, cite only if the arXiv version is needed.
- venturi2024problem: problem-space structural attacks on GNN-NIDS (arXiv).

### (2) Tree-ensemble and black-box attacks
- cascioli2024faster: NeurIPS 2024 repeated evasion against tree ensembles. Needs tree structure, so it is white-box.
- devos2023detecting: detecting evasion in deployed tree ensembles (ECML PKDD 2023), relevant to defenses.
- pashamokhtari2022adiotack: C&S 2022 volumetric attacks and refinement of decision-tree-ensemble IoT inference models.
- cheng2019queryefficient: hard-label attack with a GBDT case. Pre-2021, but it is the standard query-efficient reference for trees.

### (3) Robust trees and verification
- vos2021efficient, vos2022robust: GROOT and robust optimal classification trees (ICML 2021, AAAI 2022).
- moshkovitz2021connecting: robustness through separation in decision trees (ICML 2021).
- calzavara2023verifiable, calzavara2025verifiableboosted: verifiable robust tree ensembles (CCS 2023, S&P 2025), the strongest recent defense references.
- calzavara2021feature: feature partitioning for robust tree ensembles.

### (4) Datasets and quality critiques
- engelen2021troubleshooting, lanvin2023errors, liu2022error: documented CICIDS2017/2018 label, capture and duplicate-packet errors.
- rosay2022cicids2017: analysis of CIC-IDS2017 data issues.
- sarhan2021netflow, sarhan2022standard, sarhan2022evaluating: NetFlow v1/v2 datasets (NF-UNSW-NB15-v2 and siblings) and the standard feature set.
- layeghy2024benchmarking: real vs synthetic NIDS dataset comparison.
- flood2024spectral: spectral complexity of benchmark NIDS datasets.
- cantone2024crossdataset: cross-dataset generalization failure of ML-NIDS.
- guo2026knowml: generalization of ML-NIDS (EuroS&P 2026).
- Not found or not verified: "Flowtbl", and a NetFlow v3 paper (Sarhan 2022). I dropped both.

### (5) Generative / LLM-related
- zolbayar2022nidsgan: GAN-based flow attack (arXiv only, no venue found).
- he2022liuer: grey-box NIDS attack (arXiv version; a TNSM 2026 version exists but I could not verify its authors, so it is not cited).
- merzouk2024diffusion: diffusion purification defense for NIDS (arXiv).
- li2026controllability, atif2026threeaxis: 2026 arXiv preprints on LLM-based NIDS under attack. The first uses an XGBoost surrogate with controllable-feature splits. Both are unreviewed, so use sparingly.
- Gap: I found no peer-reviewed 2023-2026 paper on LLM or diffusion generation of evasive flows against tree-based NIDS. Say so in the paper and do not overclaim.

## Recommended new baselines (feasible against XGBoost on 12 flow features)

1. **CAPGD / CAA (Simonetto et al., NeurIPS 2024).**
   - Repos: https://github.com/serval-uni-lu/constrained-attacks (MIT, last push 2024-07) and https://github.com/serval-uni-lu/tabularbench (MIT).
   - Query-based: CAPGD needs gradients, so on XGBoost it needs a differentiable surrogate (transfer attack). MOEVA, the search half, is query-based and needs only `predict_proba`.
   - Run: encode the 12 flow features as linear/relational constraints, then call MOEVA against `xgb.predict_proba`.
   - TabularBench is documented for deep models, so confirm tree support before relying on it. The existing bib already has CAA/MOEVA, so the update here is to use the 2024 CAA code and report both the transfer and MOEVA variants separately.

2. **Cascioli et al., "Faster Repeated Evasion Attacks in Tree Ensembles" (NeurIPS 2024).**
   - Repo: https://github.com/lorenzocascioli/faster-repeated-evasion-tree-ensembles (GitHub license field NOASSERTION, check the LICENSE file).
   - Not query-based: white-box, uses tree structure. The README lists xgboost and groot-trees as dependencies and `python main.py -m model_type -a attack_type dataset` as the entry point.
   - Use it as a white-box upper bound on tree-ensemble attacks. It has no flow-validity constraints.

3. **LT-Attack (Zhang, Zhang, Hsieh, NeurIPS 2020; already cited as `zhang2020efficient`).**
   - Repo: https://github.com/chong-z/tree-ensemble-attack (MIT, last push 2021).
   - Not query-based: white-box discrete search over leaf tuples. It supports XGBoost and LightGBM models.
   - It is the standard tree-ensemble reference, so pair it with Cascioli 2024 to cover the 2020 and 2024 lines.

4. **Adversarial Robustness Toolbox (ART) black-box attacks.**
   - Repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox (MIT, active as of 2026-10).
   - Query-based decision attacks (HopSkipJump, which you already use, plus ZOO, Boundary and similar) work on XGBoost via `XGBoostClassifier`.
   - ART also provides tree-specific attacks such as the decision-tree attack and the Kantchelian-style attack (already cited as `kantchelian2016evasion`).
   - The ART wrapper makes it easy to run several attacks with one API.

5. **Debicha et al. 2026 / Zhang et al. TDSC 2025 / Adv-Bot 2023 (NIDS-specific).**
   - I found no public code for any of the three, so they are comparison points for related work, not runnable baselines. I did not verify that no code exists, only that I did not find it.

Caveat: item 4 lists what ART supports from my knowledge, and I verified only the repo's existence and MIT license, not each attack class. Check the API docs before writing the experiment.
