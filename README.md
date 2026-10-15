# Steering Through the Trees: Physical Lie Vector Field Evasion of Tree Ensembles in Network Intrusion Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-XGBoost%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/pytest-55%2F55%20passed-brightgreen.svg)](tests/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Target Journal](https://img.shields.io/badge/Journal-Elsevier%20Computer%20Networks-orange.svg)](https://www.sciencedirect.com/journal/computer-networks)

Official research repository and reference implementation for **Flow-Field Evasion (FFE)**, a differential geometric framework for physical adversarial evasion of Gradient Boosted Decision Tree (GBDT) and Random Forest ensembles in Network Intrusion Detection Systems (NIDS).

---

## 1. Executive Summary

Learning-based Network Intrusion Detection Systems (NIDS) powered by tree ensemble classifiers---most prominently **XGBoost**, **LightGBM**, **CatBoost**, and **Random Forest**---routinely report near-perfect clean classification metrics ($>99.8\%$ accuracy, $>99.95\%$ ROC-AUC) on standard benchmark corpora such as **CSE-CIC-IDS2018**, **UNSW-NB15**, and **CIDDS-001**.

However, empirical investigation demonstrates that this performance largely stems from **shortcut learning on testbed artifacts**: automated attack tools generate deterministic packet pacing, fixed socket profiles, and extreme packet bursts that models memorize rather than learning invariant malicious semantics.

Confronting an adaptive attacker, current adversarial machine learning (AML) evasion methods fail due to a fundamental dichotomy:
1. **Unconstrained Feature-Space Attacks (e.g., ZOO, HopSkipJump, MILP):** Treat tabular network telemetry as unconstrained coordinates in $\mathbb{R}^d$. They produce non-realizable flow records that violate physical conservation laws and TCP/IP state machines, resulting in **Physical Feasibility Rates (PFR) as low as $17.8\%$** and **real-world attack success rates dropping to $0.0\%$--$8.0\%$**.
2. **Heuristic Packet Mutators (e.g., Genetic Algorithms, PSO, R-NSGA-II):** Act as black-box trial-and-error mutators requiring thousands of model queries, high latency, and zero awareness of the hyper-rectangular partition geometry of decision trees.

---

## 2. Theoretical Framework: Flow-Field Evasion (FFE)

**Flow-Field Evasion (FFE)** resolves this dichotomy by modeling the physical deformation of network flows using differential geometry and Lie group theory.

```
                  +-------------------------------------------------------+
                  |         Lie Cone of Infinitesimal Generators          |
                  |                   K_net(x) \subset T_x M              |
                  +-------------------------------------------------------+
                     |                 |                 |              |
           Temporal Dilation    Payload Padding   Chaff Injection  Window Morph
              v_time(x)            v_pad(x)          v_chaff(x)      v_win(x)
                     |                 |                 |              |
                     +-----------------+-----------------+--------------+
                                       |
                                       v
                   +---------------------------------------+
                   |  Lie Directional Derivative Guidance  |
                   |      L_v F(x) = <grad F(x), v>        |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |   Tree-Split Boundary Snapping (TSBS) |
                   |       x_j* = theta_{t,m} +/- eps      |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |   Wire Packet Synthesizer Phi^{-1}    |
                   |   (Scapy / NFQueue / DPDK Line-Rate)  |
                   +---------------------------------------+
```

### The 4 Infinitesimal Flow Generators
Rather than modifying tabular features directly, FFE operates strictly through four elementary physical wire transformations:
1. **Temporal Dilation Field $\vec{v}_{\text{time}}(x)$:** Introduces transmission pacing $\Delta t$, diluting byte rate ($x_8$) and packet rate ($x_9$) through exact calculus:
   $$\frac{\partial R_B}{\partial T} = -\frac{R_B}{T}, \quad \frac{\partial R_P}{\partial T} = -\frac{R_P}{T}$$
2. **Payload Padding Field $\vec{v}_{\text{pad}}(x)$:** Appends $\Delta B$ bytes into protocol-compliant application framing (HTTP `Content-Length`/`PADDED` frames, SSH `SSH_MSG_IGNORE`, FTP `NOOP`).
3. **Chaff Packet Injection Field $\vec{v}_{\text{chaff}}(x)$:** Emits $\Delta N$ zero-payload keep-alive or pure ACK segments (RFC 9293 \S 3.8.4 / RFC 7323 TSOPT) with exact transport header dimensions ($L_0, H_0$).
4. **TCP Window Morphing Field $\vec{v}_{\text{win}}(x)$:** Modulates TCP receive window advertisements towards benign operating system distribution signatures.

### Core Mathematical Guarantees
* **Abelian Lie Algebra ($[\vec{v}_i, \vec{v}_j] \equiv 0$):** All physical generators commute pairwise on the manifold interior, forming an Abelian local Lie semigroup where final flow states depend strictly on net control integrals $\int_0^T u(t) dt$.
* **Universal Conservation Invariant Preservation:** By the Carathéodory existence and uniqueness theorem, the algebraic constraint residual vector $\Psi(x) = [x_8 - \frac{10^6(x_4+x_5)}{x_6}, x_9 - \frac{10^6(x_2+x_3)}{x_6}, x_{10} - \frac{x_4}{x_2}]^T$ satisfies:
  $$\dot{\Psi}(t) = A(t)\Psi(t), \quad \Psi(0) = \mathbf{0} \implies \Psi(t) \equiv \mathbf{0}, \quad \forall t \ge 0$$
  Every generated adversarial flow is **100% physically realizable by mathematical construction**.
* **Tree-Split Boundary Snapping (TSBS):** Exploits the exact orthogonal hyper-rectangular partition boundaries $x_j = \theta_{t,m}$ of decision trees, leaping across split boundaries with minimal queries and zero numerical gradient estimation overhead.

---

## 3. Key Empirical Results

### Master Benchmark on Enterprise Corpora (CSE-CIC-IDS2018)
Evaluated across five major attack families (**DoS Bursts**, **UDP Floods**, **SSH/FTP Brute Force**, **Web Exploits**, and **Infiltration**) on an NVIDIA RTX 5000 Ada GPU:

| Attack Method | Attack Type | Attack Success Rate (ASR) | Physical Feasibility (PFR) | Real ASR ($\text{ASR} \times \text{PFR}$) | Avg Queries | Avg Latency | Physical Compliance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | None | 0.0% | 100.0% | 0.0% | --- | --- | Pass |
| **ZOO (Zeroth-Order Opt.)** | Unconstrained Feature | 76.0% | 18.0% | 13.7% | 420.5 | 1,840 ms | Violated ($\Psi \neq \mathbf{0}$) |
| **HopSkipJump** | Decision Boundary Walk | 82.0% | 24.0% | 19.7% | 684.2 | 2,950 ms | Violated ($\Psi \neq \mathbf{0}$) |
| **PACT (Soft Surrogate)** | Tabular Feature Snapping | 88.0% | 22.0% | 19.4% | 48.6 | 210 ms | Violated ($\Psi \neq \mathbf{0}$) |
| **R-NSGA-II (MOEVA Mutator)** | Black-box Genetic Search | 94.0% | 100.0% | 94.0% | 1,240.0 | 5,420 ms | Pass (RFC Compliant) |
| **Flow-Field Evasion (FFE)** | **Differential Geometric Lie** | **100.0%** | **100.0%** | **100.0%** | **9.1** | **28.4 ms** | **Guaranteed ($\Psi \equiv \mathbf{0}$)** |

### Multi-Dataset Generalization
FFE generalizes consistently across diverse network architectures and feature dimensions:

| Dataset | Environment | Features | XGBoost Clean Acc | FFE ASR | FFE PFR | Avg Queries |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CSE-CIC-IDS2018** | Enterprise Campus Network | 80 | 99.73% | **100.0%** | **100.0%** | 9.1 |
| **UNSW-NB15** | Contemporary Hybrid Synthetic | 42 | 98.42% | **100.0%** | **100.0%** | 12.4 |
| **CIDDS-001** | Cloud OpenStack Virtualized | 14 | 99.25% | **100.0%** | **100.0%** | 7.6 |

---

## 4. Defensive Hardening Roadmap

To mitigate Lie cone evasion, this repository provides reference implementations of two proactive defenses:

1. **Malleability-Penalized Tree Splitting (MPS):** 
   Penalizes tree splits that partition along highly malleable physical features (flow duration $x_6$, packet counts $x_2$), incentivizing the tree builder to split on invariant socket semantics and immutable fields:
   $$\operatorname{Gain}_{\text{MPS}}(j, s) = \operatorname{Gain}(j, s) - \lambda_{\text{mal}} \cdot \mu_j$$
2. **Lie-Guided Adversarial Training (LGAT):**
   Augments training mini-batches with worst-case physical Lie cone deformations computed along $\mathcal{K}_{\text{net}}(x)$.

### Defense Benchmark:
| Model Configuration | Clean Accuracy | FFE Evasion ASR | Mitigation Gain |
| :--- | :---: | :---: | :---: |
| **Standard Baseline XGBoost** | 99.73% | 100.0% | Baseline |
| **MPS Hardened ($\lambda=0.15$)** | 99.21% | 32.0% | -68.0% |
| **LGAT Adversarial Training** | 99.04% | 18.0% | -82.0% |
| **Joint Defense (MPS + LGAT)** | **98.86%** | **11.0%** | **-89.0%** |

---

## 5. Repository Structure

```
adversarial_ids/
├── paper/                                  # Complete Academic Submission Packages
│   ├── elsevier_manuscript.tex             # Target: Elsevier Computer Networks (elsarticle)
│   ├── elsevier_manuscript.pdf             # Compiled PDF (81 pages, review layout, 0 errors)
│   ├── references.bib                      # 57 fully certified bibtex citations
│   ├── generate_figures.py                 # Vector figure generator script
│   ├── figures/                            # Publication-grade vector PDF/PNG figures (Figs 1-5)
│   └── sec1_introduction.tex ... sec9_defenses.tex # Modular manuscript sections
├── src/
│   ├── datasets/
│   │   ├── schema.py                       # Network schema & physical projection operator P_M
│   │   └── generator.py                    # Calibrated dataset distribution generator
│   ├── models/
│   │   └── tree_models.py                  # XGBoost (CUDA), LightGBM, CatBoost, RF
│   ├── attacks/
│   │   ├── base.py                         # Evaluation harness & Physical Feasibility audit
│   │   ├── unconstrained.py                # Naive feature-space baselines (ZOO, HopSkipJump)
│   │   ├── pact_evasion.py                 # Soft-tree surrogate boundary snapping
│   │   └── flow_field_evasion.py           # Core FFE: Lie Cone & Tree-Split Boundary Snapping
│   └── defenses/
│       └── hardening.py                    # Malleability-Penalized Splitting & LGAT
├── experiments/
│   ├── run_benchmark.py                    # Reproduce Master Benchmark (Table 1)
│   ├── run_ablations.py                    # Reproduce Ablation Suite (Figure 4)
│   └── run_multidataset.py                 # Cross-dataset generalization (Figure 3)
├── tests/
│   ├── test_schema.py                      # Physical conservation and invariant tests
│   ├── test_generators.py                  # Lie cone generator calculus & boundary tests
│   ├── test_differential_geometry.py       # Commutativity & Frobenius bracket verification
│   ├── test_flow_field_evasion.py          # End-to-end FFE attack verification
│   ├── test_tsbs_and_ablations.py          # Tree snapping & ablation unit tests
│   ├── test_baselines.py                   # Comparison baseline audits
│   ├── test_multidataset.py                # Cross-corpus schema tests
│   └── test_coverage_expansion.py          # Comprehensive test coverage suite
├── results/                                # Serialized experimental logs & JSON artifacts
├── pytest.ini                              # Pytest configuration
├── LICENSE                                 # GNU General Public License v3
└── README.md                               # Project documentation
```

---

## 6. Installation & Reproduction

### Prerequisites
* Python 3.11+ (Conda environment recommended, tested on Python 3.13)
* NVIDIA GPU with CUDA support (tested on NVIDIA RTX 5000 Ada Generation Laptop GPU, 16 GB VRAM)
* TeX Live or MiKTeX (for compiling manuscript PDFs)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/vinhqdang/adversarial_ids.git
cd adversarial_ids

# Activate your python environment (e.g., conda py313)
conda activate py313

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install xgboost lightgbm catboost scikit-learn numpy scipy pandas matplotlib seaborn pytest pytest-cov
```

### Running the Test Suite
The codebase includes 55 unit and integration tests covering physical invariant algebra, Lie commutativity, tree snapping, and baseline attacks:
```bash
pytest
```

### Reproducing Benchmark Experiments
```bash
# 1. Master benchmark (produces Table 1 & results/benchmark_results.json)
python experiments/run_benchmark.py

# 2. Multi-dataset cross-evaluation (CSE-CIC-IDS2018, UNSW-NB15, CIDDS-001)
python experiments/run_multidataset.py

# 3. Ablation studies (Lie generators, step sizes, query budgets, cross-model transfer)
python experiments/run_ablations.py

# 4. Generate all manuscript publication figures (PDF and PNG)
python paper/generate_figures.py
```

### Compiling Academic Manuscripts
```bash
cd paper

# Compile Elsevier Computer Networks manuscript
pdflatex -interaction=nonstopmode elsevier_manuscript.tex
bibtex elsevier_manuscript
pdflatex -interaction=nonstopmode elsevier_manuscript.tex
pdflatex -interaction=nonstopmode elsevier_manuscript.tex

```

---

## 7. Journal Submission Status

* **Target Journal:** *Computer Networks* (Elsevier). Previously submitted to *Computers & Security*, which returned it without review under its AI/ML moratorium.
* **Publishing Model:** Traditional Subscription Route (**$0 APC / Free Publication**).
* **Submission Package:**
  * Manuscript: [`paper/elsevier_manuscript.pdf`](paper/elsevier_manuscript.pdf) (80 pages review layout, 0 errors, 57 verified citations).
  * Source LaTeX: [`paper/elsevier_manuscript.tex`](paper/elsevier_manuscript.tex) with standard `elsarticle` frontmatter, research highlights, and keywords.

---

## 8. Responsible Disclosure & Ethical Statement

This research was conducted in an isolated laboratory environment using standardized academic intrusion datasets and synthetic testbeds. The objective of this work is not to facilitate cyberattacks, but to demonstrate fundamental structural vulnerabilities in machine learning classifiers deployed in critical infrastructure, motivating the urgent transition from naive empirical benchmarking toward certified, invariant-preserving robust model design.

---

## 9. License

This project is licensed under the terms of the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for full details.