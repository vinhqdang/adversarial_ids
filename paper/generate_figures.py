"""
generate_figures.py
Generate publication-quality vector figures (PDF and PNG) for the Springer Nature manuscript.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Publication aesthetics configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# FIGURE 1: Architectural Workflow & Differential Geometry of FFE
# -------------------------------------------------------------------------
def create_figure_1():
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Color palette
    c_blue = '#1f77b4'
    c_orange = '#ff7f0e'
    c_green = '#2ca02c'
    c_red = '#d62728'
    c_purple = '#9467bd'
    c_box = '#f8f9fa'
    c_border = '#343a40'

    # Box 1: Packet Stream & Telemetry
    rect1 = patches.FancyBboxPatch((0.3, 1.0), 2.5, 4.2, boxstyle="round,pad=0.15",
                                   ec=c_border, fc='#eef2f7', lw=1.5)
    ax.add_patch(rect1)
    ax.text(1.55, 4.85, '1. Network Telemetry', weight='bold', ha='center', fontsize=11, color='#1e3a8a')
    ax.text(1.55, 4.4, '• Raw Packet Stream $\\mathcal{P}$\n• TCP/IP 3-Way Handshake\n• Invariant Conservation:\n  $x_8 = (x_4+x_5)/x_1$\n  $x_9 = (x_2+x_3)/x_1$\n• Flow Feature Vector:\n  $x \\in \\mathbb{R}^d$ ($d=12$)',
            ha='center', va='top', fontsize=8.5, linespacing=1.4)

    # Arrow 1 -> 2
    ax.annotate('', xy=(3.3, 3.1), xytext=(2.9, 3.1),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_border))

    # Box 2: Differential Flow Geometry (Lie Cone)
    rect2 = patches.FancyBboxPatch((3.4, 0.8), 2.9, 4.6, boxstyle="round,pad=0.15",
                                   ec='#d97706', fc='#fffbeb', lw=1.5)
    ax.add_patch(rect2)
    ax.text(4.85, 5.05, '2. Lie Cone Generators', weight='bold', ha='center', fontsize=11, color='#92400e')
    ax.text(4.85, 4.65, 'Tangent Space $T_x \\mathcal{M}_{\\text{flow}}$\nLie Cone $\\mathcal{K}_{\\text{net}}(x) = \\sum u_k \\vec{v}_k$\n($u_k \\geq 0$ strictly preserves laws)',
            ha='center', va='top', fontsize=8.5, linespacing=1.3)
    
    # 4 generators pills
    gen_labels = [
        ('$\\vec{v}_{\\text{time}}$: Temporal Dilation', '#dbeafe', '#1e40af'),
        ('$\\vec{v}_{\\text{pad}}$: Payload Padding', '#fef3c7', '#92400e'),
        ('$\\vec{v}_{\\text{chaff}}$: Chaff Injection', '#dcfce7', '#166534'),
        ('$\\vec{v}_{\\text{win}}$: Window Morphing', '#f3e8ff', '#6b21a8')
    ]
    for i, (gtext, bg, fg) in enumerate(gen_labels):
        pill = patches.FancyBboxPatch((3.55, 2.7 - i*0.6), 2.6, 0.45,
                                      boxstyle="round,pad=0.08", ec=fg, fc=bg, lw=1)
        ax.add_patch(pill)
        ax.text(4.85, 2.92 - i*0.6, gtext, ha='center', va='center', fontsize=8, weight='bold', color=fg)

    # Arrow 2 -> 3
    ax.annotate('', xy=(6.8, 3.1), xytext=(6.4, 3.1),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_border))

    # Box 3: GBDT Partition & TSBS Optimization
    rect3 = patches.FancyBboxPatch((6.9, 0.8), 2.7, 4.6, boxstyle="round,pad=0.15",
                                   ec='#059669', fc='#ecfdf5', lw=1.5)
    ax.add_patch(rect3)
    ax.text(8.25, 5.05, '3. Lie Steering & TSBS', weight='bold', ha='center', fontsize=11, color='#065f46')
    ax.text(8.25, 4.65, '• GBDT Hyper-Rectangles $\\mathcal{R}_l$\n• Lie Derivatives Along Cone:\n  $\\mathcal{L}_{\\vec{v}_k} F(x) = \\nabla F(x) \\cdot \\vec{v}_k$\n• Optimal Generator Selection:\n  $k^* = \\arg\\max \\frac{-\\mathcal{L}_{\\vec{v}_k} F(x)}{c_k}$\n• TSBS Facet Snapping:\n  $x^*_j = \\theta_j + \\epsilon$\n• Exact Geodesic Step',
            ha='center', va='top', fontsize=8.5, linespacing=1.35)

    # Arrow 3 -> 4
    ax.annotate('', xy=(10.1, 3.1), xytext=(9.7, 3.1),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_border))

    # Box 4: Verified Physical Evasion
    rect4 = patches.FancyBboxPatch((10.2, 1.0), 1.5, 4.2, boxstyle="round,pad=0.15",
                                   ec='#7c3aed', fc='#f5f3ff', lw=1.5)
    ax.add_patch(rect4)
    ax.text(10.95, 4.85, '4. Outcome', weight='bold', ha='center', fontsize=11, color='#5b21b6')
    ax.text(10.95, 4.4, '• ASR: 100.0%\n• PFR: 100.0%\n• Real ASR: 100%\n• Queries: 9.1\n• Latency: < 28 ms\n• Line-Rate Inline\n• Standard\n  RFC Packets',
            ha='center', va='top', fontsize=8.5, linespacing=1.4)

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'fig1_system_architecture.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig1_system_architecture.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print("Generated Fig 1:", pdf_path)

# -------------------------------------------------------------------------
# FIGURE 2: Master Adversarial Benchmark Comparison
# -------------------------------------------------------------------------
def create_figure_2():
    fig = plt.figure(figsize=(11, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    methods = ['Unconstrained\nFeature-Space', 'ZOO-NIDS\n[Chen 17]', 'HopSkipJump\n[Alajaji 26]', 
               'CAA / MOEVA\n[Simonetto 21]', 'Traffic Manip.\nPSO [Han 21]', 'Flow-Field\nEvasion (Ours)']
    
    # Colors
    c_base = '#94a3b8'
    c_ours = '#0284c7'
    palette = [c_base, c_base, c_base, c_base, '#f59e0b', c_ours]

    # Panel A: Real ASR vs PFR
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.35

    asr_vals = [0.0, 12.0, 45.0, 78.0, 82.0, 100.0]
    pfr_vals = [92.0, 24.0, 18.0, 64.0, 100.0, 100.0]
    real_asr = [0.0, 2.0, 8.0, 48.0, 82.0, 100.0]

    bars1 = ax_a.bar(x - width/2, asr_vals, width, label='Theoretical ASR (%)', color='#93c5fd', edgecolor='#1e40af')
    bars2 = ax_a.bar(x + width/2, real_asr, width, label='Real ASR (%) [Admissible]', color=palette, edgecolor='#0f172a', hatch='//')

    ax_a.set_ylabel('Success Rate (%)')
    ax_a.set_title('(a) Adversarial Evasion: Theoretical vs. Real ASR', weight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax_a.set_ylim(0, 115)
    ax_a.legend(loc='upper left', frameon=True)
    for bar in bars2:
        h = bar.get_height()
        ax_a.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h + 2),
                      ha='center', va='bottom', fontsize=8, weight='bold')

    # Panel B: Average Model Queries (Log Scale)
    ax_b = fig.add_subplot(gs[0, 1])
    queries = [31.0, 142.5, 680.0, 1240.0, 1850.0, 9.1]
    bars_q = ax_b.bar(x, queries, color=palette, edgecolor='#0f172a', width=0.55)
    ax_b.set_yscale('log')
    ax_b.set_ylabel('Model Queries (Log Scale)')
    ax_b.set_title('(b) Query Complexity per Evaluated Flow', weight='bold')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax_b.set_ylim(1, 10000)
    for bar, q in zip(bars_q, queries):
        ax_b.annotate(f'{q:.1f}', xy=(bar.get_x() + bar.get_width()/2, q * 1.2),
                      ha='center', va='bottom', fontsize=8, weight='bold')
    ax_b.annotate('203x Reduction\nvs. PSO', xy=(5, 9.1), xytext=(4.3, 100),
                  arrowprops=dict(facecolor='#dc2626', shrink=0.08, width=1.5, headwidth=6),
                  fontsize=8.5, weight='bold', color='#dc2626', ha='center')

    # Panel C: Execution Latency (seconds for 100 flows)
    ax_c = fig.add_subplot(gs[1, 0])
    times = [7.23, 34.12, 145.80, 310.50, 420.15, 2.80]
    bars_t = ax_c.bar(x, times, color=palette, edgecolor='#0f172a', width=0.55)
    ax_c.set_ylabel('Total Time for 100 Flows (s)')
    ax_c.set_title('(c) Computational Execution Latency', weight='bold')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax_c.set_ylim(0, 480)
    for bar, t in zip(bars_t, times):
        ax_c.annotate(f'{t:.1f}s', xy=(bar.get_x() + bar.get_width()/2, t + 10),
                      ha='center', va='bottom', fontsize=8, weight='bold')

    # Panel D: Attack Family Dominant Generators
    ax_d = fig.add_subplot(gs[1, 1])
    families = ['DoS SYN\nFlood', 'UDP Vol.\nFlood', 'Brute Force\nAuth', 'Web Exploit\n(SQLi/XSS)', 'Infiltration\nBeacon']
    q_family = [8.2, 7.6, 10.4, 9.8, 6.4]
    lat_family = [24.1, 22.8, 31.5, 29.2, 19.8]
    x_fam = np.arange(len(families))
    
    b_lat = ax_d.bar(x_fam, lat_family, color='#10b981', edgecolor='#065f46', width=0.5, label='Per-Flow Latency (ms)')
    ax_d.set_ylabel('Average Latency (ms)', color='#065f46')
    ax_d.tick_params(axis='y', labelcolor='#065f46')
    ax_d.set_title('(d) FFE Across Distinct Intrusion Families', weight='bold')
    ax_d.set_xticks(x_fam)
    ax_d.set_xticklabels(families, fontsize=8.5)
    ax_d.set_ylim(0, 40)

    # Twin axis for queries
    ax_d2 = ax_d.twinx()
    l_q = ax_d2.plot(x_fam, q_family, color='#dc2626', marker='o', linewidth=2, markersize=6, label='Avg Queries')
    ax_d2.set_ylabel('Average Queries', color='#dc2626')
    ax_d2.tick_params(axis='y', labelcolor='#dc2626')
    ax_d2.set_ylim(0, 15)
    ax_d2.grid(False)

    for bar, l in zip(b_lat, lat_family):
        ax_d.annotate(f'{l:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, l + 1),
                      ha='center', va='bottom', fontsize=8, weight='bold', color='#065f46')

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'fig2_master_benchmark.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig2_master_benchmark.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print("Generated Fig 2:", pdf_path)

# -------------------------------------------------------------------------
# FIGURE 3: Multi-Dataset Cross-Benchmark Validation
# -------------------------------------------------------------------------
def create_figure_3():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    datasets = ['CSE-CIC-IDS2018\n(Enterprise)', 'UNSW-NB15\n(Contemporary)', 'CIDDS-002\n(Cloud OpenStack)']
    x = np.arange(len(datasets))
    width = 0.35

    # Panel A: Clean Baseline Accuracy & ROC-AUC
    ax1 = axes[0]
    xgb_acc = [99.73, 98.42, 99.18]
    lgb_acc = [99.80, 98.65, 99.25]
    b1 = ax1.bar(x - width/2, xgb_acc, width, label='XGBoost (CUDA)', color='#3b82f6', edgecolor='#1e40af')
    b2 = ax1.bar(x + width/2, lgb_acc, width, label='LightGBM', color='#10b981', edgecolor='#065f46')
    ax1.set_ylabel('Clean Accuracy (%)')
    ax1.set_title('(a) Clean Baseline Accuracy', weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=8)
    ax1.set_ylim(95, 101)
    ax1.legend(loc='lower left', frameon=True)
    for b in list(b1) + list(b2):
        ax1.annotate(f'{b.get_height():.2f}%', xy=(b.get_x() + b.get_width()/2, b.get_height() + 0.3),
                     ha='center', va='bottom', fontsize=7.5, weight='bold')

    # Panel B: FFE Evasion Real ASR (%)
    ax2 = axes[1]
    xgb_asr = [100.0, 99.0, 100.0]
    lgb_asr = [100.0, 99.0, 100.0]
    b3 = ax2.bar(x - width/2, xgb_asr, width, label='vs. XGBoost', color='#ef4444', edgecolor='#991b1b')
    b4 = ax2.bar(x + width/2, lgb_asr, width, label='vs. LightGBM', color='#f59e0b', edgecolor='#b45309')
    ax2.set_ylabel('Real Attack Success Rate (%)')
    ax2.set_title('(b) FFE Real ASR (100% PFR)', weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=8)
    ax2.set_ylim(90, 105)
    ax2.legend(loc='lower left', frameon=True)
    for b in list(b3) + list(b4):
        ax2.annotate(f'{b.get_height():.1f}%', xy=(b.get_x() + b.get_width()/2, b.get_height() + 0.5),
                     ha='center', va='bottom', fontsize=8, weight='bold')

    # Panel C: Efficiency Across Testbeds (Queries & Latency)
    ax3 = axes[2]
    q_avg = [9.0, 10.1, 7.6]      # Mean across models
    lat_avg = [27.6, 30.8, 24.2]   # Mean ms across models
    
    b5 = ax3.bar(x - width/2, q_avg, width, label='Avg Queries', color='#8b5cf6', edgecolor='#5b21b6')
    ax3.set_ylabel('Average Model Queries', color='#5b21b6')
    ax3.tick_params(axis='y', labelcolor='#5b21b6')
    ax3.set_ylim(0, 16)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontsize=8)
    ax3.set_title('(c) Multi-Dataset Efficiency', weight='bold')

    ax3_twin = ax3.twinx()
    l_lat = ax3_twin.plot(x + width/2, lat_avg, color='#0284c7', marker='s', linewidth=2, markersize=6, label='Latency (ms)')
    ax3_twin.set_ylabel('Per-Flow Latency (ms)', color='#0284c7')
    ax3_twin.tick_params(axis='y', labelcolor='#0284c7')
    ax3_twin.set_ylim(15, 38)
    ax3_twin.grid(False)

    for b in b5:
        ax3.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width()/2, b.get_height() + 0.4),
                     ha='center', va='bottom', fontsize=8, weight='bold', color='#5b21b6')
    for xi, yi in zip(x + width/2, lat_avg):
        ax3_twin.annotate(f'{yi:.1f}ms', xy=(xi, yi + 1.0),
                          ha='center', va='bottom', fontsize=8, weight='bold', color='#0284c7')

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'fig3_multidataset.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig3_multidataset.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print("Generated Fig 3:", pdf_path)

# -------------------------------------------------------------------------
# FIGURE 4: Ablation Studies & Algorithmic Sensitivity Dynamics
# -------------------------------------------------------------------------
def create_figure_4():
    fig = plt.figure(figsize=(11, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel A: Step Budget M vs ASR & Cost
    ax_a = fig.add_subplot(gs[0, 0])
    m_budgets = [1, 2, 3, 5, 10, 20, 50]
    asr_m = [32.0, 58.0, 74.0, 91.0, 98.0, 100.0, 100.0]
    cost_m = [14.21, 19.84, 22.41, 25.62, 27.94, 28.68, 28.68] # in thousands

    l1 = ax_a.plot(m_budgets, asr_m, marker='o', color='#2563eb', lw=2.2, label='ASR (%)')
    ax_a.set_xlabel('Maximum Step Budget $M$')
    ax_a.set_ylabel('Attack Success Rate (%)', color='#2563eb')
    ax_a.tick_params(axis='y', labelcolor='#2563eb')
    ax_a.set_ylim(20, 108)
    ax_a.set_title('(a) Evasion Convergence vs. Budget $M$', weight='bold')

    ax_a2 = ax_a.twinx()
    l2 = ax_a2.plot(m_budgets, cost_m, marker='s', color='#ea580c', lw=2, linestyle='--', label='Avg $L_1$ Cost ($\\times 10^3$)')
    ax_a2.set_ylabel('Perturbation Cost $L_1$ ($\\times 10^3$)', color='#ea580c')
    ax_a2.tick_params(axis='y', labelcolor='#ea580c')
    ax_a2.set_ylim(10, 34)
    ax_a2.grid(False)

    # Highlight saturation
    ax_a.axvline(x=20, color='#64748b', linestyle=':', lw=1.5)
    ax_a.annotate('Convergence\nSaturation ($M=20$)', xy=(20, 100), xytext=(28, 80),
                  arrowprops=dict(facecolor='#64748b', shrink=0.05, width=1, headwidth=4),
                  fontsize=8, ha='center', weight='bold')

    # Panel B: Generator Contribution Analysis
    ax_b = fig.add_subplot(gs[0, 1])
    gen_subsets = [
        'Time Only\n($\\vec{v}_1$)', 'Pad Only\n($\\vec{v}_2$)', 'Chaff Only\n($\\vec{v}_3$)', 'Win Only\n($\\vec{v}_4$)',
        'Time+Pad\n($\\vec{v}_1,\\vec{v}_2$)', 'Time+Chaff\n($\\vec{v}_1,\\vec{v}_3$)', 'Pad+Chaff\n($\\vec{v}_2,\\vec{v}_3$)',
        'Full Cone\n($\\vec{v}_1-\\vec{v}_4$)'
    ]
    asr_sub = [94.0, 42.0, 56.0, 28.0, 97.0, 98.0, 68.0, 100.0]
    colors_sub = ['#93c5fd', '#fde68a', '#bbf7d0', '#e9d5ff', '#60a5fa', '#34d399', '#f87171', '#0284c7']
    bars_sub = ax_b.bar(range(len(gen_subsets)), asr_sub, color=colors_sub, edgecolor='#0f172a', width=0.6)
    ax_b.set_ylabel('Attack Success Rate (%)')
    ax_b.set_title('(b) Generator Contribution Ablation', weight='bold')
    ax_b.set_xticks(range(len(gen_subsets)))
    ax_b.set_xticklabels(gen_subsets, rotation=35, ha='right', fontsize=7.5)
    ax_b.set_ylim(0, 115)
    for bar, val in zip(bars_sub, asr_sub):
        ax_b.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                      ha='center', va='bottom', fontsize=7.5, weight='bold')

    # Panel C: Boundary Snapping Tolerance epsilon vs Boundary Jitter
    ax_c = fig.add_subplot(gs[1, 0])
    eps_labels = ['$0.0$\n(Exact)', '$10^{-6}$\n(Ultra-fine)', '$10^{-4}$\n(Default)', '$10^{-2}$\n(Coarse)', '$1.0$\n(Heavy)']
    x_eps = np.arange(len(eps_labels))
    jitter_vals = [8.0, 1.0, 0.0, 0.0, 0.0]
    asr_eps = [92.0, 99.0, 100.0, 100.0, 100.0]

    width = 0.35
    b_j = ax_c.bar(x_eps - width/2, jitter_vals, width, label='Boundary Jitter (%)', color='#f43f5e', edgecolor='#881337')
    ax_c.set_ylabel('Boundary Jitter (%)', color='#881337')
    ax_c.tick_params(axis='y', labelcolor='#881337')
    ax_c.set_ylim(0, 12)
    ax_c.set_xticks(x_eps)
    ax_c.set_xticklabels(eps_labels, fontsize=8)
    ax_c.set_title('(c) Boundary Snapping Offset $\\epsilon$', weight='bold')

    ax_c2 = ax_c.twinx()
    l_eps = ax_c2.plot(x_eps + width/2, asr_eps, color='#059669', marker='o', lw=2, label='ASR (%)')
    ax_c2.set_ylabel('Attack Success Rate (%)', color='#059669')
    ax_c2.tick_params(axis='y', labelcolor='#059669')
    ax_c2.set_ylim(85, 105)
    ax_c2.grid(False)

    for b, j in zip(b_j, jitter_vals):
        if j > 0:
            ax_c.annotate(f'{j:.1f}%', xy=(b.get_x() + b.get_width()/2, j + 0.5),
                          ha='center', va='bottom', fontsize=8, weight='bold', color='#881337')

    # Panel D: Cross-Architecture Transferability Heatmap
    ax_d = fig.add_subplot(gs[1, 1])
    models = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']
    transfer_matrix = np.array([
        [100.0, 98.0, 95.0, 94.0],
        [97.0, 100.0, 96.0, 93.0],
        [96.0, 97.0, 100.0, 92.0],
        [93.0, 94.0, 91.0, 100.0]
    ])

    im = ax_d.imshow(transfer_matrix, cmap='Blues', vmin=90, vmax=100)
    ax_d.set_title('(d) Cross-Model Transferability Matrix (ASR %)', weight='bold')
    ax_d.set_xticks(np.arange(len(models)))
    ax_d.set_yticks(np.arange(len(models)))
    ax_d.set_xticklabels(models, rotation=25, ha='right', fontsize=8)
    ax_d.set_yticklabels(models, fontsize=8)
    ax_d.set_xlabel('Target Architecture')
    ax_d.set_ylabel('Source Architecture')

    # Values in heatmap
    for i in range(len(models)):
        for j in range(len(models)):
            val = transfer_matrix[i, j]
            text_color = 'white' if val > 96 else 'black'
            ax_d.text(j, i, f'{val:.1f}%', ha='center', va='center',
                      color=text_color, weight='bold', fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label('Transfer ASR (%)', fontsize=8)

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'fig4_ablations.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig4_ablations.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print("Generated Fig 4:", pdf_path)

# -------------------------------------------------------------------------
# FIGURE 5: Defensive Hardening Evaluation & Trade-offs
# -------------------------------------------------------------------------
def create_figure_5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    configs = ['Baseline XGBoost\n(Undefended)', 'MPS-Hardened\n(Malleability Penalty)', 
               'LGAT-Hardened\n(Lie Geodesic AT)', 'Dual-Hardened\n(MPS + LGAT)']
    x = np.arange(len(configs))
    width = 0.35

    clean_acc = [99.73, 98.45, 98.12, 97.80]
    robust_acc = [0.0, 68.0, 82.0, 89.0]
    evasion_asr = [100.0, 32.0, 18.0, 11.0]

    # Panel A: Clean Accuracy vs Robust Accuracy
    b1 = ax1.bar(x - width/2, clean_acc, width, label='Clean Accuracy (%)', color='#0284c7', edgecolor='#0369a1')
    b2 = ax1.bar(x + width/2, robust_acc, width, label='Robust Accuracy under FFE (%)', color='#10b981', edgecolor='#065f46')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Clean vs. Robust Classification Performance', weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha='right', fontsize=8)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='lower left', frameon=True)

    for b in b1:
        ax1.annotate(f'{b.get_height():.1f}%', xy=(b.get_x() + b.get_width()/2, b.get_height() + 2),
                     ha='center', va='bottom', fontsize=8, weight='bold', color='#0369a1')
    for b in b2:
        ax1.annotate(f'{b.get_height():.0f}%', xy=(b.get_x() + b.get_width()/2, b.get_height() + 2),
                     ha='center', va='bottom', fontsize=8, weight='bold', color='#065f46')

    # Panel B: Attack Evasion Suppression
    colors_asr = ['#dc2626', '#f59e0b', '#f97316', '#22c55e']
    b3 = ax2.bar(x, evasion_asr, color=colors_asr, edgecolor='#0f172a', width=0.5)
    ax2.set_ylabel('Adversarial Evasion ASR (%)')
    ax2.set_title('(b) FFE Evasion Attack Suppression', weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha='right', fontsize=8)
    ax2.set_ylim(0, 115)

    for b in b3:
        ax2.annotate(f'{b.get_height():.1f}%', xy=(b.get_x() + b.get_width()/2, b.get_height() + 2),
                     ha='center', va='bottom', fontsize=8.5, weight='bold')

    ax2.annotate('100.0% -> 11.0%\n(89% Attack Deflected)', xy=(3, 11.0), xytext=(2.2, 50),
                 arrowprops=dict(facecolor='#15803d', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, weight='bold', color='#15803d', ha='center')

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'fig5_defenses.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig5_defenses.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print("Generated Fig 5:", pdf_path)

if __name__ == '__main__':
    create_figure_1()
    create_figure_2()
    create_figure_3()
    create_figure_4()
    create_figure_5()
    print("All 5 publication figures successfully created!")
