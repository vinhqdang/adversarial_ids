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
    fig = plt.figure(figsize=(14, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # -------------------------------------------------------------------------
    # Panel (a): The Physical Flow Manifold & Lie Cone
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Feature space: Flow Duration (s) vs Forward Byte Rate (KB/s)
    # Conservation law: Rate = Total_Bytes / Duration
    duration = np.linspace(0.4, 4.5, 300)
    
    # Manifold curves for different total byte levels (hyperbolas)
    for bytes_val, col, ls, lw, lbl in [
        (40, '#cbd5e1', ':', 1.0, None),
        (60, '#94a3b8', '--', 1.0, None),
        (80, '#0284c7', '-', 2.0, r'Manifold Iso-Byte $B = 80$ KB'),
        (100, '#64748b', '--', 1.0, None),
        (120, '#cbd5e1', ':', 1.0, None)
    ]:
        rate = bytes_val / duration
        mask = (rate >= 10) & (rate <= 98)
        ax1.plot(duration[mask], rate[mask], color=col, linestyle=ls, linewidth=lw, label=lbl)

    # Base malicious point x_0 on B=80 curve
    x0_dur = 0.95
    x0_rate = 80.0 / x0_dur  # ~84.2 KB/s
    ax1.plot(x0_dur, x0_rate, marker='o', markersize=8, color='#dc2626', zorder=6)
    ax1.text(x0_dur, x0_rate + 8.5, r'$x_0$ (Active Attack)', weight='bold', color='#dc2626', fontsize=9.0, ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#fca5a5', alpha=0.9))

    # Lie Generators at x_0
    # v_time: moves along the hyperbola (dilutes rate, extends duration)
    arrow_len = 0.65
    target_dur = x0_dur + arrow_len
    target_rate = 80.0 / target_dur
    ax1.annotate('', xy=(target_dur, target_rate), xytext=(x0_dur, x0_rate),
                 arrowprops=dict(facecolor='#0284c7', edgecolor='#0369a1', width=2.5, headwidth=8, shrink=0))
    ax1.text(target_dur + 0.06, target_rate + 2.5, r'$\vec{v}_{\mathrm{time}}$ (Dilate Rate)', 
             color='#0369a1', weight='bold', fontsize=8.5, va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bae6fd', alpha=0.9))

    # v_pad: increases payload bytes at fixed duration (moves to higher hyperbola)
    ax1.annotate('', xy=(x0_dur, x0_rate + 22), xytext=(x0_dur, x0_rate),
                 arrowprops=dict(facecolor='#d97706', edgecolor='#b45309', width=2.5, headwidth=8, shrink=0))
    ax1.text(x0_dur + 0.08, x0_rate + 19, r'$\vec{v}_{\mathrm{pad}}$ (Pad Payload)', 
             color='#b45309', weight='bold', fontsize=8.5, va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#fde68a', alpha=0.9))

    # v_chaff: adds chaff packets (intermediate tangent direction)
    ax1.annotate('', xy=(x0_dur + 0.45, x0_rate + 8), xytext=(x0_dur, x0_rate),
                 arrowprops=dict(facecolor='#16a34a', edgecolor='#15803d', width=2.0, headwidth=7, shrink=0))
    ax1.text(x0_dur + 0.50, x0_rate + 8, r'$\vec{v}_{\mathrm{chaff}}$ (Chaff)', 
             color='#15803d', weight='bold', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bbf7d0', alpha=0.9))

    # Shaded Lie Cone region
    cone_x = [x0_dur, target_dur, x0_dur + 0.45, x0_dur]
    cone_y = [x0_rate, target_rate, x0_rate + 22, x0_rate]
    ax1.fill(cone_x, cone_y, color='#0ea5e9', alpha=0.18, label=r'Lie Cone $\mathcal{K}_{\mathrm{net}}(x_0)$ ($u_k \geq 0$)')

    # Tangent Space plane line
    tan_t = np.linspace(0.6, 1.4, 50)
    slope = -80.0 / (x0_dur**2)
    tan_r = x0_rate + slope * (tan_t - x0_dur)
    ax1.plot(tan_t, tan_r, color='#94a3b8', linestyle='-.', linewidth=1.1, label=r'Tangent Space $T_{x_0}\mathcal{M}_{\mathrm{flow}}$')

    # Inadmissible / off-manifold indicator badge
    ax1.text(3.15, 75, r'$\mathbf{Off\text{-}Manifold\ Region}$' + '\n' + r'$\Psi(x) \neq 0$ (Non-Routable Flows)' + '\n' + r'$0\%$ Physical Feasibility',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#f87171', alpha=0.9),
             fontsize=8.5, color='#991b1b', ha='center')

    ax1.set_xlabel('Flow Duration $x_6$ (seconds)')
    ax1.set_ylabel('Forward Byte Rate $x_8$ (KB/s)')
    ax1.set_title(r'(a) Physical Flow Manifold $\mathcal{M}_{\mathrm{flow}}$ & Lie Cone $\mathcal{K}_{\mathrm{net}}$', weight='bold')
    ax1.set_xlim(0.4, 4.4)
    ax1.set_ylim(10, 115)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel (b): Decision Tree Partition: FFE vs. Unconstrained AML
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    # GBDT Partition Regions
    dur_grid = np.linspace(0.4, 4.4, 200)
    rate_grid = np.linspace(10, 115, 200)
    D, R = np.meshgrid(dur_grid, rate_grid)

    Z = np.zeros_like(D)
    Z[(D < 1.6) & (R > 60)] = 2.5
    Z[(D >= 1.6) & (D < 2.5) & (R > 45)] = 1.6
    Z[(D >= 2.5) & (D < 3.4) & (R > 32)] = 0.8
    Z[(D < 2.2) & (R <= 60) & (R > 40)] = 0.9
    Z[(D >= 3.4) | (R <= 32)] = -1.5
    Z[(D >= 2.2) & (R <= 40) & (R > 25)] = -0.7

    # Plot decision regions: Red for Malicious (Z > 0), Green for Benign (Z <= 0)
    cmap = matplotlib.colors.ListedColormap(['#dcfce7', '#fee2e2'])
    bounds = [-3.0, 0.0, 3.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax2.contourf(D, R, Z, levels=[-3.0, 0.0, 3.0], cmap=cmap, alpha=0.75)
    
    # Draw tree split boundary lines
    ax2.contour(D, R, Z, levels=[0.0], colors='#0f172a', linewidths=2.0, linestyles='-')

    # Region labels
    ax2.text(1.1, 102, r'$\mathbf{MALICIOUS\ REGION}$' + '\n' + r'$F(x) \geq 0$ (NIDS Alert)', 
             color='#991b1b', weight='bold', fontsize=9.5, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fecaca', edgecolor='#f87171', alpha=0.7))
    ax2.text(1.8, 16, r'$\mathbf{BENIGN\ REGION}$' + '\n' + r'$F(x) < 0$ (Normal)', 
             color='#166534', weight='bold', fontsize=9.0, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#bbf7d0', edgecolor='#86efac', alpha=0.85))

    # Split threshold markers
    ax2.axvline(x=2.5, color='#475569', linestyle=':', linewidth=0.9)
    ax2.axvline(x=3.4, color='#475569', linestyle=':', linewidth=0.9)
    ax2.axhline(y=45, color='#475569', linestyle=':', linewidth=0.9)
    ax2.axhline(y=32, color='#475569', linestyle=':', linewidth=0.9)
    ax2.text(2.52, 12, r'$\theta_{\mathrm{dur},1}$', fontsize=8, color='#334155')
    ax2.text(3.42, 12, r'$\theta_{\mathrm{dur},2}$', fontsize=8, color='#334155')
    ax2.text(0.45, 46, r'$\theta_{\mathrm{rate},1}$', fontsize=8, color='#334155')
    ax2.text(0.45, 33, r'$\theta_{\mathrm{rate},2}$', fontsize=8, color='#334155')

    # Initial Attack Point x_0
    ax2.plot(0.95, 84.2, marker='o', markersize=8, color='#dc2626', zorder=6)
    ax2.text(0.95, 89.0, r'$x_0$', color='#dc2626', weight='bold', fontsize=9.5, ha='center')

    # Path 1: Unconstrained Attack (PGD / MILP)
    ax2.annotate('', xy=(0.95, 36), xytext=(0.95, 84.2),
                 arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=2.2, linestyle='--'))
    ax2.plot(0.95, 36, marker='x', markersize=10, mew=3, color='#b91c1c', zorder=6)
    ax2.text(1.55, 68, r'$\mathbf{Unconstrained\ AML}$' + '\n' + r'(Leaves $\mathcal{M}_{\mathrm{flow}}$: $0\%$ PFR)' + '\n' + r'$\times$ TCP Packets Dropped', 
             color='#b91c1c', weight='bold', fontsize=7.5, ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff', edgecolor='#fca5a5', alpha=0.95))
    ax2.annotate('', xy=(0.97, 60), xytext=(1.18, 66),
                 arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.2))

    # Path 2: Flow-Field Evasion (Ours) - Steers along curved manifold with TSBS
    ffe_path_d = [0.95, 1.45, 1.95, 2.52, 3.0, 3.42, 3.75]
    ffe_path_r = [84.2, 55.2, 41.0, 35.5, 28.5, 25.0, 22.5]
    ax2.plot(ffe_path_d, ffe_path_r, color='#0284c7', linewidth=2.5, marker='o', 
             markersize=5, zorder=6, label=r'FFE Geodesic on $\mathcal{M}_{\mathrm{flow}}$')
    
    # Highlight final evasive point
    ax2.plot(3.75, 22.5, marker='*', markersize=15, color='#15803d', zorder=7)
    ax2.text(3.75, 30.5, r'$x^*$ (Evaded & 100% PFR)', color='#15803d', weight='bold', fontsize=8.5, ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0fdf4', edgecolor='#86efac', alpha=0.95))

    # TSBS annotation
    ax2.annotate(r'TSBS Boundary Snap' + '\n' + r'$x_j^* = \theta_j + \epsilon$', xy=(2.52, 35.5), xytext=(2.95, 48.0),
                 arrowprops=dict(facecolor='#0f172a', shrink=0.08, width=1.0, headwidth=5),
                 fontsize=8, weight='bold', color='#0f172a',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff', edgecolor='#cbd5e1'))

    ax2.set_xlabel('Flow Duration $x_6$ (seconds)')
    ax2.set_ylabel('Forward Byte Rate $x_8$ (KB/s)')
    ax2.set_title('(b) Steering Through Tree Partitions: FFE vs. Unconstrained AML', weight='bold')
    ax2.set_xlim(0.4, 4.4)
    ax2.set_ylim(10, 115)
    ax2.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel (c): Real-World Packet Stream Mutations (Phi^-1 Mapping)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6.2)
    ax3.axis('off')

    # Top stream: Original Malicious Attack (Dense burst, identical packets)
    rect_top = patches.FancyBboxPatch((0.3, 3.45), 9.4, 2.65, boxstyle="round,pad=0.06",
                                      ec='#dc2626', fc='#fef2f2', lw=1.5)
    ax3.add_patch(rect_top)
    ax3.text(0.55, 5.80, r'$\mathbf{Original\ Intrusion\ Packet\ Stream\ \mathcal{P}_0\ (High\ Volumetric\ Burst)}$', 
             weight='bold', fontsize=9.2, color='#991b1b')
    ax3.text(0.55, 5.40, r'$\bullet$ Inter-arrival $\Delta t \approx 0.01$ ms  $\bullet$ Byte Rate $x_8 = 84.2$ KB/s  $\bullet$ Classified Malicious ($F(x_0)=+3.4$)',
             fontsize=7.8, color='#7f1d1d')
    ax3.text(0.55, 5.02, r'$\bullet$ RFC 793 SYN handshake triggers volumetric rate alerts in standard NIDS telemetry',
             fontsize=7.5, color='#991b1b', style='italic')

    # Draw packets in original burst (cleanly separated at y=3.68)
    burst_x = [3.8, 4.3, 4.8, 5.3, 5.8, 6.3, 6.8, 7.3, 7.8, 8.3]
    for px in burst_x:
        ax3.add_patch(patches.Rectangle((px, 3.68), 0.32, 0.95, facecolor='#ef4444', edgecolor='#b91c1c', lw=1))
        ax3.text(px + 0.16, 4.155, 'SYN', fontsize=6.5, weight='bold', color='white', ha='center', va='center', rotation=90)
    ax3.text(3.5, 4.155, r'Bursts $\to$', fontsize=7.5, color='#991b1b', weight='bold', ha='right', va='center')

    # Middle transformation badge & connectors
    # Downward connectors
    ax3.annotate('', xy=(5.0, 3.25), xytext=(5.0, 3.45),
                 arrowprops=dict(facecolor='#0284c7', edgecolor='#0369a1', width=2.0, headwidth=6))
    
    # Transformation Pill Box
    pill = patches.FancyBboxPatch((1.0, 2.82), 8.0, 0.46, boxstyle="round,pad=0.06",
                                  ec='#0284c7', fc='#e0f2fe', lw=1.2)
    ax3.add_patch(pill)
    ax3.text(5.0, 3.05, r'$\mathbf{Inline\ Proxy\ \Phi^{-1}}\text{: Dilate } (\vec{v}_{\mathrm{time}}) + \text{Pad } (\vec{v}_{\mathrm{pad}}) + \text{Chaff } (\vec{v}_{\mathrm{chaff}})$',
             fontsize=8.2, weight='bold', color='#0369a1', ha='center', va='center')

    ax3.annotate('', xy=(5.0, 2.65), xytext=(5.0, 2.82),
                 arrowprops=dict(facecolor='#0284c7', edgecolor='#0369a1', width=2.0, headwidth=6))

    # Bottom stream: FFE Evasive Stream (Dilated, padded, chaffed)
    rect_bot = patches.FancyBboxPatch((0.3, 0.08), 9.4, 2.55, boxstyle="round,pad=0.06",
                                      ec='#16a34a', fc='#f0fdf4', lw=1.5)
    ax3.add_patch(rect_bot)
    ax3.text(0.55, 2.35, r'$\mathbf{FFE\ Evasive\ Packet\ Stream\ \mathcal{P}^*\ (Strictly\ RFC\ 791/793\ Compliant)}$', 
             weight='bold', fontsize=9.2, color='#166534')
    ax3.text(0.55, 1.95, r'$\bullet$ Dilated Inter-Arrival $\Delta t = 0.45$ s  $\bullet$ Payload Padding $+64$ B  $\bullet$ Classified Benign ($F(x^*)=-0.85$)',
             fontsize=7.8, color='#14532d')
    ax3.text(0.55, 1.58, r'$\bullet$ Kernel TCP state machines preserved, valid IP checksums, 100% routable physical flows',
             fontsize=7.5, color='#166534', style='italic')

    # Draw dilated packets with padding and chaff at y=0.42
    dilated = [
        (3.4, '#3b82f6', '#1d4ed8', 'SYN\n+Pad', 0.55, 0.88),
        (4.5, '#22c55e', '#15803d', 'ACK\nChaff', 0.42, 0.75),
        (5.6, '#3b82f6', '#1d4ed8', 'DATA\n+Pad', 0.60, 0.88),
        (6.9, '#22c55e', '#15803d', 'ACK\nChaff', 0.42, 0.75),
        (8.0, '#3b82f6', '#1d4ed8', 'DATA\n+Pad', 0.60, 0.88),
    ]
    for px, fc, ec, lbl, w, h in dilated:
        ax3.add_patch(patches.Rectangle((px, 0.42), w, h, facecolor=fc, edgecolor=ec, lw=1))
        ax3.text(px + w/2, 0.42 + h/2, lbl, fontsize=6.5, weight='bold', color='white', ha='center', va='center')

    # Timeline arrow
    ax3.annotate('', xy=(9.0, 0.28), xytext=(3.0, 0.28),
                 arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    ax3.text(8.8, 0.16, 'Time $t$ (seconds)', fontsize=7.5, color='#64748b', ha='right')

    ax3.set_title(r'(c) Inline Packet Stream Translation ($\Phi^{-1}$ Scapy Proxy)', weight='bold')

    # -------------------------------------------------------------------------
    # Panel (d): TSBS Decision Score Convergence Trajectory
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    steps = np.arange(0, 9)
    scores = np.array([3.4, 2.5, 1.7, 1.05, 0.62, 0.18, -0.42, -0.85, -0.85])
    
    # Line plot with markers
    ax4.plot(steps, scores, color='#0284c7', linewidth=2.4, marker='o', markersize=6, zorder=4)

    # Shaded threshold zones
    ax4.axhspan(0, 4.0, facecolor='#fee2e2', alpha=0.5, label=r'Malicious Zone ($F(x) \geq 0$)')
    ax4.axhspan(-1.6, 0, facecolor='#dcfce7', alpha=0.5, label=r'Benign Zone ($F(x) < 0$)')
    ax4.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5, label=r'Decision Boundary ($F=0$)')

    # Annotate steps
    gen_annotations = [
        (1, r'$\vec{v}_{\mathrm{time}}$', 'bottom', -0.38),
        (2, r'$\vec{v}_{\mathrm{time}}$', 'bottom', -0.38),
        (3, r'TSBS Snap $\theta_1$', 'top', 0.32),
        (4, r'$\vec{v}_{\mathrm{pad}}$', 'bottom', -0.38),
        (5, r'$\vec{v}_{\mathrm{chaff}}$', 'bottom', -0.38),
        (6, r'TSBS Snap $\theta_2$', 'bottom', -0.38),
        (7, r'$\vec{v}_{\mathrm{win}}$', 'top', 0.32)
    ]
    for s_idx, txt, v_align, offset in gen_annotations:
        ax4.text(s_idx, scores[s_idx] + offset, txt, fontsize=7.5, weight='bold',
                 ha='center', color='#0f172a',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cbd5e1', alpha=0.92))

    ax4.set_xlabel('FFE Optimization Step $m$')
    ax4.set_ylabel(r'Classifier Decision Score $F(x_m)$ (Log-Odds)')
    ax4.set_title('(d) TSBS Decision Score Convergence Trajectory', weight='bold')
    ax4.set_xlim(-0.4, 8.4)
    ax4.set_ylim(-1.6, 3.8)
    ax4.set_xticks(steps)
    ax4.legend(loc='upper right', framealpha=0.9, fontsize=8)

    # Success badge placed safely on left
    ax4.text(1.3, -1.05, r'$\mathbf{Real\ ASR = 100\%}$' + '\n' + r'$M = 7\ \mathrm{steps},\ 9.1\ \mathrm{queries}$' + '\n' + r'$\leq 28\ \mathrm{ms\ Latency}$',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#dcfce7', edgecolor='#16a34a', lw=1.2),
             fontsize=8.0, color='#14532d', ha='center', weight='bold')

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
