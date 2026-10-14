"""
Renders Ablation Studies 1-4 of paper/sec8_ablations.tex from results/ablation_results.json
so that every reported number is traceable to experiments/run_ablations.py.

Only the region between the markers
    % BEGIN-GENERATED-ABLATIONS-1-4   and   % END-GENERATED-ABLATIONS-1-4
is rewritten; the markers are inserted on first run (before 'Ablation Study 1' and 'Ablation Study 5').
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEX = os.path.join(ROOT, 'paper', 'sec8_ablations.tex')
RESULTS = os.path.join(ROOT, 'results', 'ablation_results.json')
BEGIN = '% BEGIN-GENERATED-ABLATIONS-1-4'
END = '% END-GENERATED-ABLATIONS-1-4'
PCT = '\\%'
GEN_MACRO = {
    'time': '\\vec{v}_{\\text{time}}', 'pad': '\\vec{v}_{\\text{pad}}',
    'chaff': '\\vec{v}_{\\text{chaff}}', 'win': '\\vec{v}_{\\text{win}}',
}
GEN_LABELS = [
    ('Temporal Dilation Only', '\\textbf{Temporal Dilation Only} ($\\vec{v}_{\\text{time}}$)'),
    ('Payload Padding Only', '\\textbf{Payload Padding Only} ($\\vec{v}_{\\text{pad}}$)'),
    ('Chaff Injection Only', '\\textbf{Chaff Injection Only} ($\\vec{v}_{\\text{chaff}}$)'),
    ('Window Morphing Only', '\\textbf{Window Morphing Only} ($\\vec{v}_{\\text{win}}$)'),
    None,
    ('Time + Padding', '\\textbf{Pairwise: Time + Padding} ($\\vec{v}_1, \\vec{v}_2$)'),
    ('Time + Chaff', '\\textbf{Pairwise: Time + Chaff} ($\\vec{v}_1, \\vec{v}_3$)'),
    ('Padding + Chaff', '\\textbf{Pairwise: Padding + Chaff} ($\\vec{v}_2, \\vec{v}_3$)'),
    None,
    ('Complete Lie Cone', '\\textbf{Complete Lie Cone} ($\\vec{v}_1, \\vec{v}_2, \\vec{v}_3, \\vec{v}_4$)'),
]


def pct(value: float) -> str:
    return f'{value:.1f}{PCT}'


def table(caption: str, label: str, colspec: str, header: str, rows: list) -> str:
    body = '\n'.join(rows)
    return (
        '\\begin{table*}[htbp]\n'
        f'\\caption{{{caption}}}\\label{{{label}}}%\n'
        '\\centering\n\\resizebox{\\textwidth}{!}{%\n'
        f'\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n'
        f'{body}\n\\bottomrule\n\\end{{tabular}}%\n}}\n\\end{{table*}}\n'
    )


def render(res: dict) -> str:
    n = res['cohort_size']
    b, g, c, e = res['budget'], res['generators'], res['costs'], res['epsilon']
    budget_rows = [
        f"{m} & {pct(r['asr'])} & {pct(r['pfr'])} & {r['avg_queries']:.1f} & {r['avg_cost_l1']:.3f} & {r['total_time_s']:.2f}s \\\\"
        for m, r in b.items()
    ]
    gen_rows = []
    for item in GEN_LABELS:
        if item is None:
            gen_rows.append('\\midrule')
            continue
        key, label = item
        r = g[key]
        cells = [pct(r['asr']), pct(r['pfr']), f"{r['avg_queries']:.1f}", f"{r['mean_delay_ms'] / 1000.0:.2f}"]
        if key == 'Complete Lie Cone':
            cells = [f'\\textbf{{{x}}}' for x in cells]
            label = label.replace('\\textbf{Complete Lie Cone}', '\\textbf{Complete Lie Cone}')
        gen_rows.append(f"{label} & {' & '.join(cells)} \\\\")
    cost_rows = []
    for key, r in c.items():
        dom = GEN_MACRO.get(r['dominant_generator'], '-')
        cost_rows.append(
            f"\\textbf{{Regime {key}}} & {pct(r['asr'])} & {r['avg_queries']:.1f} & ${dom}$ (${r['dominant_share']:.0f}{PCT}$) "
            f"& ${r['mean_delay_ms']:.0f}$ ms \\\\"
        )
    eps_rows = [
        f"${float(k):g}$ & {pct(r['asr'])} & {pct(r['pfr'])} & {r['avg_queries']:.1f} & {r['avg_cost_l1']:.3f} \\\\"
        for k, r in e.items()
    ]
    a, bb, cc, dd = c.values()
    t = g['Temporal Dilation Only']
    full = g['Complete Lie Cone']
    m1, m5, m10 = b['1'], b['5'], b['10']
    parts = [
        'In operational deployments, an attacker may face strict query throttling or inspection time limits. '
        'Table~\\ref{tab:ablation_budget_expanded} evaluates FFE across seven step budgets: '
        '$M \\in \\{1, 2, 3, 5, 10, 20, 50\\}$. All ablations use the balanced cohort of $N = ' + str(n) +
        '$ correctly detected malicious flows described below and the CUDA XGBoost target of Section~\\ref{sec:experiments}; '
        'they are reproduced by \\texttt{experiments/run\\_ablations.py}. With $N = ' + str(n) +
        '$, one flow corresponds to one percentage point, so differences of a few points are within sampling noise.\n',
        table('Ablation 1: Evasion Performance vs. Maximum Step Budget $M$', 'tab:ablation_budget_expanded', 'cccccc',
              '\\textbf{Step Budget $M$} & \\textbf{ASR (\\%)} & \\textbf{PFR (\\%)} & \\textbf{Avg Queries} & '
              '\\textbf{Avg Cost ($\\bar{D}_{L_1}$)} & \\textbf{Total Time (s)}', budget_rows),
        '\\paragraph{Key Insights on Step Budget.}\n\\begin{itemize}\n'
        f"    \\item \\textbf{{Rapid Early Convergence:}} With $M = 1$, {pct(m1['asr'])} of flows already cross the boundary "
        f"({m1['avg_queries']:.1f} queries on average); ASR reaches {pct(b['2']['asr'])} at $M = 2$, {pct(b['3']['asr'])} at $M = 3$ "
        f"and {pct(m5['asr'])} at $M = 5$ ({m5['avg_queries']:.1f} queries), and is {pct(m10['asr'])} from $M = 10$ onward.\n"
        f"    \\item \\textbf{{Saturation:}} Budgets $M = 10, 20, 50$ give identical results "
        f"({m10['avg_queries']:.1f} queries, cost {m10['avg_cost_l1']:.3f}) because every flow terminates within ten iterations.\n"
        f"    \\item \\textbf{{Monotonic Cost Scaling:}} The average normalized perturbation cost grows from {m1['avg_cost_l1']:.3f} "
        f"at $M = 1$ to {m10['avg_cost_l1']:.3f} and then stabilizes, reflecting the distance to the benign decision region.\n"
        '\\end{itemize}\n',
        '\\subsubsection{Ablation Study 2: Generator Contribution Analysis}\n'
        'To quantify the role of each elementary network transformation, we evaluate FFE when individual generators or pairs of '
        'generators are disabled; a disabled generator is removed both from the directional probes of Algorithm~\\ref{alg:lie_estimator} '
        'and from the candidate set of TSBS (Algorithm~\\ref{alg:tsbs}), and the temporal-dilation fallback is available only when '
        '$\\vec{v}_{\\text{time}}$ is enabled. The cohort contains ' + str(n) + ' detected flows drawn from 20 generated flows per attack family. '
        'Table~\\ref{tab:ablation_generators} reports the results; the last column is the mean added flow duration in seconds.\n',
        table('Ablation 2: Individual and Combinatorial Generator Contributions', 'tab:ablation_generators', 'lcccc',
              '\\textbf{Active Generator Subset} & \\textbf{ASR (\\%)} & \\textbf{PFR (\\%)} & \\textbf{Avg Queries} & '
              '\\textbf{Mean Added Duration (s)}', gen_rows),
        '\\paragraph{Key Insights on Generator Mechanics.}\n\\begin{enumerate}\n'
        f"    \\item \\textbf{{Primacy of Temporal Dilation ($\\vec{{v}}_{{\\text{{time}}}}$):}} Acting alone, temporal dilation achieves "
        f"{pct(t['asr'])} ASR, the strongest single generator, because decision trees split heavily on the rate features $x_8, x_9$. "
        f"It does so at a high operational price: a mean of {t['mean_delay_ms'] / 1000.0:.1f}\\,s of added flow duration and "
        f"{t['avg_queries']:.1f} queries, versus {full['mean_delay_ms'] / 1000.0:.2f}\\,s and {full['avg_queries']:.1f} queries for the full cone.\n"
        f"    \\item \\textbf{{Necessity of the Full Cone:}} No proper subset reaches $100\\%$: padding alone ({pct(g['Payload Padding Only']['asr'])}) "
        f"and chaff alone ({pct(g['Chaff Injection Only']['asr'])}) are weak, window morphing alone reaches {pct(g['Window Morphing Only']['asr'])}, and the best pair "
        f"(time + chaff) reaches {pct(g['Time + Chaff']['asr'])}. Adding the remaining generators lets the cost-aware selector avoid long delays "
        f"while closing the gap to {pct(full['asr'])}.\n"
        '    \\item \\textbf{Zero-Feasibility Failure Avoided:} For every subset the Physical Feasibility Rate remains $100.0\\%$, '
        'consistent with Theorem~\\ref{thm:invariant_universal}; feasibility holds by construction of the projection, independently of which generators run.\n'
        '\\end{enumerate}\n',
        '\\subsubsection{Ablation Study 3: Sensitivity to Physical Operational Costs}\n'
        'In Algorithm~\\ref{alg:ffe_master}, the cost vector $\\vec{c} = [c_1, c_2, c_3, c_4]^T$ determines the efficiency penalty '
        '$(-\\nabla_k) / c_k$. We evaluate four operational cost regimes:\n'
        '\\begin{itemize}\n'
        '    \\item \\textbf{Regime A (Default / Balanced):} $c_1 = 10^{-4}$, $c_2 = 10^{-2}$, $c_3 = 1.0$, $c_4 = 10^{-3}$.\n'
        '    \\item \\textbf{Regime B (Latency-Sensitive):} $c_1 = 10.0$ (delay is penalized heavily), $c_2 = 10^{-3}, c_3 = 0.1, c_4 = 10^{-3}$.\n'
        '    \\item \\textbf{Regime C (Unconstrained Delay):} $c_1 = 10^{-5}, c_2 = 10.0, c_3 = 10.0, c_4 = 10^{-4}$.\n'
        '    \\item \\textbf{Regime D (Uniform / Flat):} $c_1 = c_2 = c_3 = c_4 = 1.0$.\n'
        '\\end{itemize}\n'
        'The ``dominant generator\'\' is the generator most often selected by the directional step of Algorithm~\\ref{alg:ffe_master} '
        '(TSBS edits are not attributed to a generator).\n',
        table('Ablation 3: Sensitivity to Operational Cost Regimes', 'tab:ablation_costs', 'lcccc',
              '\\textbf{Cost Regime} & \\textbf{ASR (\\%)} & \\textbf{Avg Queries} & \\textbf{Dominant Generator} & '
              '\\textbf{Mean Delay Added}', cost_rows),
        f"All four regimes reach {pct(a['asr'])} ASR. Penalizing delay (Regime B, $c_1 = 10$) reduces the mean added duration from "
        f"{a['mean_delay_ms']:.0f} ms (Regime A) to {bb['mean_delay_ms']:.0f} ms at essentially unchanged query cost "
        f"({bb['avg_queries']:.1f} vs.\\ {a['avg_queries']:.1f}), while the normalized perturbation cost rises slightly "
        f"({bb['avg_cost_l1']:.3f} vs.\\ {a['avg_cost_l1']:.3f}). Regime C is effectively identical to Regime A because temporal "
        f"dilation is already the cheapest per unit score in both.\n",
        '\\subsubsection{Ablation Study 4: Boundary Snapping Tolerance \\texorpdfstring{$\\epsilon$}{epsilon}}\n'
        'Algorithm~\\ref{alg:tsbs} (TSBS) snaps mutable features $j \\in \\mathcal{I}_{\\text{mono}}^+ = \\{2, 4, 6, 11\\}$ to active split '
        'thresholds. While discrete counts $\\{2, 4, 11\\}$ use an integer increment $\\epsilon_j = 1.0$, the continuous duration '
        '($j = 6$) uses a relative offset $\\epsilon_6 = \\text{round}(\\epsilon_{\\text{rel}} \\cdot \\max(1.0, |\\theta_6|))$. '
        'Table~\\ref{tab:ablation_epsilon} evaluates $\\epsilon_{\\text{rel}} \\in \\{0.0, 10^{-6}, 10^{-4}, 10^{-2}, 1.0\\}$.\n',
        table('Ablation 4: Boundary Snapping Relative Numerical Offset $\\epsilon_{\\text{rel}}$', 'tab:ablation_epsilon', 'ccccc',
              '\\textbf{Relative Offset $\\epsilon_{\\text{rel}}$} & \\textbf{ASR (\\%)} & \\textbf{PFR (\\%)} & \\textbf{Avg Queries} & '
              '\\textbf{Avg Cost ($\\bar{D}_{L_1}$)}', eps_rows),
        'On this cohort the outcome is insensitive to $\\epsilon_{\\text{rel}}$: ASR and PFR are '
        f"{pct(e['0.0']['asr'])} and {pct(e['0.0']['pfr'])} for every setting, and cost differs only at the third decimal. "
        'Because the duration snap is only one of several TSBS candidates and the directional step usually crosses the boundary '
        'first, the numerical offset is rarely decisive; we therefore make no claim of floating-point fragility at $\\epsilon_{\\text{rel}} = 0$.\n',
    ]
    return '\n'.join(parts)


def main() -> None:
    with open(RESULTS) as handle:
        res = json.load(handle)
    with open(TEX, encoding='utf-8', newline='') as handle:
        tex = handle.read()
    crlf = '\r\n' in tex
    tex = tex.replace('\r\n', '\n')
    if BEGIN not in tex:
        start = tex.index('In operational deployments, an attacker may face strict query throttling')
        stop = tex.index('\\subsubsection{Ablation Study 5')
        tex = tex[:start] + BEGIN + '\n' + END + '\n\n' + tex[stop:]
    head, rest = tex.split(BEGIN + '\n', 1)
    _, tail = rest.split(END, 1)
    out = head + BEGIN + '\n' + render(res) + END + tail
    with open(TEX, 'w', encoding='utf-8', newline='') as handle:
        handle.write(out.replace('\n', '\r\n') if crlf else out)
    print('Rendered ablations 1-4 into', TEX)


if __name__ == '__main__':
    main()
