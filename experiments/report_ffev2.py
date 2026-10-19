"""Builds results/ffev2_report.md tables from results/ffev2_*.json (numbers only; prose is in the report header)."""
import json, os, sys
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
DS = [('cse2018', 'CSE-CIC-IDS2018'), ('cic2017', 'CIC-IDS2017'), ('unsw', 'UNSW-NB15')]
MS = ['xgboost', 'lightgbm', 'catboost']
V2, V2S, V1, CT, CP = 'FFE-v2 (score-only)', 'FFE-v2 + snapping', 'Flow-Field Evasion (FFE)', 'White-box control (project in loop)', 'CAPGD (surrogate transfer)'
res = {d: json.load(open(f'{R}/ffev2_{d}.json')) for d, _ in DS}
rf = lambda a: np.array([o['success'] and o['valid'] for o in a['outcomes']])
out = []
def P(s=''): out.append(s)
P('| Corpus | Target | N | ' + ' | '.join(['FFE-v1', 'Control', 'CAPGD', 'FFE-v2 score-only', 'FFE-v2 + snapping']) + ' |')
P('|---|---|---|---|---|---|---|---|')
for d, dn in DS:
    for m in MS:
        a = res[d]['models'][m]['attacks']
        cell = lambda k: f"{a[k]['real_asr']:.1f}" if k in a else 'n/a'
        P(f"| {dn} | {m} | {a[V2]['n']} | {cell(V1)} | {cell(CT)} | {cell(CP)} | {cell(V2)} | {cell(V2S)} |")
P(); P('Queries (mean per flow) and median operational cost of the successful flows, and median cost on the flows both methods evade (paired):'); P()
P('| Corpus | Target | q v1 | q control | q v2 | q v2+snap | med cost v1 | med cost control | med cost v2 | paired med v2 / v1 (n) | paired med v2 / control (n) |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
def med(a): return a['median_op_cost_real']
def paired(a, b):
    both = rf(a) & rf(b)
    if not both.any(): return 'n/a'
    ca = np.array([o['op_cost'] for o in a['outcomes']])[both]; cb = np.array([o['op_cost'] for o in b['outcomes']])[both]
    return f"{np.median(ca):.3g} / {np.median(cb):.3g} ({int(both.sum())})"
f3 = lambda v: 'n/a' if v is None else f'{v:.3g}'
for d, dn in DS:
    for m in MS:
        a = res[d]['models'][m]['attacks']
        P(f"| {dn} | {m} | {a[V1]['avg_queries']:.1f} | {a[CT]['avg_queries']:.1f} | {a[V2]['avg_queries']:.1f} | {a[V2S]['avg_queries']:.1f} | {f3(med(a[V1]))} | {f3(med(a[CT]))} | {f3(med(a[V2]))} | {paired(a[V2], a[V1])} | {paired(a[V2], a[CT])} |")
P(); P('Exact McNemar (Real ASR, same flows), FFE-v2 score-only versus each method: v2-only / other-only / p.'); P()
P('| Corpus | Target | vs v1 | vs control | vs CAPGD | vs v2+snap |'); P('|---|---|---|---|---|---|')
for d, dn in DS:
    for m in MS:
        mc = res[d]['models'][m]['mcnemar']
        c = lambda k: (lambda r: f"{r['v2_only']}/{r['other_only']}/{r['p']:.2g}")(mc[f'{V2} vs {k}']) if f'{V2} vs {k}' in mc else 'n/a'
        P(f'| {dn} | {m} | {c(V1)} | {c(CT)} | {c(CP)} | {c(V2S)} |')
P(); P('Clean accuracy: ' + '; '.join(f"{dn} " + '/'.join(f"{res[d]['models'][m]['clean']['accuracy']*100:.2f}" for m in MS) for d, dn in DS) + ' (xgboost/lightgbm/catboost).')
P(); P('Generator usage of FFE-v2 successful flows (share of flows using each; score-only):'); P()
P('| Corpus | Target | time | pad | chaff | seg | win |'); P('|---|---|---|---|---|---|---|')
for d, dn in DS:
    for m in MS:
        oc = [o for o in res[d]['models'][m]['attacks'][V2]['outcomes'] if o['success'] and o['valid'] and o['path']]
        sh = lambda g: f"{100*np.mean([g in o['path'] for o in oc]):.0f}%" if oc else 'n/a'
        P(f'| {dn} | {m} | {sh("time")} | {sh("pad")} | {sh("chaff")} | {sh("seg")} | {sh("win")} |')
P(); P('Per-family Real ASR of FFE-v2 (score-only):'); P()
for d, dn in DS:
    for m in MS:
        e = res[d]['models'][m]; fam = np.array(e['cohort']['family']); fl = rf(e['attacks'][V2])
        P(f"* {dn} / {m}: " + ', '.join(f"{f} {100*fl[fam==f].mean():.1f} (n={int((fam==f).sum())})" for f in sorted(set(fam))))
open(f'{R}/ffev2_tables.md', 'w').write('\n'.join(out)); print('\n'.join(out))
