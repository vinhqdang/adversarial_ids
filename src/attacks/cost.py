"""Operational cost of a perturbation under the paper's cost model (Section 4.4).

c1 = 1e-4 per second of added delay, c2 = 1e-2 per byte of payload padding, c3 = 1.0 per injected
chaff packet and c4 = 1e-3 per normalised TCP-window displacement. The same function prices every
attack's output, so the numbers are comparable across methods. It prices only the four mutable
controls; changes to immutable features are not priced because such flows are infeasible anyway
(PFR), which is why costs are reported over feasible evasive flows only.
"""

import numpy as np

UNIT_COSTS = {'time': 1e-4, 'pad': 1e-2, 'chaff': 1.0, 'win': 1e-3}
TCP_PROBE_BYTES = 40.0   # empty TCP segment (RFC 9293), L0 in the generators
UDP_PROBE_BYTES = 28.0   # zero-length UDP datagram, L0 in the generators


def operational_cost(x_orig: np.ndarray, x_adv: np.ndarray, costs=UNIT_COSTS) -> float:
    """Cost of turning ``x_orig`` into ``x_adv`` with delay, padding, chaff and window changes."""
    delay_s = max(0.0, (x_adv[6] - x_orig[6]) / 1e6)
    chaff_pkts = max(0.0, x_adv[2] - x_orig[2])
    probe_bytes = TCP_PROBE_BYTES if x_orig[1] == 6 else UDP_PROBE_BYTES
    pad_bytes = max(0.0, (x_adv[4] - x_orig[4]) - probe_bytes * chaff_pkts)
    window = abs(x_adv[7] - x_orig[7]) / 65535.0
    return float(costs['time'] * delay_s + costs['pad'] * pad_bytes
                 + costs['chaff'] * chaff_pkts + costs['win'] * window)
