"""
Packet-level validation of the schema-level flow perturbations (synthetic replay).

No raw pcaps are available, so a flow record of the 12-feature schema is expanded into a
synthetic packet trace that is consistent with it, the four FFE actions are applied to the
packets, and the 12 features are recomputed with a CICFlowMeter-style extractor.

What this validates: the semantics of the actions (what each one changes in the features, in
which direction, by how much) and the schema invariants (MTU, header bounds, coupled rates).
What it does not validate: a real network stack. The trace is synthetic (packet sizes and
inter-arrival times are drawn at random subject to the record's aggregates), there is no TCP
state machine, no server reaction, no retransmission, no segmentation offload and no
middlebox; see ``results/packet_replay.md``.

Units follow the repository schema: durations and timestamps in microseconds, sizes in bytes,
``totlen_*`` are payload bytes, ``fwd_header_len`` is the sum of per-packet transport headers.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

# Feature indices of the 12-feature schema
DST_PORT, PROTO, FWD_PKTS, BWD_PKTS, LEN_FWD, LEN_BWD, DURATION, WIN, BYTES_S, PKTS_S, FWD_MEAN, FWD_HDR = range(12)
TCP, UDP = 6, 17
MTU = 1500                    # IP packet size limit
IP_HEADER = 20
SCHEMA_PAYLOAD_CAP = 1500     # per-packet payload bound implied by the schema (payload <= 1500 * packets)
MAX_TCP_HEADER = 60
CHAFF_MODES = ('schema', 'empty_compensated', 'empty_literal')


@dataclass(frozen=True)
class Trace:
    """A packet list sorted by timestamp. Arrays are never mutated in place."""
    ts: np.ndarray        # int64 microseconds
    fwd: np.ndarray       # bool, True for client-to-server packets
    payload: np.ndarray   # int64 payload bytes
    header: np.ndarray    # int64 transport header bytes
    win: np.ndarray       # int64 advertised TCP window (0 for UDP)
    proto: int
    dst_port: int

    def __len__(self) -> int:
        return int(self.ts.size)


# --------------------------------------------------------------------------------------------
# (1) trace generator
# --------------------------------------------------------------------------------------------
def _partition(total: int, n: int, cap: int, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    """Random integer vector of length ``n`` with sum ``total`` and entries in [0, cap].

    If ``total`` cannot fit (total > cap * n) the cap is relaxed to ceil(total / n).
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    total = int(total)
    if total <= 0:
        return np.zeros(n, dtype=np.int64)
    cap = max(int(cap), -(-total // n))
    w = rng.lognormal(0.0, sigma, n)
    a = np.minimum(np.floor(total * w / w.sum()), cap).astype(np.int64)
    rem = total - int(a.sum())
    while rem > 0:
        idx = np.flatnonzero(a < cap)
        share = max(1, rem // idx.size)
        give = np.minimum(cap - a[idx], share)
        # never hand out more than the remainder
        csum = np.cumsum(give)
        give = np.where(csum <= rem, give, np.maximum(0, rem - (csum - give)))
        a[idx] += give
        rem -= int(give.sum())
    return a


def trace_from_record(x: np.ndarray, rng: Optional[np.random.Generator] = None) -> Trace:
    """Build a packet sequence consistent with a 12-feature flow record.

    Forward/backward packet counts, payload totals, forward header sum, duration (first to last
    timestamp), the TCP window of the first forward packet, port and protocol are reproduced;
    individual sizes, directions and inter-arrival times are random.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    proto = int(round(x[PROTO]))
    is_tcp = proto == TCP
    n_f = max(1, int(round(x[FWD_PKTS])))
    n_b = max(0, int(round(x[BWD_PKTS])))
    n = n_f + n_b
    dur = max(0, int(round(x[DURATION])))

    # direction sequence: first packet is forward
    fwd = np.zeros(n, dtype=bool)
    fwd[0] = True
    if n_f > 1:
        fwd[1 + rng.permutation(n - 1)[:n_f - 1]] = True

    # timestamps: integer inter-arrival times summing to the duration
    ts = np.concatenate([[0], np.cumsum(_partition(dur, n - 1, dur, rng, sigma=1.5))]).astype(np.int64)

    # payloads
    payload = np.zeros(n, dtype=np.int64)
    payload[fwd] = _partition(int(round(x[LEN_FWD])), n_f, SCHEMA_PAYLOAD_CAP, rng)
    payload[~fwd] = _partition(int(round(x[LEN_BWD])), n_b, SCHEMA_PAYLOAD_CAP, rng)

    # transport headers
    h_min = 20 if is_tcp else 8
    h_max = MAX_TCP_HEADER if is_tcp else 8
    hdr_total = int(round(x[FWD_HDR]))
    extra = int(np.clip(hdr_total - h_min * n_f, 0, (h_max - h_min) * n_f))
    header = np.full(n, h_min, dtype=np.int64)
    header[fwd] = h_min + _partition(extra, n_f, h_max - h_min, rng, sigma=0.5)

    # windows: first forward packet carries the record's initial window
    win = np.zeros(n, dtype=np.int64)
    if is_tcp:
        w0 = int(round(x[WIN]))
        win[:] = w0
        win[~fwd] = 29200
    return Trace(ts, fwd, payload, header, win, proto, int(round(x[DST_PORT])))


# --------------------------------------------------------------------------------------------
# (2) CICFlowMeter-style feature extractor
# --------------------------------------------------------------------------------------------
def extract_features(t: Trace) -> np.ndarray:
    """Recompute the 12 schema features from a packet list."""
    order = np.argsort(t.ts, kind='stable')
    ts, fwd, payload, header, win = t.ts[order], t.fwd[order], t.payload[order], t.header[order], t.win[order]
    n_f = int(fwd.sum())
    n_b = int(len(ts) - n_f)
    dur = float(ts[-1] - ts[0])                       # last - first timestamp
    len_f = float(payload[fwd].sum())
    len_b = float(payload[~fwd].sum())
    dur_s = max(1e-6, dur / 1e6)
    first_fwd = int(np.argmax(fwd)) if n_f else 0
    init_win = float(win[first_fwd]) if (t.proto == TCP and n_f) else 0.0
    return np.array([
        t.dst_port, t.proto, n_f, n_b, len_f, len_b, dur, init_win,
        (len_f + len_b) / dur_s, (n_f + n_b) / dur_s, len_f / max(1, n_f), float(header[fwd].sum()),
    ], dtype=float)


# --------------------------------------------------------------------------------------------
# (3) packet-level FFE actions
# --------------------------------------------------------------------------------------------
def _sorted(t: Trace, **arrays) -> Trace:
    order = np.argsort(arrays.get('ts', t.ts), kind='stable')
    fields = {k: arrays.get(k, getattr(t, k))[order] for k in ('ts', 'fwd', 'payload', 'header', 'win')}
    return replace(t, **fields)


def temporal_dilation(t: Trace, delay_us: int) -> Trace:
    """Stretch every inter-arrival time by the same factor so that the duration grows by ``delay_us``."""
    delay_us = int(delay_us)
    if delay_us <= 0:
        return t
    rel = t.ts - t.ts[0]
    dur = int(rel[-1])
    if dur > 0:
        new_rel = np.rint(rel * ((dur + delay_us) / dur)).astype(np.int64)
        new_rel[-1] = dur + delay_us
    else:                                             # all packets simultaneous: delay the last one
        new_rel = rel.copy()
        new_rel[-1] += delay_us
    return replace(t, ts=t.ts[0] + new_rel)


def payload_headroom(t: Trace) -> np.ndarray:
    """Bytes each packet can still carry: MTU less IP header, transport header and current payload."""
    return np.maximum(0, MTU - IP_HEADER - t.header - t.payload)


def payload_padding(t: Trace, extra_bytes: int) -> Tuple[Trace, int]:
    """Add ``extra_bytes`` of payload to forward packets, never exceeding the MTU.

    Bytes are spread evenly over the forward packets that still have headroom. Returns the new
    trace and the number of bytes that did not fit.
    """
    remaining = int(extra_bytes)
    if remaining <= 0:
        return t, 0
    payload = t.payload.copy()
    head = np.where(t.fwd, payload_headroom(t), 0)
    while remaining > 0:
        idx = np.flatnonzero(head > 0)
        if idx.size == 0:
            break
        share = max(1, remaining // idx.size)
        give = np.minimum(head[idx], share)
        csum = np.cumsum(give)
        give = np.where(csum <= remaining, give, np.maximum(0, remaining - (csum - give)))
        payload[idx] += give
        head[idx] -= give
        remaining -= int(give.sum())
    return replace(t, payload=payload), remaining


def chaff_injection(t: Trace, n_chaff: int, header_bytes: int, payload_bytes: int,
                    rng: Optional[np.random.Generator] = None) -> Trace:
    """Insert ``n_chaff`` forward packets at random instants strictly inside the flow's time span.

    The duration is unchanged and the first forward packet stays the first forward packet.
    """
    n_chaff = int(n_chaff)
    if n_chaff <= 0:
        return t
    rng = rng if rng is not None else np.random.default_rng(0)
    t0, t1 = int(t.ts[0]), int(t.ts[-1])
    new_ts = rng.integers(t0 + 1, max(t0 + 1, t1) + 1, size=n_chaff) if t1 > t0 else np.full(n_chaff, t0)
    # a chaff packet at t0 would not displace the first forward packet because sorting is stable
    # and the originals come first; with t1 == t0 all packets tie and the originals keep priority.
    win0 = int(t.win[int(np.argmax(t.fwd))]) if t.proto == TCP else 0
    return _sorted(
        t,
        ts=np.concatenate([t.ts, new_ts]),
        fwd=np.concatenate([t.fwd, np.ones(n_chaff, dtype=bool)]),
        payload=np.concatenate([t.payload, np.full(n_chaff, int(payload_bytes), dtype=np.int64)]),
        header=np.concatenate([t.header, np.full(n_chaff, int(header_bytes), dtype=np.int64)]),
        win=np.concatenate([t.win, np.full(n_chaff, win0, dtype=np.int64)]),
    )


def window_morph(t: Trace, new_window: int) -> Trace:
    """Change the advertised window of the first forward packet (TCP only)."""
    if t.proto != TCP:
        return t
    win = t.win.copy()
    win[int(np.argmax(t.fwd))] = int(np.clip(round(new_window), 0, 65535))
    return replace(t, win=win)


def header_option_padding(t: Trace, extra_bytes: int) -> Tuple[Trace, int]:
    """Lengthen forward transport headers with TCP options (up to 60 B). Returns (trace, unmet bytes)."""
    remaining = int(extra_bytes)
    if remaining <= 0 or t.proto != TCP:
        return t, max(0, remaining)
    header = t.header.copy()
    head = np.where(t.fwd, MAX_TCP_HEADER - header, 0)
    while remaining > 0:
        idx = np.flatnonzero(head > 0)
        if idx.size == 0:
            break
        share = max(1, remaining // idx.size)
        give = np.minimum(head[idx], share)
        csum = np.cumsum(give)
        give = np.where(csum <= remaining, give, np.maximum(0, remaining - (csum - give)))
        header[idx] += give
        head[idx] -= give
        remaining -= int(give.sum())
    return replace(t, header=header), remaining


# --------------------------------------------------------------------------------------------
# action magnitudes from (x_orig, x_adv), as the generators of flow_field_evasion.py define them
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionPlan:
    delay_us: int
    n_chaff: int
    chaff_header: int
    chaff_payload: int
    pad_bytes: int
    header_extra: int      # header growth not explained by the chaff packets (TSBS header-length snap)
    header_deficit: int    # header bytes the chaff packets add beyond the schema's header growth (not realisable)
    new_window: Optional[int]


def chaff_geometry(x_orig: np.ndarray) -> Tuple[int, int]:
    """(h_0, l_0) of one chaff packet, as in ``PhysicalFlowGenerators.v_chaff``."""
    if int(x_orig[PROTO]) == TCP:
        h0 = 32 if (float(x_orig[FWD_HDR]) / max(1.0, float(x_orig[FWD_PKTS])) >= 26.0) else 20
    else:
        h0 = 8
    return h0, IP_HEADER + h0


def derive_actions(x_orig: np.ndarray, x_adv: np.ndarray, chaff_mode: str = 'schema') -> ActionPlan:
    """Translate the feature displacement into action magnitudes.

    delay = d(duration); chaff count = d(fwd packets); window = x_adv[7]. The generators
    account every chaff packet as ``l_0`` bytes of ``totlen_fwd`` (40/52 B for TCP, 28 B UDP):
      schema             chaff carries min(l_0, d(totlen)/N) payload bytes, the rest is padding
      empty_compensated  chaff carries no payload, all of d(totlen) is padding (features match)
      empty_literal      chaff carries no payload, padding = d(totlen) - N*l_0 (generator accounting,
                         i.e. the l_0 bytes per chaff packet are *not* delivered as payload)
    """
    if chaff_mode not in CHAFF_MODES:
        raise ValueError(f'chaff_mode must be one of {CHAFF_MODES}')
    h0, l0 = chaff_geometry(x_orig)
    n = max(0, int(round(x_adv[FWD_PKTS] - x_orig[FWD_PKTS])))
    d_len = max(0, int(round(x_adv[LEN_FWD] - x_orig[LEN_FWD])))
    if chaff_mode == 'schema':
        cp = min(l0, d_len // n) if n else 0
        pad = d_len - cp * n
    elif chaff_mode == 'empty_compensated':
        cp, pad = 0, d_len
    else:
        cp, pad = 0, max(0, d_len - n * l0)
    d_hdr = int(round(x_adv[FWD_HDR] - x_orig[FWD_HDR]))
    win = None
    if int(x_orig[PROTO]) == TCP and abs(x_adv[WIN] - x_orig[WIN]) > 0:
        win = int(round(x_adv[WIN]))
    return ActionPlan(
        delay_us=max(0, int(round(x_adv[DURATION] - x_orig[DURATION]))),
        n_chaff=n, chaff_header=h0, chaff_payload=cp, pad_bytes=pad,
        header_extra=max(0, d_hdr - n * h0), header_deficit=max(0, n * h0 - d_hdr), new_window=win)


def apply_actions(t: Trace, plan: ActionPlan, rng: Optional[np.random.Generator] = None) -> Tuple[Trace, Dict[str, int]]:
    """Apply chaff, padding, header options, dilation and window morphing; return (trace, diagnostics)."""
    t1 = chaff_injection(t, plan.n_chaff, plan.chaff_header, plan.chaff_payload, rng)
    t2, pad_unmet = payload_padding(t1, plan.pad_bytes)
    t3, hdr_unmet = header_option_padding(t2, plan.header_extra)
    t4 = temporal_dilation(t3, plan.delay_us)
    if plan.new_window is not None:
        t4 = window_morph(t4, plan.new_window)
    return t4, {'pad_unmet_bytes': pad_unmet, 'header_unmet_bytes': hdr_unmet}


def replay(x_orig: np.ndarray, x_adv: np.ndarray, chaff_mode: str = 'schema', seed: int = 0) -> Dict:
    """Expand ``x_orig`` into packets, apply the actions that turn it into ``x_adv``, re-extract."""
    rng = np.random.default_rng(seed)
    base = trace_from_record(x_orig, rng)
    plan = derive_actions(x_orig, x_adv, chaff_mode)
    tr, diag = apply_actions(base, plan, rng)
    return {'features': extract_features(tr), 'baseline_features': extract_features(base),
            'plan': plan, **diag}


def relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| / max(|a|, 1) elementwise (absolute error for features smaller than 1)."""
    return np.abs(a - b) / np.maximum(np.abs(a), 1.0)
