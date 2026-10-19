"""
Flow-Field Evasion v2 (FFE-v2): score-only multi-scale steering with an optional threshold-snapping stage.

FFE-v1 (``flow_field_evasion.py``, kept unchanged so that its published numbers stay reproducible) takes
one fixed-size probe per generator, a fixed TCP-window target of 64240 B, and uses four generators.
FFE-v2 keeps the idea (steer along physically meaningful packet operations, every candidate projected
onto the flow manifold with the closed-form operator ``project_network_invariants``) and changes the
parts that diagnostics showed to be limiting (``results/ffev2_report.md``):

1. Six admissible generators. Besides temporal dilation ``time``, payload padding ``pad``, chaff
   injection ``chaff`` and window morphing ``win`` there is ``seg`` (MSS clamp / re-segmentation:
   the SAME forward payload is carried by K more, smaller packets, so ``tot_fwd_pkts`` and the header
   total grow while ``totlen_fwd`` and the duration are fixed) and, optionally, ``hdr`` (TCP option
   padding: forward header bytes grow without any other change).
2. Multi-scale line search: every generator is probed on a geometric ladder of magnitudes, not at one
   fixed step, so plateaus of the piecewise-constant score are crossed in one iteration.
3. Data-driven window targets: the window generator probes in-support values (the benign-frequent
   advertised windows of the attacker's own benign traffic sample), in both directions, instead of
   the single hard-coded 64240 B.
4. Progress is measured on the log-odds of the returned score (a monotone transform of the same score
   the target exposes), which does not saturate as float32 probabilities do near 1.
5. Selection: a candidate that crosses the decision boundary ends the search, choosing the cheapest
   crossing one under the paper's operational cost model; otherwise the move with the best
   score-decrease per unit of operational cost, with a relax-cost fallback (after ``relax_after``
   iterations, or when no candidate has a cost-efficient decrease, pure score minimisation).
   Optionally a beam of ``beam`` states is kept (short look-ahead).
6. Optional Tier-1 snapping (``snap=True``): when greedy steering stalls, threshold crossings of ALL
   mutable features (the attacker reads the split thresholds, not the scores of the trees) are turned
   into closed-form generator magnitudes, each candidate projected, scored by the target and the best
   or cheapest crossing taken. Every candidate costs one query.

Black-box accounting: ``queries_used`` counts every candidate flow scored by the target, in or out of
a batch. Only the optional snapping stage reads the model's split thresholds.
"""

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .base import AttackResult, BaseNIDSAttack
from .cost import operational_cost
from .flow_field_evasion import extract_active_splits
from .whitebox_tree import project_batch_with_origin
from ..datasets.schema import NetworkFeatureSchema, project_network_invariants, verify_physical_feasibility

# feature indices
PORT, PROTO, FPKTS, BPKTS, LENF, LENB, DUR, WIN, BYTES_S, PKTS_S, MEAN, HDR = range(12)
GENERATORS_V2: FrozenSet[str] = frozenset({'time', 'pad', 'chaff', 'seg', 'win'})
ALL_GENERATORS_V2: FrozenSet[str] = GENERATORS_V2 | {'hdr'}
MAX_DURATION_US = 1.1e8
MAX_FWD_PKTS = 100000.0
LOGIT_CLIP = 1e-7
COST_FLOOR = 1e-4          # added to marginal cost so that free moves do not get infinite efficiency


@dataclass(frozen=True)
class FFEv2Config:
    """Hyper-parameters (all tuned on the validation split, see experiments/tune_ffev2.py)."""
    generators: FrozenSet[str] = GENERATORS_V2
    ladder_ratio: float = 6.0          # geometric ratio between consecutive probe magnitudes
    ladder_levels: int = 64            # upper bound on probes per generator and iteration
    time_min_us: float = 1000.0        # smallest probed delay (1 ms); the ladder spans up to the 110 s cap
    pad_min: float = 16.0              # smallest probed padding (bytes); spans up to the MTU headroom
    pkt_max: int = 4096                # largest probed chaff / re-segmentation packet count
    window_values: Tuple[int, ...] = ()  # in-support window targets (empty: no window move)
    beam: int = 1
    max_iter: int = 8
    cost_lambda: float = 1.0           # exponent on the marginal cost in the efficiency ratio
    relax_after: int = 3               # iterations after which selection ignores cost
    snap: bool = False
    snap_per_feature: int = 8          # thresholds tried per (feature, direction)
    min_gain: float = 1e-3             # minimal log-odds decrease for a move to count as progress
    refine_steps: int = 0              # bisection probes that shrink the magnitude of the crossing move


@dataclass(frozen=True)
class Ledger:
    """Integer packet-level actions that realise a flow (consumed by the packet-level replay)."""
    delay_us: int = 0
    n_chaff: int = 0
    n_segment: int = 0
    pad_bytes: int = 0
    header_option_bytes: int = 0
    window: Optional[int] = None

    def add(self, gen: str, amount: int) -> 'Ledger':
        if gen == 'time':
            return replace(self, delay_us=self.delay_us + amount)
        if gen == 'pad':
            return replace(self, pad_bytes=self.pad_bytes + amount)
        if gen == 'chaff':
            return replace(self, n_chaff=self.n_chaff + amount)
        if gen == 'seg':
            return replace(self, n_segment=self.n_segment + amount)
        if gen == 'hdr':
            return replace(self, header_option_bytes=self.header_option_bytes + amount)
        raise ValueError(gen)


@dataclass(frozen=True)
class _State:
    x: np.ndarray
    z: float                 # log-odds of the malicious score
    ledger: Ledger
    path: Tuple[str, ...] = ()
    parent: Optional['_State'] = None      # state before the last move (for magnitude refinement)
    move: Optional[Tuple[str, float]] = None   # (generator, requested magnitude) of the last move


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def chaff_header(x_orig: np.ndarray) -> Tuple[float, float]:
    """(h0, l0) of one chaff / re-segmented packet, fixed by the ORIGINAL flow (RFC 7323 option check)."""
    if int(x_orig[PROTO]) == 6:
        h0 = 32.0 if (float(x_orig[HDR]) / max(1.0, float(x_orig[FPKTS])) >= 26.0) else 20.0
    else:
        h0 = 8.0
    return h0, 20.0 + h0


def benign_window_candidates(X: np.ndarray, y: np.ndarray, k: int = 6) -> Tuple[int, ...]:
    """The k most frequent advertised TCP windows of benign training flows (the attacker's benign sample)."""
    tcp = (X[:, PROTO] == 6) & (y == 0)
    if not tcp.any() or k <= 0:
        return ()
    vals, counts = np.unique(X[tcp, WIN].astype(int), return_counts=True)
    order = np.argsort(-counts, kind='stable')[:k]
    return tuple(int(v) for v in vals[order])


# --------------------------------------------------------------------------------------------
# generators: x, magnitude -> raw candidate (projected afterwards)
# --------------------------------------------------------------------------------------------
def apply_generator(x: np.ndarray, gen: str, amount: float, x_orig: np.ndarray) -> np.ndarray:
    """Raw (unprojected) result of one generator with an integer-valued ``amount``."""
    c = x.copy()
    h0, l0 = chaff_header(x_orig)
    if gen == 'time':
        c[DUR] = x[DUR] + amount
    elif gen == 'pad':
        c[LENF] = x[LENF] + amount
    elif gen == 'chaff':
        c[FPKTS] = x[FPKTS] + amount
        c[LENF] = x[LENF] + amount * l0
        c[HDR] = x[HDR] + amount * h0
    elif gen == 'seg':
        c[FPKTS] = x[FPKTS] + amount
        c[HDR] = x[HDR] + amount * h0
    elif gen == 'hdr':
        c[HDR] = x[HDR] + amount
    elif gen == 'win':
        c[WIN] = amount
    else:
        raise ValueError(gen)
    return c


def _ladder(first: float, ratio: float, levels: int, cap: float, integer: bool = True) -> List[float]:
    """Geometric magnitudes first * ratio^k clipped at ``cap`` (the clip itself is kept once)."""
    if cap < 1:
        return []
    vals = []
    for k in range(levels):
        v = min(first * ratio ** k, cap)
        v = float(np.ceil(v)) if integer else v
        if not vals or v > vals[-1]:
            vals.append(v)
        if v >= cap:
            break
    return vals


def move_ladder(x: np.ndarray, x_orig: np.ndarray, cfg: FFEv2Config) -> List[Tuple[str, float]]:
    """All (generator, magnitude) probes of one iteration for flow state ``x``."""
    r, n = cfg.ladder_ratio, cfg.ladder_levels
    moves: List[Tuple[str, float]] = []
    gens = cfg.generators
    fp, ln, dur, hdr = float(x[FPKTS]), float(x[LENF]), float(x[DUR]), float(x[HDR])
    if 'time' in gens:
        moves += [('time', v) for v in _ladder(cfg.time_min_us, r, n, MAX_DURATION_US - dur)]
    if 'pad' in gens:
        moves += [('pad', v) for v in _ladder(cfg.pad_min, r, n, 1500.0 * fp - ln)]
    if 'chaff' in gens:
        moves += [('chaff', v) for v in _ladder(1.0, r, n, min(MAX_FWD_PKTS - fp, cfg.pkt_max))]
    if 'seg' in gens:
        moves += [('seg', v) for v in _ladder(1.0, r, n, min(MAX_FWD_PKTS - fp, cfg.pkt_max, np.floor(ln) - fp))]
    if 'hdr' in gens and int(x[PROTO]) == 6:
        moves += [('hdr', v) for v in _ladder(4.0, r, n, 60.0 * fp - hdr)]
    if 'win' in gens and int(x[PROTO]) == 6:
        moves += [('win', float(w)) for w in cfg.window_values if float(w) != float(x[WIN])]
    return moves


class FFEv2Attack(BaseNIDSAttack):
    """Score-only multi-scale Lie-cone steering (FFE-v2); see the module docstring."""

    def __init__(self, model, schema: NetworkFeatureSchema, config: Optional[FFEv2Config] = None,
                 feature_stds: Optional[np.ndarray] = None):
        super().__init__(model, schema)
        self.cfg = config or FFEv2Config()
        self.feature_stds = feature_stds if feature_stds is not None else np.ones(len(schema.feature_names))
        self.last_ledger = Ledger()
        self.last_path: Tuple[str, ...] = ()

    # ---- target access --------------------------------------------------------------------
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Malicious probabilities of a batch; every row is one query."""
        X = np.atleast_2d(X)
        self.query_count += len(X)
        probs = self.model.predict_proba(X)
        return np.asarray(probs[:, 1] if probs.ndim == 2 else probs, dtype=float)

    # ---- candidate generation -------------------------------------------------------------
    def _expand(self, st: _State, x_orig: np.ndarray, moves: Sequence[Tuple[str, float]]):
        if not moves:
            return []
        raw = np.stack([apply_generator(st.x, g, a, x_orig) for g, a in moves])
        proj = project_batch_with_origin(raw, x_orig, self.schema)
        seen = {st.x.tobytes()}
        out = []
        for (g, a), row in zip(moves, proj):
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((g, row, a))
        return out

    def _effective_amount(self, gen: str, before: np.ndarray, after: np.ndarray) -> int:
        if gen == 'time':
            return int(round(after[DUR] - before[DUR]))
        if gen == 'pad':
            return int(round(after[LENF] - before[LENF]))
        if gen in ('chaff', 'seg'):
            return int(round(after[FPKTS] - before[FPKTS]))
        if gen == 'hdr':
            return int(round(after[HDR] - before[HDR]))
        return 0

    def _advance(self, st: _State, gen: str, row: np.ndarray, p: float, amount: float = 0.0) -> _State:
        if gen == 'win':
            ledger = replace(st.ledger, window=int(row[WIN]))
        else:
            ledger = st.ledger.add(gen, self._effective_amount(gen, st.x, row))
        return _State(row, float(logit(p)), ledger, st.path + (gen,), st, (gen, amount))

    def _refine(self, st: _State, x_orig: np.ndarray) -> _State:
        """Shrink the magnitude of the last (crossing) move by bisection; every probe is one query.

        The result is always a flow that was itself scored as benign, so success is never assumed.
        """
        if st.parent is None or st.move is None or st.move[0] == 'win':
            return st
        gen, hi = st.move
        lo = max(0.0, np.floor(hi / self.cfg.ladder_ratio))
        best = st
        for _ in range(self.cfg.refine_steps):
            mid = float(np.round(np.sqrt(max(lo, 1.0) * hi)) if lo >= 1 else np.round(hi / 2))
            if mid <= lo or mid >= hi:
                break
            cand = self._expand(st.parent, x_orig, [(gen, mid)])
            if not cand:
                lo = mid
                continue
            p = float(self._score(cand[0][1])[0])
            if p < 0.5:
                best, hi = self._advance(st.parent, gen, cand[0][1], p, mid), mid
            else:
                lo = mid
        return best

    # ---- snapping (Tier 1) -----------------------------------------------------------------
    def _snap_moves(self, st: _State, x_orig: np.ndarray) -> List[Tuple[str, float]]:
        """Generator magnitudes that push one mutable feature across one split threshold."""
        splits = extract_active_splits(self.model, st.x, self.schema)
        if not splits:
            return []
        x, cfg, gens = st.x, self.cfg, self.cfg.generators
        h0, l0 = chaff_header(x_orig)
        fp, bp, ln, lb, dur = x[FPKTS], x[BPKTS], x[LENF], x[LENB], x[DUR]
        tb, tp = ln + lb, fp + bp
        by_feat: Dict[int, np.ndarray] = {}
        for j, th in splits:
            by_feat.setdefault(int(j), []).append(float(th))
        moves: List[Tuple[str, float]] = []

        def pick(ths: List[float], x_j: float) -> List[float]:
            arr = np.array(sorted(set(ths), key=lambda t: abs(t - x_j)))
            if len(arr) <= cfg.snap_per_feature:
                return list(arr)
            ix = np.unique(np.round(np.linspace(0, len(arr) - 1, cfg.snap_per_feature)).astype(int))
            return list(arr[ix])

        for j, ths in by_feat.items():
            if j == FPKTS:
                for th in pick([t for t in ths if t >= fp], fp):
                    k = np.floor(th) + 1 - fp
                    if 'seg' in gens and 1 <= k <= np.floor(ln) - fp:
                        moves.append(('seg', k))
                    if 'chaff' in gens and k >= 1:
                        moves.append(('chaff', k))
            elif j == LENF and 'pad' in gens:
                moves += [('pad', np.floor(t) + 1 - ln) for t in pick([t for t in ths if t >= ln], ln)]
            elif j == DUR and 'time' in gens:
                moves += [('time', np.floor(t) + 1 - dur) for t in pick([t for t in ths if t >= dur], dur)]
            elif j == WIN and 'win' in gens and int(x[PROTO]) == 6:
                for th in pick(ths, x[WIN]):
                    moves += [('win', np.floor(th) + 1), ('win', np.ceil(th) - 1)]
            elif j == BYTES_S:
                rate = x[BYTES_S]
                if 'time' in gens:
                    for th in pick([t for t in ths if 0 < t < rate], rate):
                        moves.append(('time', np.ceil(tb * 1e6 / th * (1 + 1e-6)) + 1 - dur))
                if 'pad' in gens:
                    for th in pick([t for t in ths if t > rate], rate):
                        moves.append(('pad', np.ceil(th * dur / 1e6 * (1 + 1e-6)) + 1 - tb))
            elif j == PKTS_S:
                rate = x[PKTS_S]
                if 'time' in gens:
                    for th in pick([t for t in ths if 0 < t < rate], rate):
                        moves.append(('time', np.ceil(tp * 1e6 / th * (1 + 1e-6)) + 1 - dur))
                for th in pick([t for t in ths if t > rate], rate):
                    k = np.ceil(th * dur / 1e6 * (1 + 1e-6)) + 1 - tp
                    if 'seg' in gens and 1 <= k <= np.floor(ln) - fp:
                        moves.append(('seg', k))
                    if 'chaff' in gens and k >= 1:
                        moves.append(('chaff', k))
            elif j == MEAN:
                mean = x[MEAN]
                for th in pick([t for t in ths if 0 < t < mean], mean):
                    if 'seg' in gens:
                        k = np.floor(ln / th - fp) + 1
                        if 1 <= k <= np.floor(ln) - fp:
                            moves.append(('seg', k))
                    if 'chaff' in gens and th > l0:
                        k = np.floor((ln - th * fp) / (th - l0)) + 1
                        if k >= 1:
                            moves.append(('chaff', k))
                if 'pad' in gens:
                    moves += [('pad', np.ceil(t * fp - ln)) for t in pick([t for t in ths if t > mean], mean)]
            elif j == HDR and 'hdr' in gens and int(x[PROTO]) == 6:
                moves += [('hdr', np.floor(t) + 1 - x[HDR]) for t in pick([t for t in ths if t >= x[HDR]], x[HDR])]
        return [(g, float(a)) for g, a in moves if a >= (0 if g == 'win' else 1)]

    # ---- main loop -----------------------------------------------------------------------
    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        cfg = self.cfg
        x_orig = x.copy()
        x0 = project_network_invariants(x_orig, self.schema, x_orig)
        p0 = float(self._score(x0)[0])
        start = _State(x0, float(logit(p0)), Ledger())
        final = start
        if p0 >= 0.5:
            final = self._search(start, x_orig)
        self.last_ledger, self.last_path = final.ledger, final.path
        return self._result(x_orig, final.x, self._score_final(final))

    def _score_final(self, st: _State) -> bool:
        return st.z < 0.0

    def _crossing(self, cands, probs, x_orig):
        """Cheapest candidate with score < 0.5, if any."""
        idx = [i for i, p in enumerate(probs) if p < 0.5]
        if not idx:
            return None
        return min(idx, key=lambda i: operational_cost(x_orig, cands[i][1]))

    def _search(self, start: _State, x_orig: np.ndarray) -> _State:
        cfg = self.cfg
        beam = [start]
        best = start
        for it in range(cfg.max_iter):
            pool = []                                        # (parent state, gen, row, prob)
            for st in beam:
                cands = self._expand(st, x_orig, move_ladder(st.x, x_orig, cfg))
                if cands:
                    probs = self._score(np.stack([c[1] for c in cands]))
                    pool += [(st, g, r, float(p), a) for (g, r, a), p in zip(cands, probs)]
            progressed = self._select(pool, x_orig, it)
            if progressed is None and cfg.snap:
                pool = []
                for st in beam:
                    cands = self._expand(st, x_orig, self._snap_moves(st, x_orig))
                    if cands:
                        probs = self._score(np.stack([c[1] for c in cands]))
                        pool += [(st, g, r, float(p), a) for (g, r, a), p in zip(cands, probs)]
                progressed = self._select(pool, x_orig, it, relax=True)
            if progressed is None:
                break
            beam = progressed
            best = min(beam, key=lambda s: s.z)
            if best.z < 0.0:
                return self._refine(best, x_orig) if cfg.refine_steps else best
        return best

    def _select(self, pool, x_orig, it: int, relax: bool = False) -> Optional[List[_State]]:
        """Next beam from the scored pool, or None when nothing decreases the score."""
        if not pool:
            return None
        cfg = self.cfg
        cross = self._crossing([(g, r) for _, g, r, _, _ in pool], [p for _, _, _, p, _ in pool], x_orig)
        if cross is not None:
            st, g, r, p, a = pool[cross]
            return [self._advance(st, g, r, p, a)]
        use_cost = (not relax) and it < cfg.relax_after
        scored = []
        for st, g, r, p, a in pool:
            gain = st.z - float(logit(p))
            if gain < cfg.min_gain:
                continue
            if use_cost:
                dcost = max(0.0, operational_cost(x_orig, r) - operational_cost(x_orig, st.x))
                key = gain / (dcost + COST_FLOOR) ** cfg.cost_lambda
            else:
                key = gain
            scored.append((key, st, g, r, p, a))
        if not scored and use_cost:                           # relax-cost fallback: any decrease at all
            for st, g, r, p, a in pool:
                gain = st.z - float(logit(p))
                if gain >= cfg.min_gain:
                    scored.append((gain, st, g, r, p, a))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        out, seen = [], set()
        for _, st, g, r, p, a in scored:
            k = r.tobytes()
            if k in seen:
                continue
            seen.add(k)
            out.append(self._advance(st, g, r, p, a))
            if len(out) >= cfg.beam:
                break
        return out

    def _result(self, x_orig: np.ndarray, x_adv: np.ndarray, success: bool) -> AttackResult:
        d_l1 = float(np.mean(np.abs(x_adv - x_orig) / (self.feature_stds + 1e-9)))
        return AttackResult(
            x_original=x_orig, x_adversarial=x_adv, success=bool(success),
            physically_valid=verify_physical_feasibility(x_orig, x_adv, self.schema),
            queries_used=self.query_count, weighted_cost=d_l1, feature_shifts=x_adv - x_orig)
