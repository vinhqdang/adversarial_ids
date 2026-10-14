"""
Flow-Field Evasion (FFE): Physical Lie Vector Field Steering for Tree Ensembles in NIDS.

A novel, non-incremental algorithm that operates exclusively on the Infinitesimal Generators
of admissible network packet transformations (Temporal Dilation, Payload Padding, Chaff Injection,
TCP Window Morphing) and Tree-Split Boundary Snapping (TSBS).

By integrating along the Lie Cone of physical vector fields, every generated perturbation is
mathematically guaranteed to be 100% physically realizable on the network flow manifold M_flow.
"""

from typing import List, Tuple, Dict, Optional, Callable, FrozenSet
import numpy as np
from .base import BaseNIDSAttack, AttackResult
from ..datasets.schema import NetworkFeatureSchema, project_network_invariants, verify_physical_feasibility


ALL_GENERATORS: FrozenSet[str] = frozenset({'time', 'pad', 'chaff', 'win'})

# Generator that realizes a change of each monotonic feature index (used to honour ablations in TSBS)
_MONOTONIC_FEATURE_GENERATOR = {2: 'chaff', 4: 'pad', 6: 'time', 11: 'chaff'}


class PhysicalFlowGenerators:
    """
    Defines the infinitesimal tangent vector fields v_k(x) on the network flow manifold.
    Each vector field represents the exact differential impact of an elementary packet-level action.
    """
    @staticmethod
    def v_time(x: np.ndarray) -> np.ndarray:
        """
        Temporal Dilation Generator (v_1):
        Stretching flow duration (adding packet delays).
        Reduces Flow Bytes/s and Flow Packets/s in exact closed-form.
        """
        v = np.zeros_like(x, dtype=float)
        duration = max(1.0, float(x[6]))
        bytes_rate = float(x[8])
        pkts_rate = float(x[9])
        
        v[6] = 1.0                     # dt
        v[8] = -bytes_rate / duration  # d(Rate_bytes)/dt
        v[9] = -pkts_rate / duration   # d(Rate_pkts)/dt
        return v

    @staticmethod
    def v_pad(x: np.ndarray) -> np.ndarray:
        """
        Payload Padding Generator (v_2):
        Injecting dummy payload bytes into forward packets.
        Increases forward bytes, mean packet size, and byte rate.
        """
        v = np.zeros_like(x, dtype=float)
        fwd_pkts = max(1.0, float(x[2]))
        duration_sec = max(1e-6, float(x[6]) / 1e6)
        
        v[4] = 1.0                     # +1 byte
        v[8] = 1.0 / duration_sec      # d(Rate_bytes)/dB
        v[10] = 1.0 / fwd_pkts         # d(Mean_len)/dB
        return v

    @staticmethod
    def v_chaff(x: np.ndarray) -> np.ndarray:
        """
        Chaff Packet Injection Generator (v_3):
        Injecting dummy forward packets with IPv4 (20B) + transport headers (20B/32B TCP, 8B UDP).
        Faithfully preserves negotiated RFC 7323 TCP options when present (H_0=32B, L_0=52B).
        """
        is_tcp = (int(x[1]) == 6)
        if is_tcp:
            # RFC 7323 timestamp options check
            h_0 = 32.0 if (float(x[11]) / max(1.0, float(x[2])) >= 26.0) else 20.0
        else:
            h_0 = 8.0
        l_0 = 20.0 + h_0  # 40B/52B for TCP, 28B for UDP
        
        v = np.zeros_like(x, dtype=float)
        fwd_pkts = max(1.0, float(x[2]))
        fwd_mean = float(x[10])
        
        v[2] = 1.0                     # +1 packet
        v[4] = l_0                     # +L_0 total packet bytes
        v[8] = (l_0 * 1e6) / max(1.0, float(x[6])) # +bytes rate
        v[9] = 1e6 / max(1.0, float(x[6]))         # +packets rate
        v[10] = (l_0 - fwd_mean) / fwd_pkts        # d(mean_len)/dN
        v[11] = h_0                    # +H_0 transport header length
        return v

    @staticmethod
    def v_win(x: np.ndarray, target_win: float = 64240.0) -> np.ndarray:
        """
        Window Advertisement Generator (v_4):
        Morphing TCP window advertisement towards benign OS profile (TCP flows only).
        Directional unit vector driving x_7 directly toward target_win without undershooting.
        """
        v = np.zeros_like(x, dtype=float)
        if int(x[1]) == 6:
            diff = target_win - float(x[7])
            if abs(diff) > 0:
                v[7] = np.sign(diff)
        return v


def extract_active_splits(model, x: np.ndarray, schema: NetworkFeatureSchema) -> List[Tuple[int, float]]:
    """
    Extracts split thresholds from tree ensembles along or near the sample evaluation trajectory.
    Supports XGBoost boosters, LightGBM models, and NIDSEnsembleTrainer wrappers.
    Results are cached on the model to avoid redundant tree parsing.
    """
    if hasattr(model, '_cached_splits'):
        return model._cached_splits
        
    splits = []
    
    # Check if model has custom get_split_thresholds
    if hasattr(model, 'get_split_thresholds'):
        try:
            splits = model.get_split_thresholds(schema.feature_names)
        except Exception:
            pass

    # Direct XGBoost booster
    underlying = getattr(model, 'model', model)
    if not splits and hasattr(underlying, 'get_booster'):
        try:
            booster = underlying.get_booster()
            dump = booster.get_dump(dump_format='text')
            for tree_text in dump:
                for line in tree_text.splitlines():
                    line = line.strip()
                    if '<' in line and '[' in line and ']' in line:
                        bracket_content = line[line.find('[') + 1:line.find(']')]
                        parts = bracket_content.split('<')
                        if len(parts) == 2:
                            feat_str = parts[0].strip()
                            try:
                                if feat_str.startswith('f') and feat_str[1:].isdigit():
                                    f_idx = int(feat_str[1:])
                                elif feat_str in schema.feature_names:
                                    f_idx = schema.feature_names.index(feat_str)
                                else:
                                    continue
                                thresh = float(parts[1].strip())
                                splits.append((f_idx, thresh))
                            except ValueError:
                                pass
        except Exception:
            pass

    # Direct LightGBM booster
    if not splits and hasattr(underlying, 'booster_'):
        try:
            dump = underlying.booster_.dump_model()
            def recurse_tree(node):
                if 'split_feature' in node and 'threshold' in node:
                    feat = node['split_feature']
                    thresh = float(node['threshold'])
                    if isinstance(feat, int):
                        splits.append((feat, thresh))
                    elif isinstance(feat, str):
                        if feat.startswith('Column_') and feat[7:].isdigit():
                            splits.append((int(feat[7:]), thresh))
                        elif feat in schema.feature_names:
                            splits.append((schema.feature_names.index(feat), thresh))
                    if 'left_child' in node:
                        recurse_tree(node['left_child'])
                    if 'right_child' in node:
                        recurse_tree(node['right_child'])
            for t_info in dump.get('tree_info', []):
                if 'tree_structure' in t_info:
                    recurse_tree(t_info['tree_structure'])
        except Exception:
            pass

    try:
        model._cached_splits = splits
    except Exception:
        pass

    return splits


def tree_boundary_snap(
    predict_fn: Callable[[np.ndarray], float],
    model,
    x: np.ndarray,
    schema: NetworkFeatureSchema,
    x_orig: np.ndarray,
    epsilon_rel: float = 1e-4,
    enabled: FrozenSet[str] = ALL_GENERATORS
) -> Tuple[np.ndarray, float]:
    """
    Algorithm 3: Tree-Split Boundary Snapping (TSBS).
    Snaps mutable features across the closest active decision thresholds
    (including exact closed-form inversion of coupled rate, packet length, and window thresholds)
    while strictly preserving physical manifold invariants.
    Precondition: F(x) is malicious (score >= 0.5); an already-benign flow is returned unchanged.
    Returns (x_snapped, score).
    """
    x_star = x.copy()
    current_score = predict_fn(x_star)
    if current_score < 0.5:
        return x_star, current_score
        
    splits = extract_active_splits(model, x_star, schema)
    candidates = []
    
    # 1. Direct monotonic mutable features: closest split strictly above x*[j]
    closest_mono = {}
    for j, th in splits:
        if j in schema.monotonic_increasing_indices and x_star[j] < th:
            if _MONOTONIC_FEATURE_GENERATOR.get(j) not in enabled:
                continue
            if j == 11 and int(x_star[1]) == 17:
                continue  # UDP transport header is immutable at 8B
            if j not in closest_mono or th < closest_mono[j]:
                closest_mono[j] = th
    for j, th in closest_mono.items():
        cand = x_star.copy()
        eps = 1.0 if j in [2, 4, 11] else round(epsilon_rel * max(1.0, th))
        cand[j] = th + eps
        if j == 11:
            cand[11] = min(60.0 * cand[2], cand[11])
            cand[4] = min(1500.0 * cand[2], max(cand[4], cand[11] + 20.0 * cand[2]))
        candidates.append(cand)
        
    # 2. Coupled rate splits (8: flow_bytes_s, 9: flow_pkts_s) -> inverted into duration x[6]
    r8_splits = [th for j, th in splits if j == 8 and 0 < th < x_star[8]] if 'time' in enabled else []
    if r8_splits:
        th8 = max(r8_splits)  # closest threshold below current rate
        req_dur = (1e6 * (x_star[4] + x_star[5]) / th8) + 10.0
        if x_star[6] < req_dur <= 1.1e8:
            cand = x_star.copy()
            cand[6] = req_dur
            candidates.append(cand)
            
    r9_splits = [th for j, th in splits if j == 9 and 0 < th < x_star[9]] if 'time' in enabled else []
    if r9_splits:
        th9 = max(r9_splits)
        req_dur = (1e6 * (x_star[2] + x_star[3]) / th9) + 10.0
        if x_star[6] < req_dur <= 1.1e8:
            cand = x_star.copy()
            cand[6] = req_dur
            candidates.append(cand)
            
    # 3. Coupled forward mean packet length (10: fwd_pkt_len_mean) -> inverted into chaff
    is_tcp = (int(x_star[1]) == 6)
    if is_tcp:
        h_0 = 32.0 if (float(x_star[11]) / max(1.0, float(x_star[2])) >= 26.0) else 20.0
    else:
        h_0 = 8.0
    l_0 = 20.0 + h_0  # 40B/52B for TCP, 28B for UDP
    
    th10_below = [th for j, th in splits if j == 10 and l_0 < th < x_star[10]] if 'chaff' in enabled else []
    if th10_below:
        th10 = max(th10_below)
        req_chaff = int(np.floor((x_star[4] - th10 * x_star[2]) / (th10 - l_0))) + 1  # smallest integer N with mean < th10
        if 0 < req_chaff <= 1000:
            cand = x_star.copy()
            cand[2] += req_chaff
            cand[4] += req_chaff * l_0
            cand[11] += req_chaff * h_0
            candidates.append(cand)
            
    # 4. TCP Window Advertisement (7: init_fwd_win_byts)
    if is_tcp and 'win' in enabled and x_star[7] != 64240.0:
        cand = x_star.copy()
        cand[7] = 64240.0
        candidates.append(cand)
        
    # Evaluate candidates, selecting the one that minimizes malicious score
    best_cand = x_star
    best_score = current_score
    for cand in candidates:
        proj = project_network_invariants(cand, schema, x_orig)
        score = predict_fn(proj)
        if score < best_score:
            best_score = score
            best_cand = proj
            if best_score < 0.5:
                break
                
    return best_cand, best_score


class FlowFieldEvasionAttack(BaseNIDSAttack):
    """
    Flow-Field Evasion (FFE) Algorithm.
    Steers attack flows along the optimal trajectory of the Physical Lie Cone
    with Tree-Split Boundary Snapping (TSBS) to cross decision boundaries
    with minimal network overhead and guaranteed 100% physical realizability.
    """
    def __init__(
        self,
        model,
        schema: NetworkFeatureSchema,
        step_budget: int = 20,
        epsilon_rel: float = 1e-4,
        feature_stds: Optional[np.ndarray] = None,
        enabled_generators: FrozenSet[str] = ALL_GENERATORS,
        control_costs: Optional[Dict[str, float]] = None
    ):
        super().__init__(model, schema)
        self.step_budget = step_budget
        self.epsilon_rel = epsilon_rel
        self.feature_stds = feature_stds if feature_stds is not None else np.ones(len(schema.feature_names))
        
        # Operational unit costs as formalized in Section 4.4:
        # c1 = 10^-4 per second of delay
        # c2 = 10^-2 per byte of payload padding
        # c3 = 1.0 per transmitted chaff packet
        # c4 = 10^-3 per normalized TCP window morph
        self.enabled_generators = frozenset(enabled_generators)
        self.control_costs = {
            'time': 0.0001,   # cost per second
            'pad': 0.01,      # cost per byte
            'chaff': 1.0,     # cost per packet
            'win': 0.001      # cost per normalized window [0, 1]
        }
        if control_costs:
            self.control_costs = {**self.control_costs, **control_costs}
        if any(cost <= 0.0 for cost in self.control_costs.values()):
            raise ValueError('control_costs must be strictly positive (c_k > 0)')
        self.last_generator_usage: Dict[str, int] = {}

    def _probe_generators(self, x_curr: np.ndarray, p_curr: float, x_orig: np.ndarray) -> list:
        """Algorithm 2: one exploratory probe per enabled generator.

        Returns tuples (name, gradient per unit cost scale, direction, step, probed score, unit cost).
        """
        probes = []

        def probe(name: str, direction: np.ndarray, step: float, unit: float) -> None:
            x_test = project_network_invariants(x_curr + step * direction, self.schema, x_orig)
            p_test = self.predict_malicious_prob(x_test)
            grad = (p_test - p_curr) / (unit + 1e-9)
            probes.append((name, grad, direction, step, p_test, self.control_costs[name]))

        enabled = self.enabled_generators
        if 'time' in enabled:
            scale_t = max(1000.0, float(x_curr[6]) * 0.50)
            probe('time', PhysicalFlowGenerators.v_time(x_curr), scale_t, scale_t / 1e6)
        if 'pad' in enabled:
            scale_p = max(50.0, float(x_curr[4]) * 0.25)
            probe('pad', PhysicalFlowGenerators.v_pad(x_curr), scale_p, scale_p)
        if 'chaff' in enabled:
            scale_c = max(1.0, float(np.round(x_curr[2] * 0.50)))
            probe('chaff', PhysicalFlowGenerators.v_chaff(x_curr), scale_c, scale_c)
        if 'win' in enabled and int(x_curr[1]) == 6 and x_curr[7] != 64240.0:
            diff_win = 64240.0 - float(x_curr[7])
            # v_win already carries sign(diff_win); the step length is |diff_win| (Algorithm 2)
            probe('win', PhysicalFlowGenerators.v_win(x_curr, 64240.0), abs(diff_win), abs(diff_win) / 65535.0)
        return probes

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        self.last_generator_usage = {}
        x_orig = x.copy()
        x_curr = project_network_invariants(x_orig, self.schema, x_orig)
        
        # Initial base query
        p_curr = self.predict_malicious_prob(x_curr)
        if p_curr < 0.5:
            return AttackResult(
                x_original=x_orig,
                x_adversarial=x_curr,
                success=True,
                physically_valid=True,
                queries_used=self.query_count,
                weighted_cost=0.0,
                feature_shifts=np.zeros_like(x_orig)
            )

        # Iteratively integrate along the optimal physical Lie Cone
        for step in range(self.step_budget):
            if p_curr < 0.5:
                break  # Already classified benign: nothing left to snap

            # Evaluate Candidate Directional Lie Probes (Algorithm 2: 1 probe per enabled generator)
            candidate_directions = self._probe_generators(x_curr, p_curr, x_orig)

            # Select generator with the most negative Lie derivative per unit cost
            improving = [c for c in candidate_directions if c[1] < 0]
            if improving:
                best = max(improving, key=lambda c: (-c[1]) / c[5])
                x_curr = project_network_invariants(x_curr + best[3] * best[2], self.schema, x_orig)
                p_curr = best[4]
                self.last_generator_usage[best[0]] = self.last_generator_usage.get(best[0], 0) + 1
                continue

            # If local continuous Lie derivatives are flat/stagnated,
            # evaluate Tree-Split Boundary Snapping to cross nearest threshold
            x_snapped, p_snapped = tree_boundary_snap(
                self.predict_malicious_prob, self.model, x_curr, self.schema, x_orig,
                self.epsilon_rel, self.enabled_generators
            )
            if p_snapped < p_curr:
                x_curr, p_curr = x_snapped, p_snapped
                continue
            if 'time' not in self.enabled_generators:
                break  # No admissible fallback: terminate on stagnation
            # Low-and-slow temporal dilation fallback
            v_t = PhysicalFlowGenerators.v_time(x_curr)
            dt = min(1.1e8, max(50000.0, float(x_curr[6]) * 1.5))
            x_curr = project_network_invariants(x_curr + dt * v_t, self.schema, x_orig)
            p_curr = self.predict_malicious_prob(x_curr)
            self.last_generator_usage['time'] = self.last_generator_usage.get('time', 0) + 1

        success = (p_curr < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_curr, self.schema)
        
        # Dimensionless normalized perturbation distance D_L1
        d_l1 = float(np.mean(np.abs(x_curr - x_orig) / (self.feature_stds + 1e-9)))

        return AttackResult(
            x_original=x_orig,
            x_adversarial=x_curr,
            success=success,
            physically_valid=phys_valid,
            queries_used=self.query_count,
            weighted_cost=d_l1,
            feature_shifts=x_curr - x_orig
        )
