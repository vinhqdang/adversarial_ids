"""
Problem-Space Network Invariant Definitions for Network Intrusion Detection (CIC-IDS2018).

In network intrusion detection, tabular features extracted from packet flows
(e.g., via CICFlowMeter) must obey strict physical and protocol dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import numpy as np


@dataclass
class NetworkFeatureSchema:
    """
    Categorizes features into:
    - Immutable: Cannot be modified without breaking attack delivery
    - Monotonically Non-Decreasing (Additive-Only): Can only be increased (e.g. by injecting packets or padding)
    - Free / Bounded: Can be varied within bounds (e.g. inter-arrival delays)
    - Coupled: Mathematically derived from base features and must be recalculated
    """
    feature_names: List[str]
    immutable_indices: List[int]
    monotonic_increasing_indices: List[int]
    free_indices: List[int]
    coupled_indices: List[int]
    bounds: Dict[int, tuple] = field(default_factory=dict)
    
    # Feature weights representing the physical cost of perturbation
    # E.g., altering packet sizes costs bandwidth; injecting packets costs socket resources
    perturbation_costs: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.perturbation_costs is None:
            self.perturbation_costs = np.ones(len(self.feature_names))


def get_cicids_schema() -> NetworkFeatureSchema:
    """
    Standard CIC-IDS2018 flow feature set subset highlighting core invariants:
    
    Base Features:
    0: 'dst_port'              (Immutable: target service)
    1: 'protocol'              (Immutable: TCP=6, UDP=17)
    2: 'tot_fwd_pkts'          (Monotonic +: attacker can inject dummy packets)
    3: 'tot_bwd_pkts'          (Dependent on server response or immutable by attacker)
    4: 'totlen_fwd_pkts'       (Monotonic +: attacker can add dummy payload bytes)
    5: 'totlen_bwd_pkts'       (Immutable by attacker alone)
    6: 'flow_duration'         (Monotonic +: attacker can introduce delays, increasing duration)
    7: 'init_fwd_win_byts'     (Free / Bounded: TCP window advertisement)
    
    Coupled / Derived Features (Violated by naive attacks):
    8: 'flow_bytes_s'          = (totlen_fwd_pkts + totlen_bwd_pkts) / (flow_duration / 1e6)
    9: 'flow_pkts_s'           = (tot_fwd_pkts + tot_bwd_pkts) / (flow_duration / 1e6)
    10: 'fwd_pkt_len_mean'     = totlen_fwd_pkts / max(1, tot_fwd_pkts)
    11: 'fwd_header_len'       >= tot_fwd_pkts * 20 (minimum TCP header length)
    """
    features = [
        'dst_port',             # 0
        'protocol',             # 1
        'tot_fwd_pkts',         # 2
        'tot_bwd_pkts',         # 3
        'totlen_fwd_pkts',      # 4
        'totlen_bwd_pkts',      # 5
        'flow_duration',        # 6 (in microseconds)
        'init_fwd_win_byts',    # 7
        'flow_bytes_s',         # 8 (Coupled)
        'flow_pkts_s',          # 9 (Coupled)
        'fwd_pkt_len_mean',     # 10 (Coupled)
        'fwd_header_len'        # 11 (Coupled)
    ]
    
    immutable = [0, 1, 3, 5]                  # Port, proto, server bwd packets/bytes
    monotonic = [2, 4, 6, 11]                 # Fwd pkts, fwd bytes, duration, header len
    free = [7]                                # Window size
    coupled = [8, 9, 10]                      # Dependent flow statistics (rate/mean)
    
    bounds = {
        0: (1, 65535),
        1: (6, 17),
        2: (1, 100000),
        3: (0, 100000),
        4: (20, 100000000),
        5: (0, 100000000),
        6: (1.0, 120000000.0), # up to 120s
        7: (0, 65535)
    }
    
    costs = np.array([
        1000.0, 1000.0,  # port, proto (high penalty)
        1.0,             # fwd pkts
        1000.0,          # bwd pkts (cannot directly control)
        0.01,            # fwd bytes (low penalty to pad bytes)
        1000.0,          # bwd bytes
        0.001,           # duration delay (cheap to delay)
        0.5,             # window bytes
        0.0, 0.0, 0.0,   # coupled rates (8, 9, 10)
        1.0              # transport header length (11)
    ])
    
    return NetworkFeatureSchema(
        feature_names=features,
        immutable_indices=immutable,
        monotonic_increasing_indices=monotonic,
        free_indices=free,
        coupled_indices=coupled,
        bounds=bounds,
        perturbation_costs=costs
    )


def project_network_invariants(x: np.ndarray, schema: NetworkFeatureSchema, x_orig: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Exact Problem-Space Projection Operator P_M(x):
    Ensures that discrete counts, header lengths, MTU bounds, and coupled features
    strictly match physical protocol conservation laws on M_flow.
    If x_orig is provided, enforces that monotonic features never decrease below original values
    and immutable features are strictly preserved.
    """
    x_proj = x.copy()
    
    # If original sample is provided, enforce immutability and monotonicity lower bounds
    if x_orig is not None:
        for idx in schema.immutable_indices:
            x_proj[idx] = x_orig[idx]
        for idx in schema.monotonic_increasing_indices:
            x_proj[idx] = max(x_orig[idx], x_proj[idx])
    
    is_tcp = (x_proj[1] == 6)
    h_min = 20.0 if is_tcp else 8.0
    h_max = 60.0 if is_tcp else 8.0
    
    # Base feature extraction
    tot_fwd_pkts = max(1.0, float(np.round(x_proj[2])))
    tot_bwd_pkts = max(0.0, float(np.round(x_proj[3])))
    
    # Transport header length
    raw_hdr = max(x_orig[11], float(np.round(x_proj[11]))) if x_orig is not None else float(np.round(x_proj[11]))
    fwd_hdr_len = min(tot_fwd_pkts * h_max, max(tot_fwd_pkts * h_min, raw_hdr))
    
    # Total forward payload bytes: must be >= 0 and <= MTU payload (1500 bytes * tot_fwd_pkts)
    raw_bytes = max(x_orig[4], float(np.round(x_proj[4]))) if x_orig is not None else float(np.round(x_proj[4]))
    totlen_fwd_pkts = min(tot_fwd_pkts * 1500.0, max(0.0, raw_bytes))
    totlen_bwd_pkts = max(0.0, float(x_proj[5]))
    
    raw_dur = max(x_orig[6], float(np.round(x_proj[6]))) if x_orig is not None else float(np.round(x_proj[6]))
    flow_duration = min(1.1e8, max(1.0, raw_dur)) # duration in microsecs <= 110s
    
    # TCP Window size
    win_fwd = float(np.clip(np.round(x_proj[7]), 0.0, 65535.0)) if is_tcp else 0.0
    
    # Store rounded/constrained base features
    x_proj[2] = tot_fwd_pkts
    x_proj[3] = tot_bwd_pkts
    x_proj[4] = totlen_fwd_pkts
    x_proj[6] = flow_duration
    x_proj[7] = win_fwd
    x_proj[11] = fwd_hdr_len
    
    # Invariant 1: Flow Bytes/s
    duration_sec = max(1e-6, flow_duration / 1e6)
    tot_bytes = totlen_fwd_pkts + totlen_bwd_pkts
    x_proj[8] = tot_bytes / duration_sec
    
    # Invariant 2: Flow Packets/s
    tot_pkts = tot_fwd_pkts + tot_bwd_pkts
    x_proj[9] = tot_pkts / duration_sec
    
    # Invariant 3: Fwd Packet Length Mean
    x_proj[10] = totlen_fwd_pkts / max(1.0, tot_fwd_pkts)
    
    return x_proj


def verify_physical_feasibility(x_orig: np.ndarray, x_adv: np.ndarray, schema: NetworkFeatureSchema, tol: float = 1e-2) -> bool:
    """
    Audits an adversarial flow to determine if it is physically realizable
    or if it is a flawed/impossible 'feature-space hallucination'.
    """
    # 1. Immutable feature check
    for idx in schema.immutable_indices:
        if abs(x_adv[idx] - x_orig[idx]) > tol:
            return False
            
    # 2. Monotonicity check (attacker can only add packets/bytes/delays, never remove)
    for idx in schema.monotonic_increasing_indices:
        if x_adv[idx] < x_orig[idx] - tol:
            return False
            
    # 3. Non-negativity check
    if (x_adv < -tol).any():
        return False
        
    # 4. MTU bound: payload <= 1500 * packets
    if x_adv[4] > 1500.0 * x_adv[2] + tol:
        return False
        
    # 5. Transport header bounds
    is_tcp = (x_adv[1] == 6)
    h_min = 20.0 if is_tcp else 8.0
    h_max = 60.0 if is_tcp else 8.0
    if x_adv[11] < x_adv[2] * h_min - tol or x_adv[11] > x_adv[2] * h_max + 10.0 + tol:
        return False
        
    # 6. Coupled Invariant check (strict 5% numerical tolerance on rate equations)
    dur_s = max(1e-6, x_adv[6] / 1e6)
    expected_bytes_s = (x_adv[4] + x_adv[5]) / dur_s
    if abs(x_adv[8] - expected_bytes_s) / (expected_bytes_s + 1e-4) > 0.05:
        return False
        
    expected_pkts_s = (x_adv[2] + x_adv[3]) / dur_s
    if abs(x_adv[9] - expected_pkts_s) / (expected_pkts_s + 1e-4) > 0.05:
        return False
        
    expected_mean = x_adv[4] / max(1.0, x_adv[2])
    if abs(x_adv[10] - expected_mean) / (expected_mean + 1e-4) > 0.05:
        return False
        
    # 7. Protocol-specific checks: UDP window must be strictly 0
    if not is_tcp and abs(x_adv[7]) > tol:
        return False
        
    return True

