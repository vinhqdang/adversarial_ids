"""
Problem-Space Network Invariant Definitions for Network Intrusion Detection (CIC-IDS2018).

In network intrusion detection, tabular features extracted from packet flows
(e.g., via CICFlowMeter) must obey strict physical and protocol dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable
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
    monotonic = [2, 4, 6]                     # Fwd pkts, fwd bytes, duration
    free = [7]                                # Window size
    coupled = [8, 9, 10, 11]                  # Dependent flow statistics
    
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
        0.0, 0.0, 0.0, 0.0 # coupled (cost charged on base features)
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


def project_network_invariants(x: np.ndarray, schema: NetworkFeatureSchema) -> np.ndarray:
    """
    Exact Problem-Space Projection Operator P_M(x):
    Ensures that coupled features strictly match physical conservation laws.
    """
    x_proj = x.copy()
    
    # Base feature extraction
    tot_fwd_pkts = max(1.0, np.round(x_proj[2]))
    tot_bwd_pkts = max(0.0, np.round(x_proj[3]))
    totlen_fwd_pkts = max(tot_fwd_pkts * 20.0, x_proj[4]) # At least IP/TCP headers
    totlen_bwd_pkts = max(0.0, x_proj[5])
    flow_duration = max(1.0, x_proj[6]) # duration in microsecs
    
    # Store rounded/constrained base features
    x_proj[2] = tot_fwd_pkts
    x_proj[4] = totlen_fwd_pkts
    x_proj[6] = flow_duration
    
    # Invariant 1: Flow Bytes/s
    duration_sec = flow_duration / 1e6
    tot_bytes = totlen_fwd_pkts + totlen_bwd_pkts
    x_proj[8] = tot_bytes / duration_sec
    
    # Invariant 2: Flow Packets/s
    tot_pkts = tot_fwd_pkts + tot_bwd_pkts
    x_proj[9] = tot_pkts / duration_sec
    
    # Invariant 3: Fwd Packet Length Mean
    x_proj[10] = totlen_fwd_pkts / tot_fwd_pkts
    
    # Invariant 4: Forward Header Length
    # Minimum TCP header is 20 bytes per packet; can be higher with options
    x_proj[11] = max(tot_fwd_pkts * 20.0, x_proj[11])
    
    return x_proj


def verify_physical_feasibility(x_orig: np.ndarray, x_adv: np.ndarray, schema: NetworkFeatureSchema, tol: float = 1e-3) -> bool:
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
            
    # 3. Coupled Invariant check
    x_recomputed = project_network_invariants(x_adv, schema)
    for idx in schema.coupled_indices:
        relative_err = abs(x_adv[idx] - x_recomputed[idx]) / (abs(x_recomputed[idx]) + 1e-6)
        if relative_err > 0.05: # Allow max 5% numerical tolerance
            return False
            
    # 4. Minimum packet size invariant
    if x_adv[4] < x_adv[2] * 20.0 - tol:
        return False
        
    return True
