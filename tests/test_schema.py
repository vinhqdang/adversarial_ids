"""
Comprehensive unit tests for network feature schema and manifold projection operator.
Covers:
- Schema index structure
- Projection operator idempotence P(P(x)) = P(x)
- Strict monotonic lower bounds
- Protocol mechanics: MTU bound, TCP/UDP headers, RFC 7323 timestamp options
- Non-trivial physical feasibility validation and rejection
"""

import pytest
import numpy as np
from src.datasets.schema import get_cicids_schema, project_network_invariants, verify_physical_feasibility
from src.attacks.flow_field_evasion import PhysicalFlowGenerators


def test_schema_structure():
    schema = get_cicids_schema()
    assert len(schema.feature_names) == 12
    assert schema.immutable_indices == [0, 1, 3, 5]
    assert schema.monotonic_increasing_indices == [2, 4, 6, 11]
    assert schema.free_indices == [7]
    assert schema.coupled_indices == [8, 9, 10]


def test_projection_idempotence():
    """
    Verifies that the manifold projection operator is strictly idempotent:
    P_M(P_M(x)) == P_M(x) across arbitrary randomized off-manifold inputs.
    """
    schema = get_cicids_schema()
    rng = np.random.RandomState(42)
    
    for _ in range(50):
        # Generate random unconstrained continuous vector in R^12
        x_raw = rng.uniform(0.0, 50000.0, size=12)
        x_raw[0] = rng.choice([80.0, 443.0, 53.0])
        x_raw[1] = rng.choice([6.0, 17.0])
        
        proj_1 = project_network_invariants(x_raw, schema)
        proj_2 = project_network_invariants(proj_1, schema)
        
        # Second projection must be identical to the first
        np.testing.assert_allclose(proj_1, proj_2, rtol=1e-6, atol=1e-6)
        assert verify_physical_feasibility(proj_1, proj_2, schema)


def test_project_invariants_tcp_rfc7323():
    """
    Tests TCP protocol mechanics, including RFC 7323 32-byte header preservation.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 443.0
    x[1] = 6.0    # TCP
    x[2] = 10.0   # 10 fwd pkts
    x[3] = 8.0    # 8 bwd pkts
    x[4] = 600.0  # 600 fwd bytes
    x[5] = 400.0  # 400 bwd bytes
    x[6] = 500000.0 # 0.5s
    x[7] = 64240.0
    x[11] = 320.0 # 10 pkts * 32B (RFC 7323 TSOPT)
    
    proj = project_network_invariants(x, schema)
    assert proj[2] == 10.0
    assert proj[11] == 320.0
    assert proj[4] >= proj[11] + 20.0 * proj[2]
    assert proj[8] == (proj[4] + proj[5]) / 0.5
    assert proj[9] == (proj[2] + proj[3]) / 0.5
    assert proj[10] == proj[4] / proj[2]
    assert verify_physical_feasibility(x, proj, schema)
    
    # Test chaff preserves 32B options
    v_c = PhysicalFlowGenerators.v_chaff(proj)
    assert v_c[11] == 32.0 # Transport header for chaff packet
    assert v_c[4] == 52.0  # 20B IPv4 + 32B TCP header


def test_project_invariants_udp():
    """
    Tests UDP protocol invariants: 8B header, zero window.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 53.0
    x[1] = 17.0   # UDP
    x[2] = 4.0    # 4 pkts
    x[3] = 2.0
    x[4] = 200.0
    x[5] = 100.0
    x[6] = 50000.0 # 50ms
    x[7] = 8192.0  # Non-zero window must be clamped to 0
    x[11] = 32.0   # 4 pkts * 8B UDP header
    
    proj = project_network_invariants(x, schema)
    assert proj[7] == 0.0 # Clamped
    assert proj[11] == 4.0 * 8.0 # Strictly 8B/pkt
    assert verify_physical_feasibility(x, proj, schema)
    
    # Chaff for UDP must use 8B header and 28B total
    v_c = PhysicalFlowGenerators.v_chaff(proj)
    assert v_c[11] == 8.0
    assert v_c[4] == 28.0


def test_mtu_ceiling_clamping():
    """
    Tests MTU constraint: totlen_fwd_pkts <= 1500 * tot_fwd_pkts.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[1] = 6.0
    x[2] = 5.0
    x[4] = 20000.0 # Exceeds 5 * 1500 = 7500B
    x[6] = 100000.0
    x[11] = 100.0
    
    proj = project_network_invariants(x, schema)
    assert proj[4] == 5.0 * 1500.0 # Clamped to MTU


def test_verify_physical_feasibility_rejections():
    """
    Validates that verify_physical_feasibility strictly rejects:
    1. Decreased forward packet count
    2. Modified immutable backward features
    3. Uncoupled / decoupled rates
    4. Non-zero window for UDP
    """
    schema = get_cicids_schema()
    x_orig = np.zeros(12)
    x_orig[0] = 80.0
    x_orig[1] = 6.0
    x_orig[2] = 20.0
    x_orig[3] = 10.0
    x_orig[4] = 5000.0
    x_orig[5] = 3000.0
    x_orig[6] = 1000000.0
    x_orig[7] = 16384.0
    x_orig[11] = 400.0
    x_orig = project_network_invariants(x_orig, schema)
    
    # 1. Decreased forward packets
    x_bad1 = x_orig.copy()
    x_bad1[2] = 18.0
    assert not verify_physical_feasibility(x_orig, x_bad1, schema)
    
    # 2. Modified immutable backward packets or bytes
    x_bad2 = x_orig.copy()
    x_bad2[3] = 12.0
    assert not verify_physical_feasibility(x_orig, x_bad2, schema)
    
    x_bad3 = x_orig.copy()
    x_bad3[5] = 3500.0
    assert not verify_physical_feasibility(x_orig, x_bad3, schema)
    
    # 3. Decoupled rate (e.g. naive unconstrained Cartesian shift)
    x_bad4 = x_orig.copy()
    x_bad4[8] = 99999.0
    assert not verify_physical_feasibility(x_orig, x_bad4, schema)
    
    # 4. Non-zero window on UDP
    x_udp = x_orig.copy()
    x_udp[1] = 17.0
    x_udp = project_network_invariants(x_udp, schema)
    x_bad5 = x_udp.copy()
    x_bad5[7] = 1024.0
    assert not verify_physical_feasibility(x_udp, x_bad5, schema)
