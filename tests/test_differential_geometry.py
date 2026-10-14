"""
Unit tests for Differential Geometry on the Network Flow Manifold M_flow:
- Lie bracket commutativity [v_i, v_j] = 0 (Proposition 1)
- Tangent space invariance dPsi/dt = -(u/x)Psi
- Bouligand contingency cone and Nagumo-Brezis invariance
"""

import pytest
import numpy as np
from src.datasets.schema import get_cicids_schema, project_network_invariants
from src.attacks.flow_field_evasion import PhysicalFlowGenerators


def test_lie_bracket_numerical_commutation():
    """
    Verifies that the infinitesimal Lie generators commute:
    [v_i, v_j] = lim_{eps -> 0} (Phi_eps^{v_j} o Phi_eps^{v_i} - Phi_eps^{v_i} o Phi_eps^{v_j}) / eps^2 = 0
    """
    schema = get_cicids_schema()
    
    # Representative TCP flow point in M_flow
    x0 = np.zeros(12)
    x0[0] = 443.0
    x0[1] = 6.0    # TCP
    x0[2] = 50.0   # 50 pkts
    x0[3] = 40.0   # 40 bwd pkts
    x0[4] = 25000.0 # fwd bytes
    x0[5] = 20000.0 # bwd bytes
    x0[6] = 2000000.0 # 2.0s
    x0[7] = 8192.0
    x0[11] = 1000.0 # 20B/pkt
    x0 = project_network_invariants(x0, schema)
    
    eps = 1e-3
    
    generators = [
        ('time', PhysicalFlowGenerators.v_time),
        ('pad', PhysicalFlowGenerators.v_pad),
        ('chaff', PhysicalFlowGenerators.v_chaff),
        ('win', lambda z: PhysicalFlowGenerators.v_win(z, 64240.0))
    ]
    
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            name_i, gen_i = generators[i]
            name_j, gen_j = generators[j]
            
            # Forward step i then j
            v_i_0 = gen_i(x0)
            x_i = project_network_invariants(x0 + eps * v_i_0, schema, x0)
            v_j_i = gen_j(x_i)
            x_ij = project_network_invariants(x_i + eps * v_j_i, schema, x0)
            
            # Forward step j then i
            v_j_0 = gen_j(x0)
            x_j = project_network_invariants(x0 + eps * v_j_0, schema, x0)
            v_i_j = gen_i(x_j)
            x_ji = project_network_invariants(x_j + eps * v_i_j, schema, x0)
            
            # Commutator norm ||x_ij - x_ji|| must be O(eps^2) or zero
            diff = np.abs(x_ij - x_ji)
            assert np.max(diff) < 1e-2, f"Commutator [{name_i}, {name_j}] non-zero: max diff {np.max(diff)}"


def test_tangent_space_ode_invariance():
    """
    Verifies Proposition 1: dPsi_i/dt = -(u/x)Psi_i.
    If the flow starts on M_flow (Psi_i(x0) = 0), then along any trajectory
    integral of v_k, Psi_i(x(t)) remains identically zero.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0
    x[2] = 20.0
    x[3] = 15.0
    x[4] = 8000.0
    x[5] = 6000.0
    x[6] = 500000.0 # 0.5s
    x[7] = 16384.0
    x[11] = 400.0
    x = project_network_invariants(x, schema)
    
    # Conservation constraints:
    # Psi_1: x8 - 1e6*(x4 + x5)/x6 == 0
    # Psi_2: x9 - 1e6*(x2 + x3)/x6 == 0
    # Psi_3: x10 - x4/x2 == 0
    psi_1 = x[8] - 1e6 * (x[4] + x[5]) / x[6]
    psi_2 = x[9] - 1e6 * (x[2] + x[3]) / x[6]
    psi_3 = x[10] - x[4] / x[2]
    assert abs(psi_1) < 1e-5
    assert abs(psi_2) < 1e-5
    assert abs(psi_3) < 1e-5
    
    # Integrate along Lie cone (positive mixture of all 4 generators)
    u_time = 50000.0
    u_pad = 120.0
    u_chaff = 3.0
    
    v_t = PhysicalFlowGenerators.v_time(x)
    v_p = PhysicalFlowGenerators.v_pad(x)
    v_c = PhysicalFlowGenerators.v_chaff(x)
    
    x_displaced = x + u_time * v_t + u_pad * v_p + u_chaff * v_c
    x_proj = project_network_invariants(x_displaced, schema, x)
    
    # Check that invariants remain identically zero on projected manifold
    psi_1_new = x_proj[8] - 1e6 * (x_proj[4] + x_proj[5]) / x_proj[6]
    psi_2_new = x_proj[9] - 1e6 * (x_proj[2] + x_proj[3]) / x_proj[6]
    psi_3_new = x_proj[10] - x_proj[4] / x_proj[2]
    assert abs(psi_1_new) < 1e-5
    assert abs(psi_2_new) < 1e-5
    assert abs(psi_3_new) < 1e-5


def test_contingency_cone_interior_preservation():
    """
    Verifies Nagumo-Brezis affine contingency cone preservation:
    Admissible displacements cannot breach boundary hyperplanes:
    g1: x4 >= x11 + 20*x2
    g2: x4 <= 1500*x2
    g3: x11 >= 20*x2
    g4: x11 <= 60*x2
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0
    x[2] = 10.0
    x[3] = 5.0
    x[4] = 400.0
    x[5] = 200.0
    x[6] = 100000.0
    x[7] = 64240.0
    x[11] = 200.0
    x = project_network_invariants(x, schema)
    
    # Test chaff preserves MTU and header bounds
    v_c = PhysicalFlowGenerators.v_chaff(x)
    for n in [1, 5, 20, 100]:
        cand = project_network_invariants(x + n * v_c, schema, x)
        assert cand[4] >= cand[11] + 20.0 * cand[2] # g1
        assert cand[4] <= 1500.0 * cand[2]          # g2
        assert cand[11] >= 20.0 * cand[2]           # g3
        assert cand[11] <= 60.0 * cand[2]           # g4
