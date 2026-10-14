"""
Tests for TSBS closed-form inversions, generator-subset ablations and the
consistency between probed scores and the final adversarial score.
"""

import numpy as np
import pytest

from src.attacks.flow_field_evasion import (
    ALL_GENERATORS,
    FlowFieldEvasionAttack,
    PhysicalFlowGenerators,
    tree_boundary_snap,
)
from src.datasets.schema import get_cicids_schema, verify_physical_feasibility


class ThresholdModel:
    """Scores 1.0 (malicious) while fwd_pkt_len_mean >= threshold, else 0.0."""

    def __init__(self, feature_idx: int, threshold: float):
        self.feature_idx = feature_idx
        self.threshold = threshold

    def get_split_thresholds(self, feature_names=None):
        return [(self.feature_idx, self.threshold)]

    def predict_proba(self, x):
        x = np.atleast_2d(x)
        malicious = (x[:, self.feature_idx] >= self.threshold).astype(float)
        return np.column_stack([1.0 - malicious, malicious])


def _tcp_flow(fwd_pkts=10.0, bytes_per_pkt=400.0, hdr_per_pkt=20.0, duration=2e6, win=16384.0):
    x = np.zeros(12)
    x[0], x[1], x[2], x[3] = 443.0, 6.0, fwd_pkts, 2.0
    x[4], x[5], x[6], x[7] = fwd_pkts * bytes_per_pkt, 120.0, duration, win
    x[11] = fwd_pkts * hdr_per_pkt
    return get_cicids_schema(), x


def _projected(schema, x):
    from src.datasets.schema import project_network_invariants
    return project_network_invariants(x, schema)


def _snap(model, x, schema, enabled=ALL_GENERATORS):
    def predict(v):
        return float(model.predict_proba(v)[0, 1])
    return tree_boundary_snap(predict, model, x, schema, x.copy(), 1e-4, enabled)


@pytest.mark.parametrize('threshold', [100.0, 150.5, 200.0, 333.25])
def test_chaff_inversion_is_minimal_and_sufficient(threshold):
    schema, x = _tcp_flow()
    x = _projected(schema, x)
    model = ThresholdModel(10, threshold)
    snapped, score = _snap(model, x, schema)
    assert score < 0.5
    added = snapped[2] - x[2]
    assert added >= 1
    # One fewer chaff packet must not cross the threshold (minimality).
    l_0 = 40.0
    fewer_mean = (x[4] + (added - 1) * l_0) / (x[2] + added - 1)
    assert fewer_mean >= threshold
    assert snapped[10] < threshold


def test_chaff_candidate_uses_rfc7323_header_when_timestamps_present():
    schema, x = _tcp_flow(hdr_per_pkt=32.0, bytes_per_pkt=400.0)
    x = _projected(schema, x)
    model = ThresholdModel(10, 200.0)
    snapped, score = _snap(model, x, schema)
    added = snapped[2] - x[2]
    assert score < 0.5
    assert snapped[11] - x[11] == pytest.approx(added * 32.0)
    assert snapped[4] - x[4] == pytest.approx(added * 52.0)


def test_disabled_chaff_generator_blocks_chaff_candidate():
    schema, x = _tcp_flow()
    x = _projected(schema, x)
    model = ThresholdModel(10, 200.0)
    snapped, score = _snap(model, x, schema, enabled=frozenset({'time', 'pad', 'win'}))
    assert score >= 0.5
    assert snapped[2] == x[2]


def test_window_generator_lands_exactly_on_target_without_overshoot():
    _, x = _tcp_flow(win=1024.0)
    v = PhysicalFlowGenerators.v_win(x, 64240.0)
    step = abs(64240.0 - x[7])
    assert (x + step * v)[7] == pytest.approx(64240.0)
    _, x_hi = _tcp_flow(win=65535.0)
    v_hi = PhysicalFlowGenerators.v_win(x_hi, 64240.0)
    assert (x_hi + abs(64240.0 - x_hi[7]) * v_hi)[7] == pytest.approx(64240.0)


def test_window_generator_is_zero_for_udp():
    _, x = _tcp_flow()
    x[1], x[7] = 17.0, 0.0
    assert not np.any(PhysicalFlowGenerators.v_win(x, 64240.0))


def test_attack_with_only_window_generator_changes_only_window_and_stays_feasible():
    schema, x = _tcp_flow(win=1024.0)
    x = _projected(schema, x)

    class WindowModel(ThresholdModel):
        def predict_proba(self, v):
            v = np.atleast_2d(v)
            malicious = (v[:, 7] < 30000.0).astype(float)
            return np.column_stack([1.0 - malicious, malicious])

    model = WindowModel(7, 30000.0)
    attack = FlowFieldEvasionAttack(model, schema, enabled_generators=frozenset({'win'}))
    result = attack.attack_sample(x)
    assert result.success and result.physically_valid
    shifts = result.feature_shifts
    assert shifts[7] != 0.0
    assert not np.any(np.delete(shifts, 7))
    assert attack.last_generator_usage == {'win': 1}


def test_attack_without_time_generator_terminates_on_stagnation():
    schema, x = _tcp_flow(win=64240.0)
    x = _projected(schema, x)

    class ConstantModel(ThresholdModel):
        def predict_proba(self, v):
            v = np.atleast_2d(v)
            return np.column_stack([np.zeros(len(v)), np.ones(len(v))])

    attack = FlowFieldEvasionAttack(ConstantModel(0, 0.0), schema, step_budget=20,
                                    enabled_generators=frozenset({'pad', 'chaff', 'win'}))
    result = attack.attack_sample(x)
    assert not result.success
    # Stagnation must stop the loop long before the 20-step budget would be spent.
    assert result.queries_used < 20


def test_custom_control_costs_override_only_given_keys():
    schema, _ = _tcp_flow()
    attack = FlowFieldEvasionAttack(ThresholdModel(10, 1.0), schema, control_costs={'time': 10.0})
    assert attack.control_costs['time'] == 10.0
    assert attack.control_costs['pad'] == 0.01


def test_successful_attack_final_score_matches_requery_and_is_feasible():
    schema, x = _tcp_flow()
    x = _projected(schema, x)
    model = ThresholdModel(10, 200.0)
    attack = FlowFieldEvasionAttack(model, schema)
    result = attack.attack_sample(x)
    assert result.success
    assert model.predict_proba(result.x_adversarial)[0, 1] < 0.5
    assert verify_physical_feasibility(x, result.x_adversarial, schema)


def test_window_probe_moves_toward_target_when_window_above_target():
    schema, x = _tcp_flow(win=65535.0)
    x = _projected(schema, x)

    class HighWindowModel(ThresholdModel):
        def predict_proba(self, v):
            v = np.atleast_2d(v)
            malicious = (v[:, 7] >= 65000.0).astype(float)
            return np.column_stack([1.0 - malicious, malicious])

    attack = FlowFieldEvasionAttack(HighWindowModel(7, 65000.0), schema, enabled_generators=frozenset({'win'}))
    result = attack.attack_sample(x)
    assert result.success and result.physically_valid
    assert result.x_adversarial[7] == pytest.approx(64240.0)


def test_nonpositive_control_cost_is_rejected():
    schema, _ = _tcp_flow()
    with pytest.raises(ValueError):
        FlowFieldEvasionAttack(ThresholdModel(10, 1.0), schema, control_costs={'time': 0.0})


def test_udp_chaff_uses_28_byte_probes_and_8_byte_headers():
    schema, x = _tcp_flow(bytes_per_pkt=400.0)
    x[1], x[7], x[11] = 17.0, 0.0, 8.0 * x[2]
    x = _projected(schema, x)
    model = ThresholdModel(10, 200.0)
    snapped, score = _snap(model, x, schema)
    added = snapped[2] - x[2]
    assert score < 0.5 and added >= 1
    assert snapped[4] - x[4] == pytest.approx(added * 28.0)
    assert snapped[11] - x[11] == pytest.approx(added * 8.0)


def test_already_benign_flow_costs_exactly_one_query():
    schema, x = _tcp_flow(bytes_per_pkt=50.0)
    x = _projected(schema, x)
    attack = FlowFieldEvasionAttack(ThresholdModel(10, 1000.0), schema)
    result = attack.attack_sample(x)
    assert result.success and result.queries_used == 1


def test_stagnation_fallback_dilates_when_time_generator_enabled():
    schema, x = _tcp_flow(win=64240.0)
    x = _projected(schema, x)

    class DurationModel(ThresholdModel):
        def get_split_thresholds(self, feature_names=None):
            return []  # no TSBS candidates: only the temporal-dilation fallback can help

        def predict_proba(self, v):
            v = np.atleast_2d(v)
            malicious = (v[:, 6] < 5e6).astype(float)
            return np.column_stack([1.0 - malicious, malicious])

    attack = FlowFieldEvasionAttack(DurationModel(6, 5e6), schema)
    result = attack.attack_sample(x)
    assert result.success and result.physically_valid
    assert result.x_adversarial[6] >= 5e6
    assert attack.last_generator_usage.get('time', 0) >= 1
