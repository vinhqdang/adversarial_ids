"""
Comprehensive end-to-end unit tests for Flow-Field Evasion and Tree-Split Boundary Snapping (TSBS).
Covers:
- Active split extraction from GBDT models
- Direct monotonic TSBS snapping
- Coupled rate split inversion (features 8 & 9 -> duration dilation x_6)
- Coupled packet length mean split inversion (feature 10 -> chaff packet injection)
- TCP Window advertisement morphing to benign OS standard
- Query accounting complexity bounds (queries <= 4 * M_max + 1)
- 100% Physical Feasibility Rate (PFR) across all 5 individual attack families
"""

import pytest
import numpy as np
from src.datasets.schema import get_cicids_schema, project_network_invariants, verify_physical_feasibility
from src.datasets.generator import generate_cicids_traffic, generate_attack_family_cohorts
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack, tree_boundary_snap, extract_active_splits, PhysicalFlowGenerators


@pytest.fixture(scope="module")
def trained_models_and_data():
    X, y, df = generate_cicids_traffic(n_samples=3000, attack_ratio=0.3, random_state=42)
    schema = get_cicids_schema()
    
    trainer = NIDSEnsembleTrainer('xgboost', {'n_estimators': 30, 'max_depth': 5, 'random_state': 42})
    trainer.train(X[:2200], y[:2200])
    
    feature_stds = np.std(X[:2200], axis=0) + 1e-6
    return trainer, X[2200:], y[2200:], schema, feature_stds


def test_active_splits_extraction(trained_models_and_data):
    trainer, X_test, y_test, schema, feature_stds = trained_models_and_data
    splits = extract_active_splits(trainer, X_test[0], schema)
    assert len(splits) > 0
    for feat_idx, thresh in splits:
        assert 0 <= feat_idx < 12
        assert isinstance(thresh, float)


def test_tsbs_direct_monotonic_snap(trained_models_and_data):
    trainer, X_test, y_test, schema, feature_stds = trained_models_and_data
    s = X_test[y_test == 1][0].copy()
    
    # Run TSBS directly
    x_snap, p_snap = tree_boundary_snap(
        lambda z: trainer.predict_proba(z.reshape(1, -1))[0, 1],
        trainer, s, schema, s
    )
    assert verify_physical_feasibility(s, x_snap, schema)
    assert p_snap <= trainer.predict_proba(s.reshape(1, -1))[0, 1]


def test_tsbs_coupled_rate_inversion():
    """
    Tests that TSBS inverts an active byte rate split threshold theta_8 into duration dilation:
    x_6 >= 1e6 * (x_4 + x_5) / theta_8
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0
    x[2] = 50.0
    x[3] = 2.0
    x[4] = 2000.0
    x[5] = 80.0
    x[6] = 10000.0 # 10ms -> byte rate is 208,000 B/s
    x[7] = 1024.0
    x[11] = 1000.0
    x = project_network_invariants(x, schema)
    assert x[8] == 208000.0
    
    # Mock model that flags flows with flow_bytes_s > 50000.0
    class MockRateModel:
        def predict_proba(self, X):
            probs = []
            for row in X:
                p = 0.95 if row[8] > 50000.0 else 0.10
                probs.append([1.0 - p, p])
            return np.array(probs)
        def get_split_thresholds(self, feat_names):
            return [(8, 50000.0)]
            
    mock = MockRateModel()
    p_init = mock.predict_proba(x.reshape(1, -1))[0, 1]
    assert p_init == 0.95
    
    x_snapped, p_snapped = tree_boundary_snap(
        lambda z: mock.predict_proba(z.reshape(1, -1))[0, 1],
        mock, x, schema, x
    )
    assert p_snapped < 0.5
    assert x_snapped[8] <= 50000.0
    assert x_snapped[6] >= 1e6 * (x[4] + x[5]) / 50000.0
    assert verify_physical_feasibility(x, x_snapped, schema)


def test_tsbs_coupled_packet_length_inversion():
    """
    Tests that TSBS inverts an active mean packet length split theta_10 into chaff packets.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0
    x[2] = 10.0
    x[3] = 5.0
    x[4] = 9000.0 # Mean packet length = 900B
    x[5] = 500.0
    x[6] = 1000000.0
    x[7] = 16384.0
    x[11] = 200.0
    x = project_network_invariants(x, schema)
    assert x[10] == 900.0
    
    # Mock model that flags flows with fwd_pkt_len_mean > 300.0
    class MockLengthModel:
        def predict_proba(self, X):
            probs = []
            for row in X:
                p = 0.95 if row[10] > 300.0 else 0.10
                probs.append([1.0 - p, p])
            return np.array(probs)
        def get_split_thresholds(self, feat_names):
            return [(10, 300.0)]
            
    mock = MockLengthModel()
    x_snapped, p_snapped = tree_boundary_snap(
        lambda z: mock.predict_proba(z.reshape(1, -1))[0, 1],
        mock, x, schema, x
    )
    assert p_snapped < 0.5
    assert x_snapped[10] <= 300.0
    assert x_snapped[2] > x[2] # Injected chaff packets
    assert verify_physical_feasibility(x, x_snapped, schema)


def test_ffe_query_accounting_bound(trained_models_and_data):
    """
    Verifies that query complexity satisfies the theoretical upper bound:
    queries <= 4 * step_budget + 1
    """
    trainer, X_test, y_test, schema, feature_stds = trained_models_and_data
    step_budget = 15
    attack = FlowFieldEvasionAttack(trainer, schema, step_budget=step_budget, feature_stds=feature_stds)
    
    detected = [s for s, label in zip(X_test, y_test) if label == 1 and trainer.predict_proba(s.reshape(1, -1))[0, 1] >= 0.5][:20]
    
    for sample in detected:
        res = attack.attack_sample(sample)
        assert res.queries_used >= 1
        assert res.queries_used <= 4 * step_budget + 1, f"Exceeded query bound: {res.queries_used}"
        assert res.physically_valid
        assert verify_physical_feasibility(sample, res.x_adversarial, schema)


def test_ffe_all_five_attack_families(trained_models_and_data):
    """
    Verifies 100% Physical Feasibility Rate (PFR) across all 5 distinct intrusion families.
    """
    trainer, X_test, y_test, schema, feature_stds = trained_models_and_data
    attack = FlowFieldEvasionAttack(trainer, schema, step_budget=20, feature_stds=feature_stds)
    cohorts = generate_attack_family_cohorts(n_per_family=10, random_state=42)
    
    for fam_name, samples in cohorts.items():
        for s in samples:
            res = attack.attack_sample(s)
            assert res.physically_valid
            assert verify_physical_feasibility(s, res.x_adversarial, schema)
            assert res.weighted_cost >= 0.0


def test_tsbs_rfc7323_chaff_option_synchronization():
    """
    Verifies that TSBS injects 32-byte header (52-byte total) chaff packets
    when a flow has negotiated RFC 7323 TCP timestamp options.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0 # TCP
    x[2] = 10.0 # 10 packets
    x[3] = 5.0
    x[11] = 320.0 # 32 bytes per packet (RFC 7323 TSOPT)
    x[4] = 8000.0 # Mean packet length = 800B
    x[5] = 500.0
    x[6] = 1000000.0
    x[7] = 16384.0
    x = project_network_invariants(x, schema)
    assert x[11] / x[2] == 32.0
    
    class MockOptionModel:
        def predict_proba(self, X):
            return np.array([[0.05, 0.95] if row[10] > 200.0 else [0.95, 0.05] for row in X])
        def get_split_thresholds(self, feat_names):
            return [(10, 200.0)]
            
    mock = MockOptionModel()
    x_snapped, p_snapped = tree_boundary_snap(
        lambda z: mock.predict_proba(z.reshape(1, -1))[0, 1],
        mock, x, schema, x
    )
    assert p_snapped < 0.5
    injected_chaff = x_snapped[2] - x[2]
    assert injected_chaff > 0
    added_hdr = x_snapped[11] - x[11]
    added_bytes = x_snapped[4] - x[4]
    # Each chaff packet must have exactly 32 bytes header and 52 bytes total
    assert np.isclose(added_hdr / injected_chaff, 32.0)
    assert np.isclose(added_bytes / injected_chaff, 52.0)
    assert verify_physical_feasibility(x, x_snapped, schema)


def test_tsbs_packet_rate_theta9_inversion():
    """
    Verifies that TSBS inverts an active packet rate threshold theta_9 into duration dilation.
    """
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[0] = 80.0
    x[1] = 6.0
    x[2] = 50.0
    x[3] = 50.0
    x[4] = 4000.0
    x[5] = 4000.0
    x[6] = 10000.0 # 0.01s -> flow_pkts_s = 10,000 pkts/s
    x[7] = 16384.0
    x[11] = 1000.0
    x = project_network_invariants(x, schema)
    assert x[9] == 10000.0
    
    class MockPacketRateModel:
        def predict_proba(self, X):
            return np.array([[0.05, 0.95] if row[9] > 1000.0 else [0.95, 0.05] for row in X])
        def get_split_thresholds(self, feat_names):
            return [(9, 1000.0)]
            
    mock = MockPacketRateModel()
    x_snapped, p_snapped = tree_boundary_snap(
        lambda z: mock.predict_proba(z.reshape(1, -1))[0, 1],
        mock, x, schema, x
    )
    assert p_snapped < 0.5
    assert x_snapped[9] <= 1000.0
    assert x_snapped[6] >= 1e6 * (x[2] + x[3]) / 1000.0
    assert verify_physical_feasibility(x, x_snapped, schema)


def test_udp_tcp_window_constraints():
    """
    Verifies window morphing behaviors for TCP vs UDP flows.
    """
    schema = get_cicids_schema()
    
    # UDP: window must remain strictly 0.0
    x_udp = np.array([53.0, 17.0, 10.0, 0.0, 500.0, 0.0, 50000.0, 0.0, 10000.0, 200.0, 50.0, 80.0])
    x_udp = project_network_invariants(x_udp, schema)
    v_w_udp = PhysicalFlowGenerators.v_win(x_udp, 64240.0)
    assert np.all(v_w_udp == 0.0)
    assert verify_physical_feasibility(x_udp, x_udp, schema)
    
    # TCP: window morphs toward 64240
    x_tcp = np.array([80.0, 6.0, 10.0, 5.0, 500.0, 200.0, 50000.0, 1024.0, 14000.0, 300.0, 50.0, 200.0])
    x_tcp = project_network_invariants(x_tcp, schema)
    v_w_tcp = PhysicalFlowGenerators.v_win(x_tcp, 64240.0)
    assert v_w_tcp[7] == 1.0


def test_worst_case_query_bound_adversarial_stall():
    """
    Simulates a stubborn adversary model that stalls for several steps,
    verifying query accounting strictly adheres to <= 4 * step_budget + K_TSBS + 1.
    """
    schema = get_cicids_schema()
    x = np.array([80.0, 6.0, 20.0, 10.0, 1500.0, 800.0, 20000.0, 2048.0, 115000.0, 1500.0, 75.0, 400.0])
    x = project_network_invariants(x, schema)
    
    step_calls = 0
    class StubbornModel:
        def predict_proba(self, X):
            nonlocal step_calls
            step_calls += len(X)
            # Stay malicious for first 3 iterations (each iteration evaluates probes), then turn benign
            p = 0.95 if step_calls < 15 else 0.10
            return np.array([[1.0 - p, p] for _ in X])
        def get_split_thresholds(self, feat_names):
            return [(8, 50000.0), (10, 50.0)]
            
    mock = StubbornModel()
    budget = 10
    attack = FlowFieldEvasionAttack(mock, schema, step_budget=budget)
    res = attack.attack_sample(x)
    assert res.success
    # Total queries must be strictly bounded
    assert res.queries_used <= 4 * budget + 10
    assert res.physically_valid


def test_extract_active_splits_raw_xgb_and_lgb_fallbacks():
    """Tests the fallback branch of extract_active_splits for raw XGBoost and LightGBM models."""
    schema = get_cicids_schema()
    X = np.random.RandomState(42).randn(100, 12)
    y = (X[:, 2] > 0).astype(int)

    # 1. Raw XGBClassifier
    from xgboost import XGBClassifier
    raw_xgb = XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    raw_xgb.fit(X, y)
    splits_xgb = extract_active_splits(raw_xgb, X[0], schema)
    assert isinstance(splits_xgb, list)
    assert len(splits_xgb) > 0

    # 2. Raw LGBMClassifier
    import lightgbm as lgb
    raw_lgb = lgb.LGBMClassifier(n_estimators=5, max_depth=3, random_state=42, verbosity=-1)
    raw_lgb.fit(X, y)
    splits_lgb = extract_active_splits(raw_lgb, X[0], schema)
    assert isinstance(splits_lgb, list)
    assert len(splits_lgb) > 0


