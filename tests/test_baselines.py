"""
Unit tests for competitive baseline attacks (ZOO, HopSkipJump, PSO, MOEVA).
"""

import numpy as np
import pytest

from src.attacks.base import AttackResult
from src.attacks.baselines import (
    ZOONIDSAttack,
    HopSkipJumpNIDSAttack,
    TrafficManipulatorPSOAttack,
    MOEVAAttack,
)
from src.datasets.schema import get_cicids_schema, project_network_invariants


class SimpleLinearModel:
    """Mock model where flow is classified malicious if fwd_bytes / fwd_pkts >= 300.0."""
    def __init__(self):
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        rates = X[:, 4] / np.maximum(X[:, 2], 1.0)
        malicious = (rates >= 300.0).astype(float)
        return np.column_stack([1.0 - malicious, malicious])


def test_zoo_attack_execution():
    schema = get_cicids_schema()
    model = SimpleLinearModel()
    zoo = ZOONIDSAttack(model, schema, max_iter=15, step_size=0.1)

    # Benign sample
    x_benign = np.zeros(12)
    x_benign[0], x_benign[1], x_benign[2], x_benign[4] = 443.0, 6.0, 10.0, 1000.0
    x_benign = project_network_invariants(x_benign, schema)
    res_b = zoo.attack_sample(x_benign)
    assert res_b.success is True
    assert res_b.queries_used == 1

    # Malicious sample
    x_mal = np.zeros(12)
    x_mal[0], x_mal[1], x_mal[2], x_mal[4], x_mal[6] = 443.0, 6.0, 10.0, 5000.0, 1e6
    x_mal = project_network_invariants(x_mal, schema)
    res_m = zoo.attack_sample(x_mal)
    assert isinstance(res_m, AttackResult)
    assert res_m.queries_used > 1


def test_hopskipjump_attack_execution():
    schema = get_cicids_schema()
    model = SimpleLinearModel()
    hsj = HopSkipJumpNIDSAttack(model, schema, max_iter=10)

    # Malicious sample
    x_mal = np.zeros(12)
    x_mal[0], x_mal[1], x_mal[2], x_mal[4], x_mal[6] = 443.0, 6.0, 10.0, 5000.0, 1e6
    x_mal = project_network_invariants(x_mal, schema)
    res = hsj.attack_sample(x_mal)
    assert isinstance(res, AttackResult)
    assert res.queries_used > 1


def test_traffic_manipulator_pso_execution():
    schema = get_cicids_schema()
    model = SimpleLinearModel()
    pso = TrafficManipulatorPSOAttack(model, schema, n_particles=5, max_iter=5)

    x_mal = np.zeros(12)
    x_mal[0], x_mal[1], x_mal[2], x_mal[4], x_mal[6] = 443.0, 6.0, 10.0, 5000.0, 1e6
    x_mal = project_network_invariants(x_mal, schema)
    res = pso.attack_sample(x_mal)
    assert isinstance(res, AttackResult)
    assert res.physically_valid is True  # PSO mutates duration and chaff respecting invariants


def test_moeva_attack_execution():
    schema = get_cicids_schema()
    model = SimpleLinearModel()
    moeva = MOEVAAttack(model, schema, n_particles=5, max_iter=5)

    x_mal = np.zeros(12)
    x_mal[0], x_mal[1], x_mal[2], x_mal[4], x_mal[6] = 443.0, 6.0, 10.0, 5000.0, 1e6
    x_mal = project_network_invariants(x_mal, schema)
    res = moeva.attack_sample(x_mal)
    assert isinstance(res, AttackResult)
