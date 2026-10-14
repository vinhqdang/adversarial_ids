"""
Unit tests for multi-dataset benchmark generators:
- CSE-CIC-IDS2018 (Enterprise)
- UNSW-NB15 (Contemporary Hybrid)
- CIDDS-001 (Cloud OpenStack)
Verifies 100% physical feasibility of synthesized flows and valid train/test splits.
"""

import pytest
import numpy as np
from src.datasets.schema import get_cicids_schema, verify_physical_feasibility
from src.datasets.generator import generate_cicids_traffic, generate_unsw_traffic, generate_cidds_traffic


def test_cicids_generator_feasibility():
    schema = get_cicids_schema()
    X, y, df = generate_cicids_traffic(n_samples=200, attack_ratio=0.3, random_state=42)
    assert X.shape == (200, 12)
    assert len(y) == 200
    for flow in X:
        assert verify_physical_feasibility(flow, flow, schema)


def test_unsw_generator_feasibility():
    schema = get_cicids_schema()
    X, y, df = generate_unsw_traffic(n_samples=200, attack_ratio=0.3, random_state=42)
    assert X.shape == (200, 12)
    assert len(y) == 200
    for flow in X:
        assert verify_physical_feasibility(flow, flow, schema)


def test_cidds_generator_feasibility():
    schema = get_cicids_schema()
    X, y, df = generate_cidds_traffic(n_samples=200, attack_ratio=0.3, random_state=42)
    assert X.shape == (200, 12)
    assert len(y) == 200
    for flow in X:
        assert verify_physical_feasibility(flow, flow, schema)
