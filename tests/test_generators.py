"""
Unit tests for infinitesimal physical Lie flow generators and Lie commutativity.
"""

import pytest
import numpy as np
from src.datasets.schema import get_cicids_schema, project_network_invariants
from src.attacks.flow_field_evasion import PhysicalFlowGenerators


def test_v_time_effects():
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[1] = 6.0
    x[2] = 10.0
    x[4] = 1000.0
    x[6] = 100000.0 # 0.1s
    x[11] = 200.0
    x = project_network_invariants(x, schema)
    
    v = PhysicalFlowGenerators.v_time(x)
    assert v[6] == 1.0
    assert v[8] < 0 # Byte rate decreases
    assert v[9] < 0 # Packet rate decreases
    assert v[2] == 0 # Packet count unaffected


def test_v_pad_effects():
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[1] = 6.0
    x[2] = 10.0
    x[4] = 1000.0
    x[6] = 100000.0
    x[11] = 200.0
    x = project_network_invariants(x, schema)
    
    v = PhysicalFlowGenerators.v_pad(x)
    assert v[4] == 1.0
    assert v[8] > 0 # Byte rate increases
    assert v[10] > 0 # Mean length increases
    assert v[2] == 0 # Packet count unaffected


def test_v_chaff_effects():
    schema = get_cicids_schema()
    x = np.zeros(12)
    x[1] = 6.0 # TCP
    x[2] = 10.0
    x[4] = 1000.0
    x[6] = 100000.0
    x[11] = 200.0
    x = project_network_invariants(x, schema)
    
    v = PhysicalFlowGenerators.v_chaff(x)
    assert v[2] == 1.0 # 1 packet
    assert v[4] == 40.0 # 20B IP + 20B TCP
    assert v[11] == 20.0 # Transport header
    assert v[8] > 0 # Byte rate increases
    assert v[9] > 0 # Packet rate increases


def test_v_win_effects():
    schema = get_cicids_schema()
    x_tcp = np.zeros(12)
    x_tcp[1] = 6.0
    x_tcp[7] = 1024.0
    v_tcp = PhysicalFlowGenerators.v_win(x_tcp, 64240.0)
    assert v_tcp[7] > 0 # Steers toward 64240
    
    x_udp = np.zeros(12)
    x_udp[1] = 17.0
    x_udp[7] = 0.0
    v_udp = PhysicalFlowGenerators.v_win(x_udp, 64240.0)
    assert v_udp[7] == 0.0 # Zero for UDP
