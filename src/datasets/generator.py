"""
Synthetic Flow Generator for Network Intrusion Detection (CIC-IDS2018 Characteristics).

Simulates both Benign network traffic and 5 distinct Attack families:
1. Volumetric DoS SYN Flood (TCP)
2. UDP Volumetric Flood (UDP)
3. Brute Force Authentication (TCP)
4. Web Application Exploits (SQLi / XSS, TCP)
5. Low-and-Slow Infiltration Beacon (TCP)

Enforces strict ground-truth physical network conservation laws on the flow manifold M_flow.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from .schema import get_cicids_schema, project_network_invariants


def _generate_single_flow(
    proto: int,
    port: int,
    fwd_pkts: float,
    bwd_pkts: float,
    fwd_bytes_per_pkt: float,
    bwd_bytes_per_pkt: float,
    duration_us: float,
    win: float,
    hdr_per_pkt: float,
    schema
) -> np.ndarray:
    """Helper to synthesize a single consistent 12-dimensional flow vector."""
    x = np.zeros(len(schema.feature_names))
    x[0] = float(port)
    x[1] = float(proto)
    x[2] = float(fwd_pkts)
    x[3] = float(bwd_pkts)
    x[4] = float(fwd_pkts * fwd_bytes_per_pkt)
    x[5] = float(bwd_pkts * bwd_bytes_per_pkt)
    x[6] = float(duration_us)
    x[7] = float(win if proto == 6 else 0.0)
    x[11] = float(fwd_pkts * hdr_per_pkt)
    return project_network_invariants(x, schema)


def generate_cicids_traffic(
    n_samples: int = 12000,
    attack_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generates a realistic network flow dataset respecting physical flow dependencies.
    Models 5 intrusion families alongside diverse benign background traffic.
    """
    rng = np.random.RandomState(random_state)
    schema = get_cicids_schema()
    
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    # ---------------- 1. Benign Traffic Generation ----------------
    benign_flows = []
    for _ in range(n_benign):
        is_tcp = (rng.rand() < 0.85) # 85% TCP, 15% UDP (DNS/NTP/QUIC)
        proto = 6 if is_tcp else 17
        
        if is_tcp:
            port = rng.choice([80, 443, 8080, 8443, 22, 21, 5000])
            mode = rng.choice(['interactive', 'download', 'upload', 'persistent'], p=[0.35, 0.30, 0.20, 0.15])
            if mode == 'persistent': # Long-lived persistent session / WebSocket / background sync
                pkts_fwd = rng.randint(6, 30)
                pkts_bwd = rng.randint(5, 25)
                hdr_len = rng.choice([20.0, 32.0], p=[0.6, 0.4])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(40.0, 200.0)
                bytes_bwd = rng.uniform(60.0, 300.0)
                dur = rng.uniform(8000000.0, 45000000.0) # 8s to 45s (persistent session)
                win = rng.choice([16384, 64240])
            elif mode == 'upload': # Asymmetric client upload (HTTP POST, FTP put, telemetry push)
                pkts_fwd = rng.randint(20, 300)
                pkts_bwd = rng.choice([0, 1, 2, 3, 5], p=[0.20, 0.30, 0.25, 0.15, 0.10])
                hdr_len = rng.choice([20.0, 32.0], p=[0.6, 0.4])
                payload_mean = rng.uniform(100.0, 800.0)
                bytes_fwd = hdr_len + 20.0 + payload_mean
                bytes_bwd = 40.0
                dur = rng.uniform(500000.0, 8000000.0) # 0.5s - 8s (normal benign rate)
                win = rng.choice([8192, 16384, 29200, 64240])
            elif mode == 'download': # Downstream data transfer (HTTP GET, file fetch)
                pkts_fwd = rng.randint(5, 50)
                pkts_bwd = rng.randint(20, 400)
                hdr_len = rng.choice([20.0, 32.0], p=[0.6, 0.4])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(30.0, 100.0)
                bytes_bwd = rng.uniform(400.0, 1400.0)
                dur = rng.uniform(200000.0, 6000000.0)
                win = rng.choice([16384, 29200, 64240])
            else: # Interactive browsing / API transaction
                pkts_fwd = rng.randint(10, 80)
                pkts_bwd = rng.randint(10, 120)
                hdr_len = rng.choice([20.0, 32.0], p=[0.6, 0.4])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(50.0, 400.0)
                bytes_bwd = rng.uniform(80.0, 800.0)
                dur = rng.uniform(800000.0, 12000000.0)
                win = rng.choice([8192, 16384, 64240])
        else: # UDP
            port = rng.choice([53, 123, 161, 5353, 443])
            is_stream = (rng.rand() < 0.40)
            if is_stream: # Streaming / QUIC / WebRTC bulk UDP
                pkts_fwd = rng.randint(30, 300)
                pkts_bwd = rng.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
                hdr_len = 8.0
                bytes_fwd = 28.0 + rng.uniform(100.0, 600.0)
                bytes_bwd = 60.0
                dur = rng.uniform(1000000.0, 10000000.0) # 1s to 10s (normal rate)
                win = 0.0
            else: # Transactional DNS/NTP short request/response
                pkts_fwd = rng.choice([1, 2, 4, 8], p=[0.70, 0.15, 0.10, 0.05])
                pkts_bwd = rng.choice([1, 2, 4], p=[0.80, 0.15, 0.05])
                hdr_len = 8.0
                bytes_fwd = 28.0 + rng.uniform(20.0, 120.0)
                bytes_bwd = 28.0 + rng.uniform(40.0, 400.0)
                dur = rng.uniform(500.0, 20000.0)
                win = 0.0
            
        flow = _generate_single_flow(
            proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema
        )
        benign_flows.append(flow)
        
    X_benign = np.array(benign_flows)
    
    # ---------------- 2. Attack Traffic Generation (5 Families) ----------------
    # Family distribution:
    # 1: DoS SYN Flood (30%)
    # 2: UDP Volumetric Flood (25%)
    # 3: Brute Force Authentication (20%)
    # 4: Web Application Exploits (15%)
    # 5: Low-and-Slow Infiltration (10%)
    family_types = rng.choice(
        ['dos_syn', 'udp_flood', 'bruteforce', 'web_attack', 'infiltration'],
        size=n_attack,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    
    attack_flows = []
    for fam in family_types:
        if fam == 'dos_syn':
            # High packet rate, short burst duration, minimal header-only forward packets
            proto = 6
            port = rng.choice([80, 443, 8080])
            pkts_fwd = rng.randint(40, 300)
            pkts_bwd = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            hdr_len = 20.0
            bytes_fwd = 40.0 # SYN packet: 20B IPv4 + 20B TCP
            bytes_bwd = 40.0
            dur = rng.uniform(500.0, 50000.0) # 0.5ms to 50ms (extreme rate)
            win = rng.choice([1024, 2048, 4096, 8192])
            
        elif fam == 'udp_flood':
            # High-rate UDP flood, zero window, 8B UDP transport header
            proto = 17
            port = rng.choice([53, 123, 1900, 389])
            pkts_fwd = rng.randint(50, 400)
            pkts_bwd = rng.choice([0, 1], p=[0.85, 0.15])
            hdr_len = 8.0
            bytes_fwd = 28.0 + rng.uniform(50.0, 500.0)
            bytes_bwd = 60.0
            dur = rng.uniform(1000.0, 80000.0) # 1ms to 80ms
            win = 0.0
            
        elif fam == 'bruteforce':
            # Repetitive credential submissions on SSH/RDP/FTP
            proto = 6
            port = rng.choice([22, 3389, 21])
            pkts_fwd = rng.randint(20, 90)
            pkts_bwd = rng.randint(15, 60)
            hdr_len = rng.choice([20.0, 32.0])
            bytes_fwd = hdr_len + 20.0 + rng.uniform(80.0, 200.0) # fixed-size auth payload
            bytes_bwd = rng.uniform(100.0, 400.0)
            dur = rng.uniform(500000.0, 4000000.0) # 0.5s to 4s
            win = rng.choice([8192, 16384, 29200])
            
        elif fam == 'web_attack':
            # SQL Injection / XSS: large request payload bytes with standard browsing responses
            proto = 6
            port = rng.choice([80, 443])
            pkts_fwd = rng.randint(15, 60)
            pkts_bwd = rng.randint(20, 100)
            hdr_len = 20.0
            bytes_fwd = hdr_len + 20.0 + rng.uniform(500.0, 1200.0) # Exploit strings
            bytes_bwd = rng.uniform(400.0, 1500.0)
            dur = rng.uniform(200000.0, 2500000.0) # 0.2s to 2.5s
            win = rng.choice([16384, 29200, 64240])
            
        else: # infiltration
            # Covert C2 beacon: low packet count, long sleep duration, subtle beacon payload
            proto = 6
            port = rng.choice([443, 8443, 8080])
            pkts_fwd = rng.randint(6, 25)
            pkts_bwd = rng.randint(4, 20)
            hdr_len = rng.choice([20.0, 32.0])
            bytes_fwd = hdr_len + 20.0 + rng.uniform(40.0, 150.0)
            bytes_bwd = rng.uniform(60.0, 300.0)
            dur = rng.uniform(5000000.0, 45000000.0) # 5s to 45s (low-and-slow)
            win = rng.choice([4096, 8192])
            
        flow = _generate_single_flow(
            proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema
        )
        attack_flows.append(flow)
        
    X_attack = np.array(attack_flows)
    
    # Combine and Shuffle
    X = np.vstack([X_benign, X_attack])
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    df = pd.DataFrame(X, columns=schema.feature_names)
    df['label'] = y
    
    return X, y, df


def generate_attack_family_cohorts(
    n_per_family: int = 100,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generates isolated cohorts for each of the 5 attack families (N=100 per family)
    for exact attack family breakdown and ablation evaluations.
    """
    rng = np.random.RandomState(random_state)
    schema = get_cicids_schema()
    cohorts = {}
    
    family_names = [
        'Family 1: Volumetric DoS SYN Flood',
        'Family 2: UDP Volumetric Flood',
        'Family 3: Brute Force Authentication',
        'Family 4: Web Application Exploits',
        'Family 5: Low-and-Slow Infiltration'
    ]
    
    for fam_name in family_names:
        flows = []
        for _ in range(n_per_family):
            if 'Family 1' in fam_name:
                proto, port = 6, rng.choice([80, 443])
                pkts_fwd = rng.randint(40, 250)
                pkts_bwd = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                hdr_len = 20.0
                bytes_fwd = 40.0
                bytes_bwd = 40.0
                dur = rng.uniform(500.0, 50000.0)
                win = rng.choice([1024, 2048, 4096])
            elif 'Family 2' in fam_name:
                proto, port = 17, rng.choice([53, 123, 1900])
                pkts_fwd = rng.randint(50, 300)
                pkts_bwd = rng.choice([0, 1], p=[0.85, 0.15])
                hdr_len = 8.0
                bytes_fwd = 28.0 + rng.uniform(60.0, 400.0)
                bytes_bwd = 60.0
                dur = rng.uniform(1000.0, 70000.0)
                win = 0.0
            elif 'Family 3' in fam_name:
                proto, port = 6, rng.choice([22, 3389])
                pkts_fwd = rng.randint(20, 80)
                pkts_bwd = rng.randint(15, 50)
                hdr_len = rng.choice([20.0, 32.0])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(90.0, 180.0)
                bytes_bwd = rng.uniform(120.0, 350.0)
                dur = rng.uniform(600000.0, 3500000.0)
                win = rng.choice([8192, 16384])
            elif 'Family 4' in fam_name:
                proto, port = 6, rng.choice([80, 443])
                pkts_fwd = rng.randint(15, 50)
                pkts_bwd = rng.randint(20, 80)
                hdr_len = 20.0
                bytes_fwd = hdr_len + 20.0 + rng.uniform(600.0, 1100.0)
                bytes_bwd = rng.uniform(500.0, 1400.0)
                dur = rng.uniform(300000.0, 2000000.0)
                win = rng.choice([16384, 29200])
            else: # Family 5
                proto, port = 6, rng.choice([443, 8443])
                pkts_fwd = rng.randint(6, 20)
                pkts_bwd = rng.randint(4, 15)
                hdr_len = rng.choice([20.0, 32.0])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(40.0, 120.0)
                bytes_bwd = rng.uniform(60.0, 250.0)
                dur = rng.uniform(8000000.0, 40000000.0)
                win = rng.choice([4096, 8192])
                
            flow = _generate_single_flow(
                proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema
            )
            flows.append(flow)
        cohorts[fam_name] = np.array(flows)
        
    return cohorts


def generate_unsw_traffic(
    n_samples: int = 12000,
    attack_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generates synthetic network telemetry calibrated to UNSW-NB15 empirical distributions:
    Blends modern hybrid normal activities with subtler modern low-footprint attack categories
    (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).
    """
    rng = np.random.RandomState(random_state)
    schema = get_cicids_schema()
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    # Benign: hybrid ACCS testbed profile
    benign_flows = []
    for _ in range(n_benign):
        is_tcp = (rng.rand() < 0.80)
        proto = 6 if is_tcp else 17
        port = rng.choice([80, 443, 21, 22, 25, 53, 110, 143])
        if is_tcp:
            pkts_fwd = rng.randint(4, 150)
            pkts_bwd = rng.randint(3, 180)
            hdr_len = rng.choice([20.0, 32.0])
            bytes_fwd = hdr_len + 20.0 + rng.uniform(40.0, 600.0)
            bytes_bwd = rng.uniform(50.0, 1200.0)
            dur = rng.uniform(50000.0, 15000000.0)
            win = rng.choice([8192, 16384, 29200, 64240])
        else:
            pkts_fwd = rng.randint(1, 20)
            pkts_bwd = rng.randint(1, 20)
            hdr_len = 8.0
            bytes_fwd = 28.0 + rng.uniform(20.0, 250.0)
            bytes_bwd = 28.0 + rng.uniform(40.0, 400.0)
            dur = rng.uniform(1000.0, 50000.0)
            win = 0.0
        benign_flows.append(_generate_single_flow(proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema))
        
    # Attack: subtler signatures (Analysis, Fuzzers, Exploits)
    attack_flows = []
    for _ in range(n_attack):
        is_tcp = (rng.rand() < 0.85)
        proto = 6 if is_tcp else 17
        port = rng.choice([80, 443, 22, 21, 53])
        if is_tcp:
            pkts_fwd = rng.randint(6, 80)
            pkts_bwd = rng.randint(4, 60)
            hdr_len = 20.0
            bytes_fwd = hdr_len + 20.0 + rng.uniform(150.0, 800.0)
            bytes_bwd = rng.uniform(60.0, 500.0)
            dur = rng.uniform(20000.0, 3000000.0)
            win = rng.choice([1024, 2048, 4096, 8192])
        else:
            pkts_fwd = rng.randint(10, 60)
            pkts_bwd = rng.choice([0, 1])
            hdr_len = 8.0
            bytes_fwd = 28.0 + rng.uniform(80.0, 400.0)
            bytes_bwd = 60.0
            dur = rng.uniform(5000.0, 80000.0)
            win = 0.0
        attack_flows.append(_generate_single_flow(proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema))
        
    X = np.vstack([np.array(benign_flows), np.array(attack_flows)])
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    df = pd.DataFrame(X, columns=schema.feature_names)
    df['label'] = y
    return X, y, df


def generate_cidds_traffic(
    n_samples: int = 12000,
    attack_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generates synthetic network telemetry calibrated to CIDDS-001 OpenStack cloud distributions:
    Models OpenStack tenant virtual machines and perimeter biflow collector telemetry.
    """
    rng = np.random.RandomState(random_state)
    schema = get_cicids_schema()
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    # Benign cloud tenant traffic (interactive VMs + async monitoring pushes)
    benign_flows = []
    for _ in range(n_benign):
        is_tcp = (rng.rand() < 0.90)
        proto = 6 if is_tcp else 17
        port = rng.choice([80, 443, 8080, 22, 3306, 5432])
        if is_tcp:
            mode = rng.choice(['bidirectional', 'telemetry_push'], p=[0.75, 0.25])
            if mode == 'bidirectional':
                pkts_fwd = rng.randint(5, 180)
                pkts_bwd = rng.randint(5, 200)
                hdr_len = rng.choice([20.0, 32.0])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(60.0, 600.0)
                bytes_bwd = rng.uniform(80.0, 1000.0)
                dur = rng.uniform(100000.0, 15000000.0)
                win = rng.choice([16384, 29200, 64240])
            else: # Asynchronous tenant telemetry/log push
                pkts_fwd = rng.randint(10, 120)
                pkts_bwd = rng.choice([0, 1, 2, 4], p=[0.50, 0.25, 0.15, 0.10])
                hdr_len = rng.choice([20.0, 32.0])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(50.0, 350.0)
                bytes_bwd = 40.0
                dur = rng.uniform(200000.0, 8000000.0)
                win = rng.choice([8192, 16384, 64240])
        else:
            is_push = (rng.rand() < 0.35)
            if is_push:
                pkts_fwd = rng.randint(5, 50)
                pkts_bwd = rng.choice([0, 1], p=[0.8, 0.2])
                hdr_len = 8.0
                bytes_fwd = 28.0 + rng.uniform(40.0, 250.0)
                bytes_bwd = 28.0
                dur = rng.uniform(10000.0, 500000.0)
                win = 0.0
            else:
                pkts_fwd = rng.randint(1, 15)
                pkts_bwd = rng.randint(1, 15)
                hdr_len = 8.0
                bytes_fwd = 28.0 + rng.uniform(30.0, 200.0)
                bytes_bwd = 28.0 + rng.uniform(50.0, 300.0)
                dur = rng.uniform(2000.0, 30000.0)
                win = 0.0
        benign_flows.append(_generate_single_flow(proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema))
        
    # Attack: cloud volumetric bursts, brute force, and port probes
    attack_flows = []
    for _ in range(n_attack):
        is_tcp = (rng.rand() < 0.85)
        proto = 6 if is_tcp else 17
        port = rng.choice([80, 443, 22, 3306])
        if is_tcp:
            atk_type = rng.choice(['burst', 'brute'], p=[0.60, 0.40])
            if atk_type == 'burst':
                pkts_fwd = rng.randint(30, 250)
                pkts_bwd = rng.choice([0, 1, 2])
                hdr_len = 20.0
                bytes_fwd = 40.0 + rng.uniform(0.0, 80.0)
                bytes_bwd = 40.0
                dur = rng.uniform(2000.0, 100000.0)
                win = rng.choice([1024, 2048, 4096])
            else: # Cloud brute force on SSH / DB port (receives auth prompt/response packets)
                pkts_fwd = rng.randint(15, 60)
                pkts_bwd = rng.randint(8, 35)
                hdr_len = rng.choice([20.0, 32.0])
                bytes_fwd = hdr_len + 20.0 + rng.uniform(80.0, 250.0)
                bytes_bwd = rng.uniform(100.0, 400.0)
                dur = rng.uniform(40000.0, 800000.0)
                win = rng.choice([2048, 4096, 8192])
        else:
            pkts_fwd = rng.randint(40, 200)
            pkts_bwd = rng.choice([0, 1], p=[0.85, 0.15])
            hdr_len = 8.0
            bytes_fwd = 28.0 + rng.uniform(50.0, 300.0)
            bytes_bwd = 28.0
            dur = rng.uniform(1000.0, 50000.0)
            win = 0.0
        attack_flows.append(_generate_single_flow(proto, port, pkts_fwd, pkts_bwd, bytes_fwd, bytes_bwd, dur, win, hdr_len, schema))
        
    X = np.vstack([np.array(benign_flows), np.array(attack_flows)])
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    df = pd.DataFrame(X, columns=schema.feature_names)
    df['label'] = y
    return X, y, df
