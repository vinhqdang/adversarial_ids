"""
Synthetic Flow Generator for Network Intrusion Detection (CIC-IDS2018 Characteristics).

Simulates both Benign network traffic and Attack flows (e.g., DoS/DDoS, Brute Force)
with realistic flow properties and ground-truth physical constraints.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from .schema import get_cicids_schema, project_network_invariants


def generate_cicids_traffic(n_samples: int = 10000, attack_ratio: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generates realistic network flow dataset respecting physical flow dependencies.
    
    Produces:
    - X: Feature matrix (N, 12)
    - y: Labels (0 = Benign, 1 = Attack)
    - df: Pandas DataFrame with named features and label
    """
    rng = np.random.RandomState(random_state)
    schema = get_cicids_schema()
    
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    # ---------------- Benign Traffic Generation ----------------
    # Port 80, 443, 8080 or random ephemeral ports
    benign_ports = rng.choice([80, 443, 8080, 53, 21], size=n_benign, p=[0.4, 0.45, 0.05, 0.08, 0.02])
    benign_proto = rng.choice([6, 17], size=n_benign, p=[0.9, 0.1])
    
    # Forward packets: log-normal distribution (human web browsing / API calls)
    benign_fwd_pkts = np.clip(np.round(rng.lognormal(mean=2.0, sigma=0.8, size=n_benign)), 2, 500)
    # Backward packets: server response packets
    benign_bwd_pkts = np.clip(np.round(benign_fwd_pkts * rng.uniform(0.8, 2.5, size=n_benign)), 1, 1000)
    
    # Forward bytes: mix of headers + small/medium payloads
    benign_fwd_bytes = benign_fwd_pkts * rng.uniform(40, 600, size=n_benign)
    benign_bwd_bytes = benign_bwd_pkts * rng.uniform(100, 1400, size=n_benign)
    
    # Duration: benign flows last from a few ms to several seconds
    benign_duration = rng.exponential(scale=1.5e6, size=n_benign) + 5000.0 # microseconds
    
    # TCP window sizes: standard client OS values
    benign_win = rng.choice([8192, 16384, 29200, 64240, 65535], size=n_benign)
    
    # ---------------- Attack Traffic Generation (DoS / Brute Force) ----------------
    # Target high-value ports (e.g., 80, 22, 443)
    attack_ports = rng.choice([80, 22, 443, 8080], size=n_attack, p=[0.5, 0.25, 0.15, 0.1])
    attack_proto = np.full(n_attack, 6) # Almost all DoS/BruteForce are TCP
    
    # Attacks exhibit characteristic patterns:
    # E.g., DoS: High packet rate, short duration bursts, small fixed payloads
    # Brute Force: High number of attempts, deterministic payload size
    attack_type = rng.choice(['dos_burst', 'bruteforce'], size=n_attack, p=[0.7, 0.3])
    
    attack_fwd_pkts = np.zeros(n_attack)
    attack_bwd_pkts = np.zeros(n_attack)
    attack_fwd_bytes = np.zeros(n_attack)
    attack_bwd_bytes = np.zeros(n_attack)
    attack_duration = np.zeros(n_attack)
    attack_win = np.zeros(n_attack)
    
    for i in range(n_attack):
        if attack_type[i] == 'dos_burst':
            # Rapid SYN flood or UDP flood: very high packets, minimal bytes, short duration
            pkts = rng.randint(20, 300)
            attack_fwd_pkts[i] = pkts
            attack_bwd_pkts[i] = rng.choice([0, 1, 2]) # Server struggles or drops
            attack_fwd_bytes[i] = pkts * rng.uniform(40, 70) # SYN packets (header only)
            attack_bwd_bytes[i] = attack_bwd_pkts[i] * 40
            attack_duration[i] = rng.uniform(100.0, 50000.0) # Burst under 50ms
            attack_win[i] = rng.choice([1024, 2048, 4096]) # Botnet default windows
        else: # bruteforce
            pkts = rng.randint(15, 80)
            attack_fwd_pkts[i] = pkts
            attack_bwd_pkts[i] = rng.randint(10, 50)
            attack_fwd_bytes[i] = pkts * rng.uniform(80, 220)
            attack_bwd_bytes[i] = attack_bwd_pkts[i] * rng.uniform(100, 300)
            attack_duration[i] = rng.uniform(200000.0, 3000000.0)
            attack_win[i] = 8192
            
    # Combine Base Matrices
    ports = np.concatenate([benign_ports, attack_ports])
    protos = np.concatenate([benign_proto, attack_proto])
    fwd_pkts = np.concatenate([benign_fwd_pkts, attack_fwd_pkts])
    bwd_pkts = np.concatenate([benign_bwd_pkts, attack_bwd_pkts])
    fwd_bytes = np.concatenate([benign_fwd_bytes, attack_fwd_bytes])
    bwd_bytes = np.concatenate([benign_bwd_bytes, attack_bwd_bytes])
    durations = np.concatenate([benign_duration, attack_duration])
    windows = np.concatenate([benign_win, attack_win])
    
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    
    # Construct preliminary (N, 12) matrix and enforce mathematical flow coupling invariants
    X = np.zeros((n_samples, len(schema.feature_names)))
    X[:, 0] = ports
    X[:, 1] = protos
    X[:, 2] = fwd_pkts
    X[:, 3] = bwd_pkts
    X[:, 4] = fwd_bytes
    X[:, 5] = bwd_bytes
    X[:, 6] = durations
    X[:, 7] = windows
    
    # Project coupled features for every flow
    for i in range(n_samples):
        X[i] = project_network_invariants(X[i], schema)
        
    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    df = pd.DataFrame(X, columns=schema.feature_names)
    df['label'] = y
    
    return X, y, df
