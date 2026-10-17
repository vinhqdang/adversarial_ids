"""
Real Network Intrusion Detection Dataset Loader and Cache.
Fetches and standardizes real benchmark datasets:
1. CSE-CIC-IDS2018 (official AWS S3 repository)
2. UNSW-NB15 (hybrid network testbed)
3. CIC-IDS2017 (CICFlowMeter benchmark)

Maps all datasets into the standardized 12-feature problem-space invariant schema.
"""

import os
import io
import time
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

from .schema import get_cicids_schema, project_network_invariants

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_and_build_real_cicids2018(random_state: int = 42) -> str:
    """Downloads real flows from CSE-CIC-IDS2018 AWS S3 bucket and builds standard benchmark."""
    _ensure_data_dir()
    cache_path = os.path.join(DATA_DIR, 'cse_cic_ids2018.csv')
    if os.path.exists(cache_path):
        print(f"[Dataset] Found cached real CSE-CIC-IDS2018: {cache_path}")
        return cache_path

    print("[Dataset] Fetching real CSE-CIC-IDS2018 flows from AWS S3...")
    base_url = 'https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/'
    
    cols_map = {
        'Dst Port': 'dst_port',
        'Protocol': 'protocol',
        'Tot Fwd Pkts': 'tot_fwd_pkts',
        'Tot Bwd Pkts': 'tot_bwd_pkts',
        'TotLen Fwd Pkts': 'totlen_fwd_pkts',
        'TotLen Bwd Pkts': 'totlen_bwd_pkts',
        'Flow Duration': 'flow_duration',
        'Init Fwd Win Byts': 'init_fwd_win_byts',
        'Flow Byts/s': 'flow_bytes_s',
        'Flow Pkts/s': 'flow_pkts_s',
        'Fwd Pkt Len Mean': 'fwd_pkt_len_mean',
        'Fwd Header Len': 'fwd_header_len',
        'Label': 'label_orig'
    }

    sources = [
        ('DoS', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 0, 7000000),
        ('DDoS', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', 0, 7000000),
        ('BruteForce', 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 0, 7000000),
        ('WebExploit', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', 0, 16000000),
        ('WebExploit', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 0, 10000000),
        ('InfiltrationBeacon', 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv', 20000000, 28000000)
    ]

    attack_dfs = []
    benign_dfs = []

    for fam_name, fname, start, end in sources:
        print(f"  Downloading chunk from {fname}...")
        if start > 0:
            rh = requests.get(base_url + fname, headers={'Range': 'bytes=0-2000'}, timeout=20)
            header = rh.text.split('\n')[0]
            r = requests.get(base_url + fname, headers={'Range': f'bytes={start}-{end}'}, timeout=30)
            first_nl = r.text.find('\n')
            last_nl = r.text.rfind('\n')
            csv_text = header + '\n' + r.text[first_nl+1:last_nl]
        else:
            r = requests.get(base_url + fname, headers={'Range': f'bytes={start}-{end}'}, timeout=30)
            last_nl = r.text.rfind('\n')
            csv_text = r.text[:last_nl]

        df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
        # Rename and cast
        keep_cols = [c for c in cols_map.keys() if c in df.columns]
        df = df[keep_cols].rename(columns=cols_map)
        
        # Clean numeric
        for c in ['dst_port', 'protocol', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts',
                  'totlen_bwd_pkts', 'flow_duration', 'init_fwd_win_byts', 'flow_bytes_s',
                  'flow_pkts_s', 'fwd_pkt_len_mean', 'fwd_header_len']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna()
        df = df[(df['flow_duration'] > 0) & (df['tot_fwd_pkts'] > 0)]
        df = df[df['protocol'].isin([6, 17])]  # TCP or UDP

        # Split attack vs benign
        benign_part = df[df['label_orig'] == 'Benign'].copy()
        attack_part = df[df['label_orig'] != 'Benign'].copy()
        
        benign_dfs.append(benign_part)
        attack_part['attack_family'] = fam_name
        attack_dfs.append(attack_part)

    # Balance exactly 600 per attack family = 3,000 attacks
    rng = np.random.RandomState(random_state)
    all_attacks = pd.concat(attack_dfs, ignore_index=True)
    sampled_attacks = []
    for fam_name in ['DoS', 'DDoS', 'BruteForce', 'WebExploit', 'InfiltrationBeacon']:
        fam_df = all_attacks[all_attacks['attack_family'] == fam_name]
        n_available = len(fam_df)
        n_take = min(600, n_available)
        idx = rng.choice(n_available, size=n_take, replace=False)
        sampled_attacks.append(fam_df.iloc[idx])
        print(f"  Extracted {n_take} real flows for family {fam_name}")

    attacks_combined = pd.concat(sampled_attacks, ignore_index=True)
    attacks_combined['label'] = 1

    # Sample 9,000 benign
    benign_combined = pd.concat(benign_dfs, ignore_index=True)
    idx_benign = rng.choice(len(benign_combined), size=9000, replace=False)
    benign_sampled = benign_combined.iloc[idx_benign].copy()
    benign_sampled['label'] = 0
    benign_sampled['attack_family'] = 'Benign'

    full_df = pd.concat([benign_sampled, attacks_combined], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    full_df.to_csv(cache_path, index=False)
    print(f"[Dataset] Saved real CSE-CIC-IDS2018 dataset ({len(full_df)} rows) to: {cache_path}")
    return cache_path


def download_and_build_real_unsw_nb15(random_state: int = 42) -> str:
    """Downloads real UNSW-NB15 flows and standardizes to 12-feature schema."""
    _ensure_data_dir()
    cache_path = os.path.join(DATA_DIR, 'unsw_nb15.csv')
    if os.path.exists(cache_path):
        print(f"[Dataset] Found cached real UNSW-NB15: {cache_path}")
        return cache_path

    print("[Dataset] Fetching real UNSW-NB15 parquet from Hugging Face...")
    url = 'https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/test.parquet'
    df = pd.read_parquet(url)

    rng = np.random.RandomState(random_state)
    
    # Map UNSW-NB15 to 12-feature schema
    # dur in seconds -> microseconds
    # proto -> 6 for tcp, 17 for udp
    proto_map = {'tcp': 6.0, 'udp': 17.0}
    proto_col = df['proto'].str.lower().map(proto_map).fillna(6.0).values
    
    # Port mapping: based on service or typical ports
    service_ports = {
        'http': 80.0, 'ftp': 21.0, 'smtp': 25.0, 'ssh': 22.0,
        'dns': 53.0, 'ftp-data': 20.0, 'irc': 6667.0, 'pop3': 110.0,
        'ssl': 443.0, 'snmp': 161.0, 'dhcp': 67.0, 'radius': 1812.0
    }
    ports = df['service'].str.lower().map(service_ports).fillna(443.0).values
    
    tot_fwd_pkts = np.maximum(1.0, df['spkts'].values.astype(float))
    tot_bwd_pkts = np.maximum(0.0, df['dpkts'].values.astype(float))
    totlen_fwd = np.maximum(0.0, df['sbytes'].values.astype(float))
    totlen_bwd = np.maximum(0.0, df['dbytes'].values.astype(float))
    dur_us = np.maximum(10.0, df['dur'].values.astype(float) * 1e6)
    win = np.clip(df['swin'].values.astype(float), 0.0, 65535.0)
    win = np.where(proto_col == 6.0, win, 0.0)
    
    # Coupled rates
    dur_sec = dur_us / 1e6
    bytes_s = (totlen_fwd + totlen_bwd) / dur_sec
    pkts_s = (tot_fwd_pkts + tot_bwd_pkts) / dur_sec
    mean_len = totlen_fwd / tot_fwd_pkts
    hdr_len = np.where(proto_col == 6.0, tot_fwd_pkts * 32.0, tot_fwd_pkts * 8.0)

    unsw_clean = pd.DataFrame({
        'dst_port': ports,
        'protocol': proto_col,
        'tot_fwd_pkts': tot_fwd_pkts,
        'tot_bwd_pkts': tot_bwd_pkts,
        'totlen_fwd_pkts': totlen_fwd,
        'totlen_bwd_pkts': totlen_bwd,
        'flow_duration': dur_us,
        'init_fwd_win_byts': win,
        'flow_bytes_s': bytes_s,
        'flow_pkts_s': pkts_s,
        'fwd_pkt_len_mean': mean_len,
        'fwd_header_len': hdr_len,
        'label': df['label'].values.astype(int),
        'attack_family': df['attack_cat'].values
    })

    # Filter out any infs
    unsw_clean = unsw_clean.replace([np.inf, -np.inf], np.nan).dropna()

    # Balance 9,000 benign and 3,000 attacks
    benign = unsw_clean[unsw_clean['label'] == 0]
    attacks = unsw_clean[unsw_clean['label'] == 1]

    idx_b = rng.choice(len(benign), size=min(9000, len(benign)), replace=False)
    idx_a = rng.choice(len(attacks), size=min(3000, len(attacks)), replace=False)

    final_df = pd.concat([benign.iloc[idx_b], attacks.iloc[idx_a]], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    final_df.to_csv(cache_path, index=False)
    print(f"[Dataset] Saved real UNSW-NB15 dataset ({len(final_df)} rows) to: {cache_path}")
    return cache_path


def download_and_build_real_cicids2017(random_state: int = 42) -> str:
    """Downloads real CIC-IDS2017 flows from Hugging Face and standardizes to schema."""
    _ensure_data_dir()
    cache_path = os.path.join(DATA_DIR, 'cic_ids2017.csv')
    if os.path.exists(cache_path):
        print(f"[Dataset] Found cached real CIC-IDS2017: {cache_path}")
        return cache_path

    print("[Dataset] Fetching real CIC-IDS2017 flows from Hugging Face...")
    cols_map = {
        'Dst Port': 'dst_port',
        'Protocol': 'protocol',
        'Total Fwd Packet': 'tot_fwd_pkts',
        'Total Bwd packets': 'tot_bwd_pkts',
        'Total Length of Fwd Packet': 'totlen_fwd_pkts',
        'Total Length of Bwd Packet': 'totlen_bwd_pkts',
        'Flow Duration': 'flow_duration',
        'Init Fwd Win Byts': 'init_fwd_win_byts',
        'Flow Bytes/s': 'flow_bytes_s',
        'Flow Packets/s': 'flow_pkts_s',
        'Fwd Packet Length Mean': 'fwd_pkt_len_mean',
        'Fwd Header Length': 'fwd_header_len',
        'Label': 'label_orig'
    }

    # Wednesday has DoS attacks (Slowloris, SlowHTTPTest, Hulk, GoldenEye)
    # Friday has PortScan, DDoS, Botnet
    urls = [
        ('https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/wednesday.csv', 0, 8000000),
        ('https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv', 20000000, 28000000)
    ]

    dfs = []
    for url, start, end in urls:
        if start > 0:
            rh = requests.get(url, headers={'Range': 'bytes=0-2000'}, allow_redirects=True, timeout=20)
            header = rh.text.split('\n')[0]
            r = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, allow_redirects=True, timeout=30)
            first_nl = r.text.find('\n')
            last_nl = r.text.rfind('\n')
            csv_text = header + '\n' + r.text[first_nl+1:last_nl]
        else:
            r = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, allow_redirects=True, timeout=30)
            last_nl = r.text.rfind('\n')
            csv_text = r.text[:last_nl]

        df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
        keep = [c for c in cols_map.keys() if c in df.columns]
        df = df[keep].rename(columns=cols_map)
        dfs.append(df)

    comb = pd.concat(dfs, ignore_index=True)
    
    # If init_fwd_win_byts is missing in some CICIDS2017 CSV headers, default to 29200
    if 'init_fwd_win_byts' not in comb.columns:
        comb['init_fwd_win_byts'] = 29200.0

    for c in ['dst_port', 'protocol', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts',
              'totlen_bwd_pkts', 'flow_duration', 'init_fwd_win_byts', 'flow_bytes_s',
              'flow_pkts_s', 'fwd_pkt_len_mean', 'fwd_header_len']:
        comb[c] = pd.to_numeric(comb[c], errors='coerce')

    comb = comb.dropna()
    comb = comb[(comb['flow_duration'] > 0) & (comb['tot_fwd_pkts'] > 0)]
    comb = comb[comb['protocol'].isin([6, 17])]

    benign = comb[comb['label_orig'].str.upper() == 'BENIGN'].copy()
    attacks = comb[comb['label_orig'].str.upper() != 'BENIGN'].copy()

    rng = np.random.RandomState(random_state)
    idx_b = rng.choice(len(benign), size=min(9000, len(benign)), replace=False)
    idx_a = rng.choice(len(attacks), size=min(3000, len(attacks)), replace=False)

    benign_s = benign.iloc[idx_b].copy()
    benign_s['label'] = 0
    benign_s['attack_family'] = 'Benign'

    attacks_s = attacks.iloc[idx_a].copy()
    attacks_s['label'] = 1
    attacks_s['attack_family'] = attacks_s['label_orig']

    final_df = pd.concat([benign_s, attacks_s], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    final_df.to_csv(cache_path, index=False)
    print(f"[Dataset] Saved real CIC-IDS2017 dataset ({len(final_df)} rows) to: {cache_path}")
    return cache_path


def load_real_cicids_traffic(n_samples: int = 12000, attack_ratio: float = 0.25, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads standardized real CSE-CIC-IDS2018 flows."""
    path = download_and_build_real_cicids2018(random_state=random_state)
    df = pd.read_csv(path)
    schema = get_cicids_schema()
    
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    b_df = df[df['label'] == 0]
    a_df = df[df['label'] == 1]
    
    rng = np.random.RandomState(random_state)
    idx_b = rng.choice(len(b_df), size=min(n_benign, len(b_df)), replace=False)
    idx_a = rng.choice(len(a_df), size=min(n_attack, len(a_df)), replace=False)
    
    selected = pd.concat([b_df.iloc[idx_b], a_df.iloc[idx_a]], ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    X = selected[schema.feature_names].values.astype(float)
    X = np.array([project_network_invariants(row, schema) for row in X])
    y = selected['label'].values.astype(int)
    return X, y, selected


def load_real_attack_family_cohorts(n_per_family: int = 100, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Loads real malicious flows broken down into 5 standard attack families."""
    path = download_and_build_real_cicids2018(random_state=random_state)
    df = pd.read_csv(path)
    schema = get_cicids_schema()
    
    families = ['DoS', 'DDoS', 'BruteForce', 'WebExploit', 'InfiltrationBeacon']
    cohorts = {}
    rng = np.random.RandomState(random_state)
    
    for fam in families:
        fam_df = df[df['attack_family'] == fam]
        n_avail = len(fam_df)
        n_take = min(n_per_family, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        raw_cohort = fam_df.iloc[idx][schema.feature_names].values.astype(float)
        cohorts[fam] = np.array([project_network_invariants(row, schema) for row in raw_cohort])
        
    return cohorts


def load_real_unsw_traffic(n_samples: int = 12000, attack_ratio: float = 0.25, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads standardized real UNSW-NB15 flows."""
    path = download_and_build_real_unsw_nb15(random_state=random_state)
    df = pd.read_csv(path)
    schema = get_cicids_schema()
    
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    b_df = df[df['label'] == 0]
    a_df = df[df['label'] == 1]
    
    rng = np.random.RandomState(random_state)
    idx_b = rng.choice(len(b_df), size=min(n_benign, len(b_df)), replace=False)
    idx_a = rng.choice(len(a_df), size=min(n_attack, len(a_df)), replace=False)
    
    selected = pd.concat([b_df.iloc[idx_b], a_df.iloc[idx_a]], ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    X = selected[schema.feature_names].values.astype(float)
    X = np.array([project_network_invariants(row, schema) for row in X])
    y = selected['label'].values.astype(int)
    return X, y, selected


def load_real_cicids2017_traffic(n_samples: int = 12000, attack_ratio: float = 0.25, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads standardized real CIC-IDS2017 flows."""
    path = download_and_build_real_cicids2017(random_state=random_state)
    df = pd.read_csv(path)
    schema = get_cicids_schema()
    
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack
    
    b_df = df[df['label'] == 0]
    a_df = df[df['label'] == 1]
    
    rng = np.random.RandomState(random_state)
    idx_b = rng.choice(len(b_df), size=min(n_benign, len(b_df)), replace=False)
    idx_a = rng.choice(len(a_df), size=min(n_attack, len(a_df)), replace=False)
    
    selected = pd.concat([b_df.iloc[idx_b], a_df.iloc[idx_a]], ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    X = selected[schema.feature_names].values.astype(float)
    X = np.array([project_network_invariants(row, schema) for row in X])
    y = selected['label'].values.astype(int)
    return X, y, selected
