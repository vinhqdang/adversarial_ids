"""
Full-scale builders for CSE-CIC-IDS2018, CIC-IDS2017 and UNSW-NB15.

Unlike the legacy 12,000-flow samplers, these builders read every raw capture
file under ``data/raw`` (see ``download_raw``), map the flows onto the
12-feature problem-space schema with vectorised operations, drop exact
duplicate records (so train and test never share a flow), and cache the result
as Parquet. Nothing is subsampled and no value is imputed with a constant.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from .schema import get_cicids_schema

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
CACHE_DIR = os.path.join(ROOT, 'data', 'full')

FEATURES: List[str] = get_cicids_schema().feature_names

# Per-dataset header aliases -> schema feature names.
ALIASES: Dict[str, Dict[str, str]] = {
    'cse2018': {
        'Dst Port': 'dst_port', 'Protocol': 'protocol',
        'Tot Fwd Pkts': 'tot_fwd_pkts', 'Tot Bwd Pkts': 'tot_bwd_pkts',
        'TotLen Fwd Pkts': 'totlen_fwd_pkts', 'TotLen Bwd Pkts': 'totlen_bwd_pkts',
        'Flow Duration': 'flow_duration', 'Init Fwd Win Byts': 'init_fwd_win_byts',
        'Flow Byts/s': 'flow_bytes_s', 'Flow Pkts/s': 'flow_pkts_s',
        'Fwd Pkt Len Mean': 'fwd_pkt_len_mean', 'Fwd Header Len': 'fwd_header_len',
        'Label': 'label_orig',
    },
    'cic2017': {
        'Dst Port': 'dst_port', 'Protocol': 'protocol',
        'Total Fwd Packet': 'tot_fwd_pkts', 'Total Bwd packets': 'tot_bwd_pkts',
        'Total Length of Fwd Packet': 'totlen_fwd_pkts',
        'Total Length of Bwd Packet': 'totlen_bwd_pkts',
        'Flow Duration': 'flow_duration', 'FWD Init Win Bytes': 'init_fwd_win_byts',
        'Init Fwd Win Byts': 'init_fwd_win_byts', 'Init_Win_bytes_forward': 'init_fwd_win_byts',
        'Flow Bytes/s': 'flow_bytes_s', 'Flow Packets/s': 'flow_pkts_s',
        'Fwd Packet Length Mean': 'fwd_pkt_len_mean',
        'Fwd Header Length': 'fwd_header_len', 'Label': 'label_orig',
    },
}

RAW_URLS = {
    'cse2018': ('https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'
                'Processed%20Traffic%20Data%20for%20ML%20Algorithms/{name}_TrafficForML_CICFlowMeter.csv',
                ['Wednesday-14-02-2018', 'Thursday-15-02-2018', 'Friday-16-02-2018',
                 'Wednesday-21-02-2018', 'Thursday-22-02-2018', 'Friday-23-02-2018',
                 'Wednesday-28-02-2018', 'Thursday-01-03-2018', 'Friday-02-03-2018']),
    'cic2017': ('https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/{name}.csv',
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']),
}


def download_raw(dataset: str) -> None:
    """Stream every raw capture file of a dataset into ``data/raw`` (resumable)."""
    template, names = RAW_URLS[dataset]
    out_dir = os.path.join(RAW_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        dest = os.path.join(out_dir, f'{name}.csv')
        url = template.format(name=name)
        expected = int(requests.head(url, allow_redirects=True, timeout=60).headers['Content-Length'])
        if os.path.exists(dest) and os.path.getsize(dest) == expected:
            continue
        with requests.get(url, stream=True, timeout=120) as r, open(dest, 'wb') as fh:
            r.raise_for_status()
            for block in r.iter_content(1 << 20):
                fh.write(block)


def _numeric_frame(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    keep = {src: dst for src, dst in aliases.items() if src in df.columns}
    out = df[list(keep)].rename(columns=keep)
    # Two aliases can collapse onto one name; keep the first occurrence.
    out = out.loc[:, ~out.columns.duplicated()]
    missing = [c for c in FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f'raw file lacks required features: {missing}')
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df = df[(df['flow_duration'] > 0) & (df['tot_fwd_pkts'] > 0)]
    df = df[df['protocol'].isin([6, 17])]
    df = df[(df['init_fwd_win_byts'] >= 0) & (df['flow_bytes_s'] >= 0)]
    return df


def _family_2018(label: str) -> str:
    low = label.lower()
    if low == 'benign':
        return 'Benign'
    if 'ddos' in low:
        return 'DDoS'
    if 'dos' in low:
        return 'DoS'
    if 'sql' in low or 'xss' in low or 'web' in low:
        return 'WebExploit'
    if 'brute' in low:
        return 'BruteForce'
    if 'infil' in low:
        return 'InfiltrationBeacon'
    if 'bot' in low:
        return 'Botnet'
    return label


def _family_2017(label: str) -> str:
    low = label.lower()
    if low == 'benign':
        return 'Benign'
    if 'ddos' in low:
        return 'DDoS'
    if 'dos' in low:
        return 'DoS'
    if 'web' in low or 'sql' in low or 'xss' in low:
        return 'WebExploit'
    if 'patator' in low or 'brute' in low:
        return 'BruteForce'
    if 'infil' in low:
        return 'InfiltrationBeacon'
    if 'bot' in low:
        return 'Botnet'
    if 'portscan' in low:
        return 'PortScan'
    return label


def _build_csv_dataset(dataset: str, family_fn) -> pd.DataFrame:
    folder = os.path.join(RAW_DIR, dataset)
    files = sorted(f for f in os.listdir(folder) if f.endswith('.csv'))
    if not files:
        raise FileNotFoundError(f'no raw files in {folder}; run download_raw({dataset!r})')
    parts = []
    for fname in files:
        raw = pd.read_csv(os.path.join(folder, fname), low_memory=False)
        raw.columns = [c.strip() for c in raw.columns]
        frame = _numeric_frame(raw, ALIASES[dataset])
        frame['label_orig'] = raw['Label'].astype(str).str.strip().values
        frame['source'] = fname.replace('.csv', '')
        frame['row_order'] = np.arange(len(frame))
        parts.append(_clean(frame))
        print(f'  [{dataset}] {fname}: {len(raw):,} raw -> {len(parts[-1]):,} usable')
    return pd.concat(parts, ignore_index=True)


def _build_unsw() -> pd.DataFrame:
    folder = os.path.join(RAW_DIR, 'unsw')
    parts = []
    for split in ('train', 'test'):
        path = os.path.join(folder, f'{split}.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path).assign(source=split))
    df = pd.concat(parts, ignore_index=True)
    proto = df['proto'].str.lower().map({'tcp': 6.0, 'udp': 17.0})
    df = df[proto.notna()].copy()
    proto = proto[df.index]
    service_ports = {'http': 80.0, 'ftp': 21.0, 'smtp': 25.0, 'ssh': 22.0, 'dns': 53.0,
                     'ftp-data': 20.0, 'irc': 6667.0, 'pop3': 110.0, 'ssl': 443.0,
                     'snmp': 161.0, 'dhcp': 67.0, 'radius': 1812.0}
    fwd = np.maximum(1.0, df['spkts'].astype(float))
    bwd = np.maximum(0.0, df['dpkts'].astype(float))
    lf = np.maximum(0.0, df['sbytes'].astype(float))
    lb = np.maximum(0.0, df['dbytes'].astype(float))
    dur_us = np.maximum(10.0, df['dur'].astype(float) * 1e6)
    is_tcp = proto == 6.0
    out = pd.DataFrame({
        'dst_port': df['service'].str.lower().map(service_ports).fillna(0.0),
        'protocol': proto,
        'tot_fwd_pkts': fwd, 'tot_bwd_pkts': bwd,
        'totlen_fwd_pkts': lf, 'totlen_bwd_pkts': lb,
        'flow_duration': dur_us,
        'init_fwd_win_byts': np.where(is_tcp, np.clip(df['swin'].astype(float), 0, 65535), 0.0),
        'flow_bytes_s': (lf + lb) / (dur_us / 1e6),
        'flow_pkts_s': (fwd + bwd) / (dur_us / 1e6),
        'fwd_pkt_len_mean': lf / fwd,
        'fwd_header_len': np.where(is_tcp, fwd * 32.0, fwd * 8.0),
        'label_orig': df['attack_cat'].astype(str).str.strip(),
        'label': df['label'].astype(int),
        'source': df['source'],
    })
    out['row_order'] = np.arange(len(out))
    return _clean(out)


def build_full_dataset(name: str, force: bool = False) -> pd.DataFrame:
    """Return the deduplicated full dataset (``cse2018``, ``cic2017`` or ``unsw``)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{name}.parquet')
    if os.path.exists(cache) and not force:
        df = pd.read_parquet(cache)
        stats_path = os.path.join(CACHE_DIR, f'{name}_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as fh:
                df.attrs['dedup_stats'] = json.load(fh)
        return df

    if name == 'cse2018':
        df = _build_csv_dataset(name, _family_2018)
        df['attack_family'] = df['label_orig'].map(_family_2018)
        df['label'] = (df['attack_family'] != 'Benign').astype(int)
    elif name == 'cic2017':
        df = _build_csv_dataset(name, _family_2017)
        df['attack_family'] = df['label_orig'].map(_family_2017)
        df['label'] = (df['attack_family'] != 'Benign').astype(int)
    elif name == 'unsw':
        df = _build_unsw()
        df['attack_family'] = np.where(df['label'] == 0, 'Benign', df['label_orig'])
    else:
        raise ValueError(f'unknown dataset {name!r}')

    n_before = len(df)
    # A flow that appears with conflicting labels is unusable as ground truth.
    key = FEATURES
    n_labels = df.groupby(key, sort=False)['label'].transform('nunique')
    df = df[n_labels == 1]
    n_conflict = n_before - len(df)
    df = df.drop_duplicates(subset=key + ['label'], keep='first').reset_index(drop=True)
    stats = {'before_dedup': n_before, 'label_conflicts_removed': int(n_conflict),
             'after_dedup': len(df), 'attacks': int(df['label'].sum())}
    print(f'  [{name}] dedup summary: {stats}')
    df.attrs['dedup_stats'] = stats
    df.to_parquet(cache, index=False)
    pd.Series(stats).to_json(os.path.join(CACHE_DIR, f'{name}_stats.json'))
    return df


def stratified_split(df: pd.DataFrame, seed: int = 42,
                     fractions: Tuple[float, float, float] = (0.7, 0.1, 0.2)
                     ) -> Dict[str, pd.DataFrame]:
    """Stratified (label x family) random train/val/test split of a deduplicated frame."""
    rng = np.random.RandomState(seed)
    strata = df['label'].astype(str) + '|' + df['attack_family'].astype(str)
    assign = np.empty(len(df), dtype=np.int8)
    for _, idx in df.groupby(strata, sort=False).indices.items():
        idx = rng.permutation(idx)
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        assign[idx[:n_tr]] = 0
        assign[idx[n_tr:n_tr + n_va]] = 1
        assign[idx[n_tr + n_va:]] = 2
    return {k: df[assign == i].reset_index(drop=True)
            for i, k in enumerate(('train', 'val', 'test'))}
