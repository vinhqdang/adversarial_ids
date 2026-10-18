"""Tests for the full-scale data pipeline and the surrogate-based baselines."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments')))

from src.datasets import full_data
from src.datasets.full_data import FEATURES, build_full_dataset, stratified_split
from src.datasets.schema import (get_cicids_schema, project_network_invariants,
                                 project_network_invariants_batch, verify_physical_feasibility)
from src.attacks.modern_baselines import (CAPGDTransferAttack, LogStandardizer, NIDSGANAttack,
                                          train_surrogate)

CSE_HEADER = ['Dst Port', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
              'TotLen Bwd Pkts', 'Flow Duration', 'Init Fwd Win Byts', 'Flow Byts/s',
              'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Fwd Header Len', 'Label']


def _flow(port=80, proto=6, fwd=4, bwd=3, lf=400, lb=900, dur=2_000_000, win=8192, label='Benign'):
    secs = max(dur, 1) / 1e6
    return [port, proto, fwd, bwd, lf, lb, dur, win, (lf + lb) / secs, (fwd + bwd) / secs,
            lf / fwd, fwd * 20, label]


def _random_flows(n, seed=0):
    rng = np.random.RandomState(seed)
    X = np.column_stack([
        rng.randint(1, 65535, n), rng.choice([6, 17], n), rng.randint(1, 200, n),
        rng.randint(0, 200, n), rng.uniform(0, 3e5, n), rng.uniform(0, 3e5, n),
        rng.uniform(10, 1.2e8, n), rng.uniform(-10, 70000, n), rng.uniform(0, 1e7, n),
        rng.uniform(0, 1e4, n), rng.uniform(0, 1500, n), rng.uniform(0, 12000, n)])
    return X


def test_batch_projection_matches_scalar_projection():
    # Arrange
    schema = get_cicids_schema()
    X = _random_flows(400)
    # Act
    batch = project_network_invariants_batch(X)
    scalar = np.array([project_network_invariants(x, schema) for x in X])
    # Assert
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-9)


def test_batch_projection_does_not_mutate_input():
    X = _random_flows(20)
    before = X.copy()
    project_network_invariants_batch(X)
    np.testing.assert_array_equal(X, before)


@pytest.fixture
def raw_cse(tmp_path, monkeypatch):
    folder = tmp_path / 'raw' / 'cse2018'
    folder.mkdir(parents=True)
    rows = [
        _flow(label='Benign'), _flow(label='Benign'),                     # exact duplicate
        _flow(port=22, fwd=9, lf=900, label='SSH-Bruteforce'),
        _flow(port=53, proto=17, win=-1, label='DDOS attack-LOIC-UDP'),    # UDP: CICFlowMeter writes -1
        _flow(port=443, fwd=7, lf=700, label='Benign'),                   # same features, two labels
        _flow(port=443, fwd=7, lf=700, label='DoS attacks-SlowHTTPTest'),
        _flow(port=8080, fwd=5, lf=500, label='Infilteration'),
        _flow(port=80, fwd=6, lf=600, label='Brute Force -Web'),
        _flow(port=80, fwd=3, lf=300, label='SQL Injection'),
        _flow(port=80, fwd=2, lf=200, win=-1, label='Benign'),            # TCP with no observed window
        _flow(dur=0, label='Benign'),                                     # invalid duration
        _flow(proto=1, label='Benign'),                                   # not TCP/UDP
    ]
    pd.DataFrame(rows, columns=CSE_HEADER).to_csv(folder / 'day1.csv', index=False)
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    return tmp_path


def test_build_full_dataset_cleans_dedups_and_maps_families(raw_cse):
    df = build_full_dataset('cse2018', force=True)
    stats = df.attrs['dedup_stats']
    # 12 raw rows: invalid duration, protocol 1 and TCP without window dropped -> 9 valid;
    # one exact duplicate collapsed -> 8; the conflicting pair is kept and counted.
    assert stats['raw_rows'] == 12 and stats['valid_records'] == 9
    assert stats['duplicates_removed'] == 1 and len(df) == 8
    assert stats['conflicting_label_records'] == 2
    fams = dict(zip(df['label_orig'], df['attack_family']))
    assert fams['SSH-Bruteforce'] == 'BruteForce'
    assert fams['DDOS attack-LOIC-UDP'] == 'DDoS'
    assert fams['DoS attacks-SlowHTTPTest'] == 'DoS'
    assert fams['Infilteration'] == 'InfiltrationBeacon'
    assert fams['Brute Force -Web'] == 'WebExploit' and fams['SQL Injection'] == 'WebExploit'
    assert set(df['label']) == {0, 1}
    assert (df['flow_duration'] > 0).all() and df['protocol'].isin([6, 17]).all()


def test_udp_flows_survive_with_zero_window(raw_cse):
    df = build_full_dataset('cse2018', force=True)
    udp = df[df['protocol'] == 17]
    assert len(udp) == 1 and (udp['init_fwd_win_byts'] == 0).all()


def test_projection_is_idempotent_so_dedup_after_projection_is_stable():
    X = project_network_invariants_batch(_random_flows(300, seed=9))
    np.testing.assert_allclose(project_network_invariants_batch(X), X, rtol=1e-9, atol=1e-6)


def test_build_full_dataset_reuses_cache_and_keeps_stats(raw_cse):
    build_full_dataset('cse2018', force=True)
    cached = build_full_dataset('cse2018')
    assert len(cached) == 8
    assert cached.attrs['dedup_stats']['after_dedup'] == 8


def test_build_full_dataset_rejects_unknown_name(raw_cse):
    with pytest.raises(ValueError):
        build_full_dataset('nope')


def test_build_full_dataset_requires_raw_files(tmp_path, monkeypatch):
    (tmp_path / 'raw' / 'cic2017').mkdir(parents=True)
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    with pytest.raises(FileNotFoundError):
        build_full_dataset('cic2017', force=True)


def test_stratified_split_is_disjoint_exhaustive_and_stratified():
    rng = np.random.RandomState(1)
    n = 5000
    df = pd.DataFrame({'label': (rng.rand(n) < 0.2).astype(int)})
    df['attack_family'] = np.where(df['label'] == 0, 'Benign', rng.choice(['A', 'B'], n))
    df['id'] = np.arange(n)
    parts = stratified_split(df, seed=3)
    ids = [set(p['id']) for p in parts.values()]
    assert sum(len(i) for i in ids) == n
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert abs(len(parts['train']) / n - 0.7) < 0.01
    for p in parts.values():
        assert abs(p['label'].mean() - df['label'].mean()) < 0.02


def test_stratified_split_is_deterministic_for_a_seed():
    df = pd.DataFrame({'label': [0, 1] * 200, 'attack_family': ['Benign', 'A'] * 200, 'id': range(400)})
    a = stratified_split(df, seed=5)['test']['id'].tolist()
    b = stratified_split(df, seed=5)['test']['id'].tolist()
    assert a == b


def test_wilson_interval_brackets_the_estimate_and_handles_empty():
    from run_full_benchmark import wilson
    lo, hi = wilson(60, 100)
    assert lo < 60 < hi and 49 < lo < 52 and 68 < hi < 70
    assert wilson(0, 0) == [0.0, 0.0]
    assert wilson(100, 100)[1] == pytest.approx(100.0)


class _StubModel:
    """Black-box target: malicious when forward packets are few."""

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        p = 1.0 / (1.0 + np.exp((X[:, 2] - 20.0) / 3.0))
        return np.column_stack([1 - p, p])


@pytest.fixture(scope='module')
def surrogate_bundle():
    schema = get_cicids_schema()
    raw = _random_flows(4000, seed=2)
    raw[:, 2] = np.random.RandomState(2).randint(1, 41, len(raw))  # balanced around the target's threshold
    X = project_network_invariants_batch(raw)
    y = (X[:, 2] < 20).astype(int)
    norm = LogStandardizer.fit(X)
    net = train_surrogate(X, y, norm, epochs=10)
    gen = NIDSGANAttack.train_generator(X[y == 1], net, norm, schema, epochs=2)
    return schema, X, y, norm, net, gen


def test_log_standardizer_roundtrip():
    X = project_network_invariants_batch(_random_flows(50, seed=4))
    norm = LogStandardizer.fit(X)
    np.testing.assert_allclose(norm.inverse(norm.forward(X)), X, rtol=1e-6, atol=1e-6)


def test_capgd_returns_feasible_flows_and_keeps_immutables(surrogate_bundle):
    schema, X, y, norm, net, _ = surrogate_bundle
    atk = CAPGDTransferAttack(_StubModel(), schema, net, norm, max_iter=10)
    flow = X[y == 1][0]
    res = atk.attack_sample(flow)
    assert 1 <= res.queries_used <= 10
    assert res.physically_valid == verify_physical_feasibility(flow, res.x_adversarial, schema)
    for idx in schema.immutable_indices:
        assert res.x_adversarial[idx] == pytest.approx(flow[idx])
    assert res.success == (atk.model.predict_proba(res.x_adversarial)[0, 1] < 0.5)


def test_capgd_evades_a_weak_target_with_feasible_flows(surrogate_bundle):
    schema, X, y, norm, net, _ = surrogate_bundle
    atk = CAPGDTransferAttack(_StubModel(), schema, net, norm, max_iter=20)
    flows = X[y == 1][:20]
    results = [atk.attack_sample(f) for f in flows]
    assert sum(r.success for r in results) >= 1
    assert all(r.physically_valid for r in results if r.success)


def test_nidsgan_style_attack_queries_at_most_n_draws(surrogate_bundle):
    schema, X, y, norm, net, gen = surrogate_bundle
    atk = NIDSGANAttack(_StubModel(), schema, net, norm, gen, n_draws=3)
    res = atk.attack_sample(X[y == 1][0])
    assert 1 <= res.queries_used <= 3
    for idx in schema.immutable_indices:
        assert res.x_adversarial[idx] == pytest.approx(X[y == 1][0][idx])


CIC17_HEADER = ['Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
                'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Flow Bytes/s',
                'Flow Packets/s', 'Fwd Packet Length Mean', 'Fwd Header Length',
                'FWD Init Win Bytes', 'Label']


def test_build_cic2017_maps_aliases_families_and_drops_attempted(tmp_path, monkeypatch):
    folder = tmp_path / 'raw' / 'cic2017'
    folder.mkdir(parents=True)
    rows = [[80, 6, 2_000_000, 4, 3, 400, 900, 650.0, 3.5, 100.0, 80, 8192, 'BENIGN'],
            [21, 6, 1_000_000, 5, 2, 500, 100, 600.0, 7.0, 100.0, 100, 4096, 'FTP-Patator'],
            [80, 6, 3_000_000, 6, 1, 600, 50, 216.0, 2.3, 100.0, 120, 2048, 'DoS Hulk'],
            [80, 6, 4_000_000, 8, 1, 800, 50, 212.0, 2.2, 100.0, 160, 1024, 'DDoS'],
            [80, 6, 5_000_000, 9, 1, 900, 50, 190.0, 2.0, 100.0, 180, 512, 'Portscan'],
            [80, 6, 6_000_000, 3, 1, 300, 50, 58.0, 0.6, 100.0, 60, 256, 'Infiltration - Portscan'],
            [80, 6, 7_000_000, 2, 1, 200, 50, 35.0, 0.4, 100.0, 40, 128, 'Web Attack - XSS - Attempted']]
    pd.DataFrame(rows, columns=CIC17_HEADER).to_csv(folder / 'monday.csv', index=False)
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    df = build_full_dataset('cic2017', force=True)
    assert dict(zip(df['label_orig'], df['attack_family'])) == {
        'BENIGN': 'Benign', 'FTP-Patator': 'BruteForce', 'DoS Hulk': 'DoS', 'DDoS': 'DDoS',
        'Portscan': 'PortScan', 'Infiltration - Portscan': 'PortScan'}
    assert df.attrs['dedup_stats']['attempted_removed'] == 1
    assert df.loc[df['label_orig'] == 'BENIGN', 'init_fwd_win_byts'].iloc[0] == 8192


def test_build_unsw_maps_protocol_ports_and_labels(tmp_path, monkeypatch):
    folder = tmp_path / 'raw' / 'unsw'
    folder.mkdir(parents=True)
    base = {'proto': ['tcp', 'udp', 'icmp', 'tcp'], 'service': ['http', 'dns', '-', 'ssh'],
            'spkts': [10, 2, 1, 8], 'dpkts': [8, 2, 0, 6], 'sbytes': [900, 120, 60, 700],
            'dbytes': [5000, 150, 0, 400], 'dur': [0.5, 0.01, 0.2, 1.5], 'swin': [255, 0, 0, 255],
            'label': [0, 1, 1, 1], 'attack_cat': ['Normal', 'Generic', 'Worms', 'Exploits']}
    pd.DataFrame(base).to_parquet(folder / 'train.parquet')
    pd.DataFrame(base).to_parquet(folder / 'test.parquet')
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    df = build_full_dataset('unsw', force=True)
    assert len(df) == 3                       # icmp dropped, train/test duplicates collapsed
    assert set(df['protocol']) == {6.0, 17.0}
    assert df.loc[df['attack_family'] == 'Generic', 'dst_port'].iloc[0] == 53.0
    assert df.loc[df['label'] == 0, 'attack_family'].iloc[0] == 'Benign'
    udp = df[df['protocol'] == 17.0].iloc[0]
    assert udp['init_fwd_win_byts'] == 0.0 and udp['fwd_header_len'] == udp['tot_fwd_pkts'] * 8


def test_download_raw_skips_complete_files_and_streams_missing(tmp_path, monkeypatch):
    payload = b'a,b\n1,2\n'
    calls = {'get': 0}

    class _Resp:
        headers = {'Content-Length': str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, _size):
            yield payload

    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path))
    monkeypatch.setattr(full_data.requests, 'head', lambda *a, **k: _Resp())

    def fake_get(*a, **k):
        calls['get'] += 1
        return _Resp()
    monkeypatch.setattr(full_data.requests, 'get', fake_get)
    full_data.download_raw('cic2017')
    assert calls['get'] == 5                  # one per daily file
    full_data.download_raw('cic2017')
    assert calls['get'] == 5                  # complete files are not fetched again
    assert (tmp_path / 'cic2017' / 'friday.csv').read_bytes() == payload


def test_numeric_frame_requires_all_schema_features():
    raw = pd.DataFrame({'Dst Port': [1], 'Protocol': [6]})
    with pytest.raises(ValueError):
        full_data._numeric_frame(raw, full_data.ALIASES['cse2018'])


def test_unsw_drops_zero_duration_flows(tmp_path, monkeypatch):
    folder = tmp_path / 'raw' / 'unsw'
    folder.mkdir(parents=True)
    base = {'proto': ['tcp', 'tcp'], 'service': ['http', 'ssh'], 'spkts': [4, 5], 'dpkts': [2, 3],
            'sbytes': [400, 500], 'dbytes': [100, 200], 'dur': [0.0, 0.5], 'swin': [255, 255],
            'label': [0, 1], 'attack_cat': ['Normal', 'Exploits']}
    pd.DataFrame(base).to_parquet(folder / 'train.parquet')
    pd.DataFrame(base).to_parquet(folder / 'test.parquet')
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    df = build_full_dataset('unsw', force=True)
    assert len(df) == 1 and df['label'].iloc[0] == 1


def test_detected_by_all_requires_every_model_and_masked_wrapper_hides_columns():
    from common import ColumnMaskedModel, detected_by_all

    class _ColsModel:
        def __init__(self, col):
            self.col = col

        def predict_proba(self, X):
            X = np.atleast_2d(X)
            p = (X[:, self.col] > 0.5).astype(float)
            return np.column_stack([1 - p, p])

    X = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    mask = detected_by_all([_ColsModel(0), _ColsModel(1)], X)
    assert mask.tolist() == [True, False, False, False]

    class _Trainer:
        seen = None

        def predict_proba(self, X):
            _Trainer.seen = X
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        def get_split_thresholds(self):
            return [(0, 0.5), (1, 2.5)]

    wrapped = ColumnMaskedModel(_Trainer(), masked=[1], n_features=4)
    wrapped.predict_proba(np.arange(8.0).reshape(2, 4))
    assert _Trainer.seen.shape == (2, 3)                            # column 1 never reaches the model
    assert wrapped.get_split_thresholds() == [(0, 0.5), (2, 2.5)]   # indices mapped back


def test_catboost_split_thresholds_are_extracted():
    from src.models.tree_models import NIDSEnsembleTrainer
    rng = np.random.RandomState(0)
    X = rng.rand(400, 12)
    y = (X[:, 2] > 0.5).astype(int)
    t = NIDSEnsembleTrainer('catboost', {'iterations': 8, 'depth': 3, 'learning_rate': 0.3,
                                         'random_seed': 1, 'verbose': False})
    t.train(X, y)
    splits = t.get_split_thresholds()
    assert splits and all(0 <= f < 12 for f, _ in splits)
    assert any(f == 2 for f, _ in splits)


def test_operational_cost_prices_delay_padding_chaff_and_window():
    from src.attacks.cost import operational_cost
    x0 = np.array([80, 6, 4, 3, 400, 900, 1_000_000, 1000, 0, 0, 0, 80], dtype=float)
    assert operational_cost(x0, x0) == 0.0
    delayed = x0.copy(); delayed[6] += 2_000_000                    # 2 s of delay
    assert operational_cost(x0, delayed) == pytest.approx(2e-4)
    padded = x0.copy(); padded[4] += 100                             # 100 padding bytes
    assert operational_cost(x0, padded) == pytest.approx(1.0)
    chaff = x0.copy(); chaff[2] += 2; chaff[4] += 80                 # two 40-byte TCP probes, no padding
    assert operational_cost(x0, chaff) == pytest.approx(2.0)
    win = x0.copy(); win[7] += 6553.5                                # 10% of the window range
    assert operational_cost(x0, win) == pytest.approx(1e-4)
    shrunk = x0.copy(); shrunk[6] -= 5e5; shrunk[4] -= 50           # removing delay or bytes is free
    assert operational_cost(x0, shrunk) == 0.0
