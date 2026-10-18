"""Tests for the InSDN, USB-IDS-1 and CIDDS-001 builders."""

import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import full_data
from src.datasets.full_data import build_full_dataset

CSE_HEADER = ['Dst Port', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
              'TotLen Bwd Pkts', 'Flow Duration', 'Init Fwd Win Byts', 'Flow Byts/s',
              'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Fwd Header Len', 'Label']
CIC17_HEADER = ['Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
                'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Flow Bytes/s',
                'Flow Packets/s', 'Fwd Packet Length Mean', 'Fwd Header Length',
                'FWD Init Win Bytes', 'Label']


def _cse_row(port, proto, fwd, label, win=-1, dur=2_000_000):
    lf = fwd * 100
    secs = dur / 1e6
    return [port, proto, fwd, 3, lf, 900, dur, win, (lf + 900) / secs, (fwd + 3) / secs, 100.0, fwd * 20, label]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(full_data, 'RAW_DIR', str(tmp_path / 'raw'))
    monkeypatch.setattr(full_data, 'CACHE_DIR', str(tmp_path / 'cache'))
    return tmp_path


def test_insdn_ignores_missing_window_keeps_tcp_udp_and_strips_labels(dirs):
    folder = dirs / 'raw' / 'insdn' / 'InSDN_DatasetCSV'      # the archive extracts into a sub-folder
    folder.mkdir(parents=True)
    rows = [_cse_row(80, 6, 4, 'Normal'), _cse_row(53, 17, 2, 'Normal'),
            _cse_row(80, 6, 9, 'DoS'), _cse_row(80, 6, 7, 'DDoS '),
            _cse_row(22, 6, 5, 'BFA'), _cse_row(80, 0, 6, 'DDoS')]    # protocol 0 is neither TCP nor UDP
    pd.DataFrame(rows, columns=CSE_HEADER).to_csv(folder / 'OVS.csv', index=False)
    df = build_full_dataset('insdn', force=True)
    assert len(df) == 5 and set(df['protocol']) == {6.0, 17.0}
    assert (df['init_fwd_win_byts'] == 0).all()                # window was -1 everywhere: constant 0
    fam = dict(zip(df['label_orig'], df['attack_family']))
    assert fam['DDoS'] == 'DDoS' and fam['BFA'] == 'BruteForce' and fam['Normal'] == 'Benign'


def test_usbids1_reads_gzip_and_maps_tools_to_families(dirs):
    folder = dirs / 'raw' / 'usbids1'
    folder.mkdir(parents=True)
    rows = [[80, 6, 2_000_000, 4, 3, 400, 900, 650.0, 3.5, 100.0, 80, 8192, 'BENIGN'],
            [80, 6, 1_000_000, 9, 1, 900, 50, 950.0, 10.0, 100.0, 180, 512, 'Hulk-NoDefense'],
            [80, 6, 3_000_000, 6, 1, 600, 50, 216.0, 2.3, 100.0, 120, 2048, 'TCPFlood-Evasive']]
    with gzip.open(folder / 'sample.csv.gz', 'wt') as fh:
        pd.DataFrame(rows, columns=CIC17_HEADER).to_csv(fh, index=False)
    df = build_full_dataset('usbids1', force=True)
    assert dict(zip(df['label_orig'], df['attack_family'])) == {
        'BENIGN': 'Benign', 'Hulk-NoDefense': 'Hulk', 'TCPFlood-Evasive': 'TCPFlood'}
    assert df['label'].tolist().count(1) == 2


def test_parse_bytes_understands_netflow_suffixes():
    s = pd.Series(['174', '2.1 M', ' 3.4 G', '12 K', 'x'])
    out = full_data._parse_bytes(s)
    assert out.iloc[:4].tolist() == pytest.approx([174.0, 2.1e6, 3.4e9, 12e3])
    assert np.isnan(out.iloc[4])


def test_cidds_keeps_normal_and_attacker_and_derives_unidirectional_features(dirs):
    folder = dirs / 'raw' / 'cidds001' / 'CIDDS-001' / 'traffic' / 'OpenStack'
    folder.mkdir(parents=True)
    head = 'Date first seen,Duration,Proto,Src IP Addr,Src Pt,Dst IP Addr,Dst Pt,Packets,Bytes,Flows,Flags,Tos,class,attackType,attackID,attackDescription'
    lines = [head,
             '2017-04-05 00:00:00.266,    2.000,TCP  ,10.0.0.1, 40289,10.0.0.2,   445,      10,     1000,    1,.AP...,  0,normal,---,---,---',
             '2017-04-05 00:00:01.000,    1.000,UDP  ,10.0.0.1,   137,10.0.0.2,   137,       4,   2.1 M,    1,......,  0,attacker,dos,12,x',
             '2017-04-05 00:00:02.000,    1.000,TCP  ,10.0.0.2,   445,10.0.0.1, 40289,       3,      300,    1,.AP...,  0,victim,dos,12,x',
             '2017-04-05 00:00:03.000,    0.000,TCP  ,10.0.0.1, 40000,10.0.0.2,    80,       1,       60,    1,.AP...,  0,normal,---,---,---',
             '2017-04-05 00:00:04.000,    1.000,ICMP ,10.0.0.1,     0,10.0.0.2,   0.0,       1,       60,    1,......,  0,normal,---,---,---',
             '2017-04-05 00:00:05.000,    1.000,TCP  ,10.0.0.1, 40001,10.0.0.2,    22,       5,      500,    1,.AP...,  0,suspicious,---,---,---']
    (folder / 'week1.csv').write_text('\n'.join(lines) + '\n')
    df = build_full_dataset('cidds001', force=True)
    stats = df.attrs['dedup_stats']
    assert stats['raw_rows'] == 6 and len(df) == 2             # victim, zero duration, ICMP, suspicious dropped
    atk = df[df['label'] == 1].iloc[0]
    assert atk['attack_family'] == 'DoS' and atk['protocol'] == 17.0
    assert atk['tot_bwd_pkts'] == 0 and atk['totlen_bwd_pkts'] == 0 and atk['init_fwd_win_byts'] == 0
    assert atk['tot_fwd_pkts'] == 4 and atk['fwd_header_len'] == 4 * 8
    assert atk['totlen_fwd_pkts'] <= 1500 * atk['tot_fwd_pkts']  # projected: payload within the MTU bound
    ben = df[df['label'] == 0].iloc[0]
    assert ben['attack_family'] == 'Benign' and ben['init_fwd_win_byts'] == 0


def test_download_raw_extracts_zip_datasets(dirs, monkeypatch):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('InSDN_DatasetCSV/Normal_data.csv', 'a,b\n1,2\n')
    payload = buf.getvalue()

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

    monkeypatch.setattr(full_data.requests, 'head', lambda *a, **k: _Resp())
    monkeypatch.setattr(full_data.requests, 'get', lambda *a, **k: _Resp())
    full_data.download_raw('insdn')
    assert (dirs / 'raw' / 'insdn' / 'InSDN_DatasetCSV' / 'Normal_data.csv').exists()
