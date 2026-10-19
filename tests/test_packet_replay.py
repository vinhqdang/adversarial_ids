"""Tests for the synthetic packet-level replay (src/validation/packet_replay.py)."""

import numpy as np
import pytest

from src.datasets.schema import (get_cicids_schema, project_network_invariants_batch,
                                 verify_physical_feasibility)
from src.validation.packet_replay import (
    BYTES_S, DST_PORT, DURATION, FWD_HDR, FWD_MEAN, FWD_PKTS, BWD_PKTS, LEN_BWD, LEN_FWD, MTU,
    PKTS_S, PROTO, WIN, ActionPlan, apply_actions, chaff_injection, chaff_geometry, derive_actions,
    extract_features, header_option_padding, payload_padding, relative_error, replay,
    temporal_dilation, trace_from_record, window_morph,
)


def make_records(n: int = 200, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 12))
    X[:, 0] = rng.choice([22, 53, 80, 443], n)
    X[:, 1] = rng.choice([6, 17], n)
    X[:, 2] = rng.randint(1, 60, n)
    X[:, 3] = rng.randint(0, 60, n)
    X[:, 4] = np.floor(rng.uniform(0, 1400, n) * X[:, 2])
    X[:, 5] = np.floor(rng.uniform(0, 1400, n) * X[:, 3])
    X[:, 6] = rng.randint(50, 5_000_000, n)
    X[:, 7] = rng.choice([0, 8192, 29200, 65535], n)
    X[:, 11] = X[:, 2] * rng.choice([20, 24, 32], n)
    P = project_network_invariants_batch(X)
    P[:, 2] = np.where(P[:, 2] + P[:, 3] < 2, P[:, 2] + 1, P[:, 2])   # >= 2 packets so duration is observable
    return project_network_invariants_batch(P)


RECORDS = make_records()
TCP_RECORDS = RECORDS[RECORDS[:, PROTO] == 6]
UDP_RECORDS = RECORDS[RECORDS[:, PROTO] == 17]


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---- round trip -----------------------------------------------------------------------------
@pytest.mark.parametrize('x', list(RECORDS))
def test_roundtrip_reproduces_record(x):
    got = extract_features(trace_from_record(x, rng(7)))
    assert relative_error(x, got).max() < 1e-9


def test_trace_is_physically_plausible():
    for x in RECORDS[:50]:
        t = trace_from_record(x, rng(3))
        assert t.fwd[0] and np.all(np.diff(t.ts) >= 0) and t.ts[0] == 0
        assert (t.payload >= 0).all()
        assert (t.payload <= 1500).all()    # schema bound; payload plus headers may exceed the MTU
        if x[PROTO] == 17:
            assert (t.win == 0).all() and (t.header == 8).all()
        else:
            assert (t.header[t.fwd] >= 20).all() and (t.header[t.fwd] <= 60).all()


def test_trace_depends_on_seed_but_features_do_not():
    x = TCP_RECORDS[0]
    a, b = trace_from_record(x, rng(1)), trace_from_record(x, rng(2))
    assert not np.array_equal(a.ts, b.ts) or not np.array_equal(a.payload, b.payload)
    assert np.array_equal(extract_features(a), extract_features(b))


def test_extractor_definitions_on_handmade_trace():
    x = np.array([443, 6, 3, 2, 300, 100, 2000, 5000, 0, 0, 0, 66], dtype=float)
    x = project_network_invariants_batch(x[None])[0]
    f = extract_features(trace_from_record(x, rng(0)))
    assert f[DURATION] == 2000 and f[WIN] == 5000
    assert f[BYTES_S] == pytest.approx(400 / 0.002) and f[PKTS_S] == pytest.approx(5 / 0.002)
    assert f[FWD_MEAN] == pytest.approx(100.0) and f[FWD_HDR] == x[FWD_HDR]


# ---- each action changes only what it should -----------------------------------------------
@pytest.mark.parametrize('x', list(TCP_RECORDS[:40]))
def test_temporal_dilation_changes_only_duration_and_rates(x):
    t = trace_from_record(x, rng(1))
    f0, f1 = extract_features(t), extract_features(temporal_dilation(t, 123_456))
    assert f1[DURATION] == f0[DURATION] + 123_456
    assert f1[BYTES_S] < f0[BYTES_S] + 1e-9 and f1[PKTS_S] < f0[PKTS_S] + 1e-9
    same = [i for i in range(12) if i not in (DURATION, BYTES_S, PKTS_S)]
    assert np.array_equal(f0[same], f1[same])


def test_temporal_dilation_preserves_order_and_zero_is_identity():
    t = trace_from_record(TCP_RECORDS[3], rng(1))
    d = temporal_dilation(t, 10_000)
    assert np.all(np.diff(d.ts) >= 0) and (d.fwd == t.fwd).all() and (d.payload == t.payload).all()
    assert temporal_dilation(t, 0) is t


@pytest.mark.parametrize('x', list(TCP_RECORDS[:40]))
def test_padding_changes_only_forward_bytes(x):
    t = trace_from_record(x, rng(1))
    head = int(np.where(t.fwd, MTU - 20 - t.header - t.payload, 0).sum())
    pad = min(500, head)
    p, unmet = payload_padding(t, pad)
    f0, f1 = extract_features(t), extract_features(p)
    assert unmet == 0 and f1[LEN_FWD] == f0[LEN_FWD] + pad
    assert f1[FWD_MEAN] >= f0[FWD_MEAN] and f1[BYTES_S] >= f0[BYTES_S]
    same = [i for i in range(12) if i not in (LEN_FWD, FWD_MEAN, BYTES_S)]
    assert np.array_equal(f0[same], f1[same])
    assert (p.payload + p.header + 20 <= np.maximum(MTU, t.payload + t.header + 20)).all()   # MTU respected


def test_padding_is_capped_at_mtu_and_reports_unmet():
    x = project_network_invariants_batch(np.array([[80, 6, 2, 2, 2800, 100, 1000, 100, 0, 0, 0, 40.0]]))[0]
    t = trace_from_record(x, rng(0))
    p, unmet = payload_padding(t, 10_000)
    assert unmet > 0
    over = p.payload + p.header + 20 > MTU
    assert (p.payload[over] == t.payload[over]).all()   # padding never pushes a packet beyond the MTU
    assert p.payload[~p.fwd].sum() == t.payload[~t.fwd].sum()   # backward untouched


@pytest.mark.parametrize('x', list(TCP_RECORDS[:30]) + list(UDP_RECORDS[:10]))
def test_chaff_adds_forward_packets_and_keeps_duration(x):
    t = trace_from_record(x, rng(1))
    h0, _ = chaff_geometry(x)
    c = chaff_injection(t, 7, h0, 0, rng(5))
    f0, f1 = extract_features(c), extract_features(t)
    assert f0[FWD_PKTS] == f1[FWD_PKTS] + 7 and f0[FWD_HDR] == f1[FWD_HDR] + 7 * h0
    assert f0[LEN_FWD] == f1[LEN_FWD] and f0[DURATION] == f1[DURATION]
    assert f0[BWD_PKTS] == f1[BWD_PKTS] and f0[LEN_BWD] == f1[LEN_BWD] and f0[WIN] == f1[WIN]
    assert f0[PKTS_S] >= f1[PKTS_S] and f0[FWD_MEAN] <= f1[FWD_MEAN] + 1e-9
    assert c.fwd[np.argmax(c.fwd)] and c.ts[0] == t.ts[0]


def test_window_morph_changes_only_initial_window():
    x = TCP_RECORDS[0]
    t = trace_from_record(x, rng(1))
    f0, f1 = extract_features(t), extract_features(window_morph(t, 64240))
    assert f1[WIN] == 64240
    assert np.array_equal(np.delete(f0, WIN), np.delete(f1, WIN))


def test_window_morph_is_noop_on_udp_and_clipped():
    u = trace_from_record(UDP_RECORDS[0], rng(0))
    assert extract_features(window_morph(u, 64240))[WIN] == 0
    t = trace_from_record(TCP_RECORDS[0], rng(0))
    assert extract_features(window_morph(t, 10 ** 6))[WIN] == 65535


def test_header_option_padding_bounded():
    t = trace_from_record(TCP_RECORDS[1], rng(0))
    p, unmet = header_option_padding(t, 10 ** 6)
    assert unmet > 0 and (p.header[p.fwd] == 60).all() and (p.header[~p.fwd] == t.header[~t.fwd]).all()


# ---- derive + apply: invariants ------------------------------------------------------------
def adversarial_pair(x: np.ndarray):
    """A hand-built perturbation using all four actions, projected as FFE does."""
    h0, l0 = chaff_geometry(x)
    xa = x.copy()
    xa[FWD_PKTS] += 3
    xa[LEN_FWD] += 3 * l0 + 200
    xa[FWD_HDR] += 3 * h0
    xa[DURATION] += 400_000
    if x[PROTO] == 6:
        xa[WIN] = 64240
    return xa, project_network_invariants_batch(xa[None])[0]


@pytest.mark.parametrize('mode', ['schema', 'empty_compensated'])
def test_replay_reproduces_schema_level_features(mode):
    schema = get_cicids_schema()
    for x in RECORDS[:60]:
        _, xa = adversarial_pair(x)
        out = replay(x, xa, mode, seed=3)
        assert relative_error(xa, out['features']).max() < 1e-9
        assert verify_physical_feasibility(x, out['features'], schema)
        assert np.allclose(out['baseline_features'], x)


def test_empty_literal_chaff_underdelivers_payload_bytes():
    x = TCP_RECORDS[0]
    _, xa = adversarial_pair(x)
    f = replay(x, xa, 'empty_literal', seed=1)['features']
    _, l0 = chaff_geometry(x)
    assert f[LEN_FWD] == xa[LEN_FWD] - 3 * l0 and f[FWD_PKTS] == xa[FWD_PKTS]


def test_replay_never_violates_monotonicity_or_immutables():
    for x in RECORDS[:60]:
        _, xa = adversarial_pair(x)
        f = replay(x, xa, 'schema', seed=2)['features']
        assert f[FWD_PKTS] >= x[FWD_PKTS] and f[LEN_FWD] >= x[LEN_FWD] and f[DURATION] >= x[DURATION]
        assert f[FWD_HDR] >= x[FWD_HDR] and f[DST_PORT] == x[DST_PORT] and f[PROTO] == x[PROTO]
        assert f[BWD_PKTS] == x[BWD_PKTS] and f[LEN_BWD] == x[LEN_BWD]
        assert f[FWD_PKTS] * 1500 >= f[LEN_FWD]


def test_derive_actions_identity_and_modes():
    x = TCP_RECORDS[0]
    p = derive_actions(x, x)
    assert p == ActionPlan(0, 0, p.chaff_header, 0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        derive_actions(x, x, 'bogus')
    _, xa = adversarial_pair(x)
    ps, pc, pl = (derive_actions(x, xa, m) for m in ('schema', 'empty_compensated', 'empty_literal'))
    assert ps.n_chaff == pc.n_chaff == pl.n_chaff == 3
    assert pc.chaff_payload == 0 and pc.pad_bytes == xa[LEN_FWD] - x[LEN_FWD]
    assert pl.pad_bytes == pc.pad_bytes - 3 * chaff_geometry(x)[1]


def test_apply_actions_is_pure():
    x = TCP_RECORDS[2]
    t = trace_from_record(x, rng(0))
    ts, payload = t.ts.copy(), t.payload.copy()
    _, xa = adversarial_pair(x)
    apply_actions(t, derive_actions(x, xa), rng(1))
    assert np.array_equal(ts, t.ts) and np.array_equal(payload, t.payload)
