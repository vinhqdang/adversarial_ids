"""Tests for FFE-v2 (src/attacks/ffe_v2.py) and the packet-level semantics of its generators."""

import numpy as np
import pytest

from src.attacks.ffe_v2 import (
    DUR, FPKTS, HDR, LENF, WIN, FFEv2Attack, FFEv2Config, Ledger, _State, apply_generator,
    benign_window_candidates, chaff_header, logit, move_ladder,
)
from src.attacks.whitebox_tree import project_batch_with_origin
from src.datasets.generator import generate_cicids_traffic
from src.datasets.schema import get_cicids_schema, project_network_invariants, verify_physical_feasibility
from src.models.tree_models import NIDSEnsembleTrainer
from src.validation.packet_replay import (
    CHAFF_MODES, extract_features, plan_from_ledger, relative_error, replay_plan, resegment,
    trace_from_record,
)

SCHEMA = get_cicids_schema()


def project(x, x_orig):
    return project_network_invariants(x, SCHEMA, x_orig)


def records(n=150, seed=5):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 12))
    X[:, 0] = rng.choice([22, 80, 443], n)
    X[:, 1] = rng.choice([6, 6, 17], n)
    X[:, 2] = rng.randint(2, 40, n)
    X[:, 3] = rng.randint(0, 40, n)
    X[:, 4] = np.floor(rng.uniform(100, 1400, n) * X[:, 2])
    X[:, 5] = np.floor(rng.uniform(0, 1400, n) * X[:, 3])
    X[:, 6] = rng.randint(1000, 5_000_000, n)
    X[:, 7] = rng.choice([0, 8192, 29200, 65535], n)
    X[:, 11] = X[:, 2] * rng.choice([20, 24, 32], n)
    return np.array([project_network_invariants(x, SCHEMA) for x in X])


RECS = records()


@pytest.fixture(scope='module')
def setup():
    X, y, _ = generate_cicids_traffic(n_samples=4000, attack_ratio=0.3, random_state=3)
    X = np.array([project_network_invariants(x, SCHEMA) for x in X])
    model = NIDSEnsembleTrainer('xgboost', {'n_estimators': 30, 'max_depth': 5, 'random_state': 1})
    model.train(X[:3000], y[:3000])
    stds = X[:3000].std(0) + 1e-6
    mal = X[3000:][(y[3000:] == 1) & (model.predict_proba(X[3000:])[:, 1] >= 0.5)]
    return model, mal, X[:3000], y[:3000], stds


# ---- generators ------------------------------------------------------------------------------
@pytest.mark.parametrize('x', list(RECS[:40]))
def test_segmentation_changes_only_count_and_headers(x):
    k = 3
    adv = project(apply_generator(x, 'seg', k, x), x)
    h0, _ = chaff_header(x)
    assert adv[FPKTS] == x[FPKTS] + k
    assert adv[LENF] == x[LENF] and adv[DUR] == x[DUR] and adv[WIN] == x[WIN]
    assert adv[HDR] == x[HDR] + k * h0
    assert adv[10] == pytest.approx(x[LENF] / (x[FPKTS] + k))
    assert verify_physical_feasibility(x, adv, SCHEMA)


def test_every_ladder_candidate_is_feasible_and_batch_matches_scalar_projection():
    cfg = FFEv2Config(generators=frozenset({'time', 'pad', 'chaff', 'seg', 'win', 'hdr'}),
                      window_values=(0, 8192, 65535))
    for x in RECS:
        moves = move_ladder(x, x, cfg)
        assert moves
        raw = np.stack([apply_generator(x, g, a, x) for g, a in moves])
        batch = project_batch_with_origin(raw, x, SCHEMA)
        for r, b in zip(raw, batch):
            assert verify_physical_feasibility(x, b, SCHEMA)
            assert np.allclose(project(r, x), b, rtol=1e-9)


def test_segmentation_never_exceeds_payload_and_window_skips_udp():
    cfg = FFEv2Config(window_values=(8192, 65535))
    x = project(np.array([53, 17, 2, 1, 5, 0, 1000, 0, 0, 0, 0, 16], float), None)
    moves = move_ladder(x, x, cfg)
    assert all(g != 'win' for g, _ in moves)
    assert all(a <= x[LENF] - x[FPKTS] for g, a in moves if g == 'seg')


def test_logit_is_monotone_and_finite():
    z = logit(np.array([0.0, 1e-9, 0.3, 0.5, 0.999999, 1.0]))
    assert np.all(np.isfinite(z)) and np.all(np.diff(z) >= 0)


def test_benign_window_candidates_are_the_most_frequent_benign_tcp_windows():
    X = np.zeros((10, 12))
    X[:, 1] = 6
    X[:, 7] = [100, 100, 100, 200, 200, 300, 400, 100, 100, 200]
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert benign_window_candidates(X, y, 2) == (100, 200)
    assert benign_window_candidates(X, y, 0) == ()


# ---- packet-level semantics ------------------------------------------------------------------
def test_resegment_splits_payload_without_changing_totals():
    x = RECS[RECS[:, 1] == 6][0]
    t = trace_from_record(x, np.random.default_rng(0))
    t2, unmet = resegment(t, 4, 20)
    f0, f1 = extract_features(t), extract_features(t2)
    assert unmet == 0 and f1[FPKTS] == f0[FPKTS] + 4
    assert f1[LENF] == f0[LENF] and f1[DUR] == f0[DUR] and f1[WIN] == f0[WIN] and f1[HDR] == f0[HDR] + 80
    assert t2.fwd[0] and (t2.payload >= 0).all()


def test_resegment_reports_unmet_splits_for_one_byte_payloads():
    x = project(np.array([80, 6, 3, 1, 3, 0, 1000, 8192, 0, 0, 0, 60], float), None)
    _, unmet = resegment(trace_from_record(x, np.random.default_rng(0)), 5, 20)
    assert unmet == 5


@pytest.mark.parametrize('mode', CHAFF_MODES[:2])
@pytest.mark.parametrize('i', range(30))
def test_ledger_replay_reproduces_generator_features(i, mode):
    x = RECS[i]
    rng = np.random.RandomState(i)
    steps = [('time', int(x[DUR] * rng.uniform(0.2, 3))), ('pad', int(rng.randint(1, 200))),
             ('chaff', int(rng.randint(0, 6))), ('seg', int(rng.randint(0, 5)))]
    adv, ledger = x.copy(), Ledger()
    for g, a in steps:
        if a <= 0:
            continue
        nxt = project(apply_generator(adv, g, a, x), x)
        delta = {'time': nxt[DUR] - adv[DUR], 'pad': nxt[LENF] - adv[LENF],
                 'chaff': nxt[FPKTS] - adv[FPKTS], 'seg': nxt[FPKTS] - adv[FPKTS]}[g]
        ledger, adv = ledger.add(g, int(round(delta))), nxt
    window = None
    if x[1] == 6:
        adv, window = project(apply_generator(adv, 'win', 65535, x), x), 65535
    plan = plan_from_ledger(x, ledger.delay_us, ledger.n_chaff, ledger.n_segment, ledger.pad_bytes,
                            ledger.header_option_bytes, window, mode)
    out = replay_plan(x, plan, seed=1)
    if out['segment_unmet'] == 0 and out['pad_unmet_bytes'] == 0:
        assert relative_error(adv, out['features']).max() < 1e-9


# ---- attack ----------------------------------------------------------------------------------
class CountingModel:
    def __init__(self, inner):
        self.inner, self.rows = inner, 0

    def predict_proba(self, X):
        self.rows += len(np.atleast_2d(X))
        return self.inner.predict_proba(X)

    def get_split_thresholds(self, names=None):
        return self.inner.get_split_thresholds(names)


@pytest.mark.parametrize('snap', [False, True])
def test_attack_is_feasible_counts_queries_and_ledger_replays(setup, snap):
    model, mal, X_tr, y_tr, stds = setup
    wrapped = CountingModel(model)
    cfg = FFEv2Config(window_values=benign_window_candidates(X_tr, y_tr, 4), snap=snap)
    atk = FFEv2Attack(wrapped, SCHEMA, cfg, stds)
    wins = 0
    for x in mal[:25]:
        before = wrapped.rows
        res = atk.attack_sample(x)
        assert wrapped.rows - before == res.queries_used
        assert res.physically_valid
        assert (model.predict_proba(res.x_adversarial)[0, 1] < 0.5) == res.success
        wins += res.success
        led = atk.last_ledger
        plan = plan_from_ledger(x, led.delay_us, led.n_chaff, led.n_segment, led.pad_bytes,
                                led.header_option_bytes, led.window, 'empty_compensated')
        out = replay_plan(x, plan, seed=2)
        if out['segment_unmet'] == 0 and out['pad_unmet_bytes'] == 0 and led.header_option_bytes == 0:
            assert relative_error(res.x_adversarial, out['features']).max() < 1e-6
    assert wins >= 10


def test_already_benign_flow_costs_one_query(setup):
    model, _, X_tr, y_tr, stds = setup
    benign = X_tr[y_tr == 0][0]
    atk = FFEv2Attack(CountingModel(model), SCHEMA, FFEv2Config(), stds)
    res = atk.attack_sample(benign)
    assert res.success and res.queries_used == 1 and atk.last_ledger == Ledger()


def test_snap_moves_cross_the_named_threshold(setup):
    model, mal, X_tr, y_tr, stds = setup
    atk = FFEv2Attack(model, SCHEMA, FFEv2Config(snap=True, snap_per_feature=4,
                                                 window_values=(0, 8192, 65535)), stds)
    x = mal[0]
    st = _State(project(x, x), 0.0, Ledger())
    moves = atk._snap_moves(st, x)
    assert moves
    thresholds = {}
    for j, t in model.get_split_thresholds():
        thresholds.setdefault(j, set()).add(t)
    crossed = 0
    for g, a in moves:
        row = project(apply_generator(st.x, g, a, x), x)
        if any(row[j] != st.x[j] and any((st.x[j] < t) != (row[j] < t) for t in thresholds.get(j, ()))
               for j in (FPKTS, LENF, DUR, WIN, 8, 9, 10, HDR)):
            crossed += 1
    assert crossed >= 0.6 * len(moves)
