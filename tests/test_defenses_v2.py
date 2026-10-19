"""Tests for the v2 defense package (certified gate, augmentation, fusion, generic white-box, registry)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments')))

from src.defenses.models import FusedModel, OffsetMargin, RobustTreeModel  # noqa: E402 (xgboost first)
from src.datasets.schema import get_cicids_schema, project_network_invariants_batch, verify_physical_feasibility  # noqa: E402
from src.defenses.attackers import ATTACKS, AttackContext, register_attack, run_attack, wilson  # noqa: E402
from src.defenses.augment import iterative_adversarial_training, make_orbit_set, orbit_augment  # noqa: E402
from src.defenses.whitebox_generic import GenericWhiteBoxAttack  # noqa: E402

SCHEMA = get_cicids_schema()
CPU = {'xgboost': {'device': 'cpu', 'n_estimators': 30}, 'lightgbm': {'n_estimators': 30, 'n_jobs': 2},
       'catboost': {'iterations': 30, 'thread_count': 2}, 'random_forest': {'n_estimators': 20, 'n_jobs': 2}}
GATE_HIDDEN, GATE_MONO = (7, 8, 9, 10), {2: 1, 4: 1, 6: 1, 11: 1}


def synthetic_flows(n: int, seed: int = 0):
    """Valid flows; malicious = low backward traffic, or a short flow with few packets (movable features matter)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 12))
    X[:, 0] = rng.choice([22, 80, 443, 3389], n)
    X[:, 1] = rng.choice([6, 17], n, p=[0.9, 0.1])
    X[:, 2] = rng.randint(1, 40, n)
    X[:, 3] = rng.randint(0, 40, n)
    X[:, 4] = X[:, 2] * rng.randint(0, 600, n)
    X[:, 5] = X[:, 3] * rng.randint(0, 900, n)
    X[:, 6] = np.exp(rng.uniform(4, 16, n))
    X[:, 7] = np.where(X[:, 1] == 6, rng.choice([0, 8192, 29200, 64240], n), 0)
    X[:, 11] = X[:, 2] * 20
    X = project_network_invariants_batch(X)
    y = ((X[:, 3] < 8) | ((X[:, 6] < 3e4) & (X[:, 2] < 10))).astype(int)
    return X, y


@pytest.fixture(scope='module')
def data():
    return synthetic_flows(6000)


def test_orbit_augment_is_feasible_and_leaves_immutables(data):
    X, _ = data
    A = orbit_augment(X[:500], np.random.RandomState(1))
    assert np.array_equal(A[:, [0, 1, 3, 5]], X[:500, [0, 1, 3, 5]])
    assert all(verify_physical_feasibility(x, a, SCHEMA) for x, a in zip(X[:500], A))
    assert (A[:, [2, 4, 6, 11]] >= X[:500, [2, 4, 6, 11]]).all()
    assert (A != X[:500]).any(axis=1).mean() > 0.9        # almost every flow really changes


def test_make_orbit_set_balanced_and_labels_kept(data):
    X, y = data
    Xo, yo = make_orbit_set(X, y, 400, np.random.RandomState(2))
    assert len(Xo) == 400 and yo.sum() == 200


@pytest.mark.parametrize('library', ['xgboost', 'lightgbm', 'catboost', 'random_forest'])
def test_gate_score_never_decreases_under_feasible_transformations(data, library):
    """Certificate: monotone gate + hidden window/rates/mean => margin non-decreasing on the orbit."""
    X, y = data
    gate = RobustTreeModel(library, hidden=GATE_HIDDEN, monotone=GATE_MONO, params=CPU[library]).train(X, y)
    rng = np.random.RandomState(3)
    base = gate.margin(X[:800])
    worst = np.inf
    for _ in range(6):
        worst = min(worst, float(np.min(gate.margin(orbit_augment(X[:800], rng)) - base)))
    assert worst >= -1e-6


def test_unconstrained_model_can_lose_score_on_the_orbit(data):
    """Sanity: without the constraint the same transformations do lower scores (the test is not vacuous)."""
    X, y = data
    free = RobustTreeModel('lightgbm', params=CPU['lightgbm']).train(X, y)
    rng = np.random.RandomState(4)
    drop = free.margin(orbit_augment(X[:800], rng)) - free.margin(X[:800])
    assert drop.min() < -0.5


def test_gate_keeps_clean_accuracy_reasonable(data):
    X, y = data
    gate = RobustTreeModel('lightgbm', hidden=GATE_HIDDEN, monotone=GATE_MONO, params=CPU['lightgbm']).train(X[:5000], y[:5000])
    assert gate.evaluate(X[5000:], y[5000:])['accuracy'] > 0.8


def test_fused_model_is_or_of_components(data):
    X, y = data
    a = RobustTreeModel('lightgbm', params=CPU['lightgbm']).train(X, y)
    b = RobustTreeModel('lightgbm', hidden=GATE_HIDDEN, monotone=GATE_MONO, params=CPU['lightgbm']).train(X, y)
    fused = FusedModel(a, OffsetMargin(b, 1.0))
    expect = np.maximum(a.margin(X[:300]), b.margin(X[:300]) - 1.0)
    assert np.allclose(fused.margin(X[:300]), expect)
    flagged = fused.predict_proba(X[:300])[:, 1] >= 0.5
    assert np.array_equal(flagged, (a.margin(X[:300]) >= 0) | (b.margin(X[:300]) >= 1.0))
    assert fused.predict_proba(X[0]).shape == (1, 2)


def test_threshold_extraction_uses_original_indices(data):
    X, y = data
    m = RobustTreeModel('lightgbm', hidden=(7, 8, 9, 10), params=CPU['lightgbm']).train(X, y)
    used = {j for j, _ in m.get_split_thresholds()}
    assert used and used <= {0, 1, 2, 3, 4, 5, 6, 11}


@pytest.mark.parametrize('library', ['lightgbm', 'catboost', 'xgboost'])
def test_generic_whitebox_evades_free_model_but_not_gate(data, library):
    X, y = data
    free = RobustTreeModel(library, params=CPU[library]).train(X, y)
    gate = RobustTreeModel(library, hidden=GATE_HIDDEN, monotone=GATE_MONO, params=CPU[library]).train(X, y)
    pool = X[y == 1][:200]
    for model, expect_evasions in ((free, True), (gate, False)):
        det = pool[model.predict_proba(pool)[:, 1] >= 0.5][:60]
        atk = GenericWhiteBoxAttack(model, SCHEMA)
        real = sum(r.success and r.physically_valid for r in map(atk.attack_sample, det))
        assert (real > 0) == expect_evasions or not expect_evasions and real == 0


def test_attack_registry_plugin_and_run_attack(data):
    X, y = data

    class Noop:
        def __init__(self, model, ctx):
            self.model = model

        def attack_sample(self, x):
            from src.attacks.base import AttackResult
            return AttackResult(x, x, False, True, 1, 0.0, np.zeros_like(x))

    register_attack('noop_test', lambda model, ctx: Noop(model, ctx))
    try:
        model = RobustTreeModel('lightgbm', params=CPU['lightgbm']).train(X, y)
        ctx = AttackContext(SCHEMA, X.std(axis=0) + 1e-6)
        res = run_attack('noop_test', model, ctx, X[:5], n_jobs=1)
        assert len(res) == 5 and not any(r[0] for r in res) and np.array_equal(res[0][3], X[0])
    finally:
        ATTACKS.pop('noop_test')


def test_iterative_adversarial_training_adds_true_labelled_flows(data):
    X, y = data
    seen = {}

    def train_fn(a, b, w):
        seen['n'], seen['pos'] = len(a), int(b.sum())
        return RobustTreeModel('lightgbm', params=CPU['lightgbm']).train(a, b, w)

    def craft(model, Xm, r):
        return Xm.copy()

    _, log = iterative_adversarial_training(train_fn, X, y, craft, rounds=2, n_craft=100, n_orbit=200)
    assert len(log) == 2 and seen['n'] == len(X) + 200 + 200
    assert seen['pos'] >= int(y.sum()) + 200


def test_wilson_interval_brackets_proportion():
    lo, hi = wilson(30, 100)
    assert lo < 30 < hi and wilson(0, 0) == [0.0, 0.0]


def test_mcnemar_pairs_only_flows_detected_by_both():
    import run_defenses_v2 as r
    a = np.array([1, 1, 0, -1, 1, 0])
    b = np.array([0, 0, 0, 1, -1, 1])
    res = r.mcnemar(a, b)
    assert res['n_paired'] == 4 and res['only_first'] == 2 and res['only_second'] == 1
