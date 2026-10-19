"""Tests for the white-box leaf-tuple-style tree attack."""

import numpy as np
import pytest

import xgboost as xgb  # noqa: F401  (must be imported before anything pulling in torch)

from src.attacks.whitebox_tree import (
    TreeDump, WhiteBoxLeafTupleAttack, project_batch_with_origin,
)
from src.datasets.schema import (
    get_cicids_schema, project_network_invariants, project_network_invariants_batch,
    verify_physical_feasibility,
)


class _Wrapper:
    """Minimal stand-in for NIDSEnsembleTrainer."""

    def __init__(self, clf):
        self.model = clf

    def predict_proba(self, X):
        return self.model.predict_proba(np.atleast_2d(X).astype(np.float32))


def _toy_flows(n, rng):
    X = np.zeros((n, 12))
    X[:, 0] = rng.choice([22, 80, 443], n)
    X[:, 1] = 6
    X[:, 2] = rng.integers(1, 40, n)
    X[:, 3] = rng.integers(0, 40, n)
    X[:, 4] = X[:, 2] * rng.integers(0, 1400, n)
    X[:, 5] = X[:, 3] * rng.integers(0, 1400, n)
    X[:, 6] = rng.integers(1000, 5_000_000, n)
    X[:, 7] = rng.integers(0, 65535, n)
    X[:, 11] = X[:, 2] * 20
    return project_network_invariants_batch(X)


@pytest.fixture(scope='module')
def toy():
    rng = np.random.default_rng(0)
    X = _toy_flows(3000, rng)
    # malicious = many small forward packets over a short window
    y = ((X[:, 2] > 12) & (X[:, 6] < 2_000_000)).astype(int)
    clf = xgb.XGBClassifier(n_estimators=25, max_depth=4, tree_method='hist', device='cpu',
                            random_state=0).fit(X.astype(np.float32), y)
    model = _Wrapper(clf)
    mal = X[(y == 1) & (model.predict_proba(X)[:, 1] >= 0.5)][:30]
    return model, mal, X


def test_tree_dump_margin_matches_xgboost(toy):
    model, _, X = toy
    dump = TreeDump(model.model)
    p_dump = 1.0 / (1.0 + np.exp(-dump.margin(X[:500])))
    np.testing.assert_allclose(p_dump, model.predict_proba(X[:500])[:, 1], atol=1e-5)


@pytest.mark.parametrize('in_loop', [False, True])
def test_attack_lowers_score_and_stays_on_manifold(toy, in_loop):
    model, mal, _ = toy
    schema = get_cicids_schema()
    atk = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=in_loop)
    dump = atk.dump
    wins = 0
    for x in mal:
        res = atk.attack_sample(x)
        reproj = project_network_invariants(res.x_adversarial, schema, x)
        np.testing.assert_allclose(res.x_adversarial, reproj, rtol=1e-6, atol=1e-6)
        assert verify_physical_feasibility(x, res.x_adversarial, schema) == res.physically_valid
        assert dump.margin(res.x_adversarial)[0] <= dump.margin(x)[0] + 1e-9 or not res.success
        wins += res.success
    assert wins > 0


def test_in_loop_projection_never_loses_to_projection_at_end_on_feasibility(toy):
    model, mal, _ = toy
    schema = get_cicids_schema()
    atk = WhiteBoxLeafTupleAttack(model, schema, project_in_loop=True)
    for x in mal[:10]:
        res = atk.attack_sample(x)
        if res.success:
            assert res.physically_valid


def test_attack_is_deterministic(toy):
    model, mal, _ = toy
    schema = get_cicids_schema()
    a = WhiteBoxLeafTupleAttack(model, schema)
    b = WhiteBoxLeafTupleAttack(model, schema)
    for x in mal[:10]:
        r1, r2 = a.attack_sample(x), b.attack_sample(x)
        np.testing.assert_array_equal(r1.x_adversarial, r2.x_adversarial)
        assert (r1.success, r1.queries_used) == (r2.success, r2.queries_used)


def test_batch_projection_with_origin_matches_scalar(toy):
    _, mal, X = toy
    schema = get_cicids_schema()
    rng = np.random.default_rng(1)
    x0 = mal[0]
    cand = np.abs(x0 + rng.standard_normal((50, 12)) * np.maximum(np.abs(x0), 1.0))
    batch = project_batch_with_origin(cand, x0, schema)
    for c, b in zip(cand, batch):
        np.testing.assert_allclose(project_network_invariants(c, schema, x0), b, rtol=1e-9, atol=1e-9)
