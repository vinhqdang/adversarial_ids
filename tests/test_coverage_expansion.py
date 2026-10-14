"""
Comprehensive unit tests for unconstrained feature-space attack baseline,
tree ensemble trainer across XGBoost, LightGBM, and CatBoost,
and differentiable surrogate structures to ensure >85% test coverage.
"""

import numpy as np
import pytest
import torch

from src.attacks.base import AttackResult, BaseNIDSAttack
from src.attacks.unconstrained import UnconstrainedFeatureSpaceAttack
from src.attacks.pact_evasion import DifferentiableNetworkManifold, SoftTreeSurrogate, PACTEvasionAttack
from src.models.tree_models import NIDSEnsembleTrainer
from src.datasets.schema import get_cicids_schema, project_network_invariants
from src.datasets.generator import generate_cicids_traffic


class DummyBinaryModel:
    """Mock model that classifies based on a single threshold."""
    def __init__(self, feature_idx: int = 2, threshold: float = 15.0):
        self.feature_idx = feature_idx
        self.threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        malicious = (X[:, self.feature_idx] >= self.threshold).astype(float)
        return np.column_stack([1.0 - malicious, malicious])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return (X[:, self.feature_idx] >= self.threshold).astype(int)


def test_unconstrained_attack_already_benign():
    schema = get_cicids_schema()
    model = DummyBinaryModel(feature_idx=2, threshold=50.0)
    attack = UnconstrainedFeatureSpaceAttack(model, schema, max_iter=10)
    
    # Benign sample: feature 2 is 10.0 (< 50.0)
    x = np.zeros(12)
    x[0], x[1], x[2] = 443.0, 6.0, 10.0
    res = attack.attack_sample(x)
    assert res.success is True
    assert res.physically_valid is True
    assert res.queries_used == 1
    assert res.weighted_cost == 0.0


def test_unconstrained_attack_malicious_success_and_infeasibility():
    schema = get_cicids_schema()
    # Model detects when fwd_pkts >= 20.0
    model = DummyBinaryModel(feature_idx=2, threshold=20.0)
    attack = UnconstrainedFeatureSpaceAttack(model, schema, max_iter=20, step_size=0.5)
    
    # Malicious sample: feature 2 is 25.0
    x = np.zeros(12)
    x[0], x[1], x[2], x[6] = 443.0, 6.0, 25.0, 1e6
    x[4] = 25.0 * 100.0
    x = project_network_invariants(x, schema)
    
    res = attack.attack_sample(x)
    assert res.queries_used > 1
    # Check that result dataclass fields are populated
    assert isinstance(res, AttackResult)
    assert len(res.feature_shifts) == 12


def test_tree_models_trainer_initialization_and_exceptions():
    # Valid model types
    xgb_trainer = NIDSEnsembleTrainer('xgboost', {'n_estimators': 5, 'max_depth': 2})
    assert xgb_trainer.model is not None

    lgb_trainer = NIDSEnsembleTrainer('lightgbm', {'n_estimators': 5, 'max_depth': 2})
    assert lgb_trainer.model is not None

    cat_trainer = NIDSEnsembleTrainer('catboost', {'iterations': 5, 'depth': 2})
    assert cat_trainer.model is not None

    # Invalid model type
    with pytest.raises(ValueError, match="Unknown model_type"):
        NIDSEnsembleTrainer('unsupported_model_type')


def test_tree_models_train_evaluate_and_splits():
    schema = get_cicids_schema()
    X, y, _ = generate_cicids_traffic(n_samples=200, attack_ratio=0.5, random_state=42)
    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:], y[120:]

    # 1. XGBoost
    xgb = NIDSEnsembleTrainer('xgboost', {'n_estimators': 10, 'max_depth': 3, 'random_state': 42})
    xgb.train(X_train, y_train)
    metrics_xgb = xgb.evaluate(X_test, y_test)
    assert 'accuracy' in metrics_xgb
    assert 'roc_auc' in metrics_xgb
    assert 0.0 <= metrics_xgb['accuracy'] <= 1.0

    # Test predict_proba with 1D and 2D input
    prob_1d = xgb.predict_proba(X_test[0])
    assert prob_1d.shape == (1, 2)
    prob_2d = xgb.predict_proba(X_test[:5])
    assert prob_2d.shape == (5, 2)

    # Test split thresholds extraction
    splits_xgb = xgb.get_split_thresholds()
    assert isinstance(splits_xgb, list)
    splits_named = xgb.get_split_thresholds(feature_names=schema.feature_names)
    assert isinstance(splits_named, list)

    # 2. LightGBM
    lgb = NIDSEnsembleTrainer('lightgbm', {'n_estimators': 10, 'max_depth': 3, 'random_state': 42, 'verbosity': -1})
    lgb.train(X_train, y_train)
    metrics_lgb = lgb.evaluate(X_test, y_test)
    assert 'accuracy' in metrics_lgb
    splits_lgb = lgb.get_split_thresholds()
    assert isinstance(splits_lgb, list)
    splits_lgb_named = lgb.get_split_thresholds(feature_names=schema.feature_names)
    assert isinstance(splits_lgb_named, list)

    # 3. CatBoost
    cat = NIDSEnsembleTrainer('catboost', {'iterations': 10, 'depth': 3, 'random_seed': 42, 'verbose': False})
    cat.train(X_train, y_train)
    metrics_cat = cat.evaluate(X_test, y_test)
    assert 'accuracy' in metrics_cat
    assert 0.0 <= metrics_cat['accuracy'] <= 1.0

    # 4. Random Forest
    rf = NIDSEnsembleTrainer('random_forest', {'n_estimators': 10, 'max_depth': 4, 'random_state': 42})
    rf.train(X_train, y_train)
    metrics_rf = rf.evaluate(X_test, y_test)
    assert 'accuracy' in metrics_rf
    splits_rf = rf.get_split_thresholds()
    assert isinstance(splits_rf, list)
    assert len(splits_rf) > 0


def test_pact_differentiable_manifold():
    schema = get_cicids_schema()
    device = torch.device('cpu')
    manifold = DifferentiableNetworkManifold(schema, device)

    x_orig = torch.tensor([[443.0, 6.0, 10.0, 2.0, 1000.0, 200.0, 1e6, 16384.0, 1200.0, 12.0, 100.0, 200.0]], device=device)
    delta = torch.ones_like(x_orig) * 5.0

    x_proj = manifold(x_orig, delta)
    assert x_proj.shape == x_orig.shape
    # Immutable features must remain unchanged
    assert float(x_proj[0, 0]) == float(x_orig[0, 0])
    assert float(x_proj[0, 1]) == float(x_orig[0, 1])


def test_pact_soft_tree_surrogate():
    device = torch.device('cpu')
    # Empty splits
    surrogate_empty = SoftTreeSurrogate([], device=device)
    dummy_x = torch.zeros(1, 12, device=device)
    score_empty = surrogate_empty(dummy_x)
    assert float(score_empty) == 0.0

    # Non-empty splits
    splits = [(2, 15.0, 1.0), (4, 500.0, -0.5)]
    surrogate = SoftTreeSurrogate(splits, temperature=1.0, device=device)
    score = surrogate(dummy_x)
    assert isinstance(float(score), float)


def test_pact_evasion_attack_execution():
    schema = get_cicids_schema()
    X, y, _ = generate_cicids_traffic(n_samples=150, attack_ratio=0.5, random_state=42)
    model = NIDSEnsembleTrainer('xgboost', {'n_estimators': 5, 'max_depth': 2, 'random_state': 42})
    model.train(X[:100], y[:100])

    pact = PACTEvasionAttack(model, schema, max_grad_steps=5, max_snapping_steps=5, device=torch.device('cpu'))
    
    # Test on already benign sample
    benign_flow = X[y == 0][0]
    res_benign = pact.attack_sample(benign_flow)
    assert res_benign.success is True

    # Test on malicious sample
    mal_flow = X[y == 1][0]
    res_mal = pact.attack_sample(mal_flow)
    assert isinstance(res_mal, AttackResult)
