"""Shared setup for the FFE-v2 experiments: models identical to the main benchmark, cohorts from the
validation split (tuning) or the test split (final), and a pickle cache of trained models."""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import xgboost  # noqa: F401,E402  (before torch)
import lightgbm  # noqa: F401,E402
from common import load_arrays, detected_by_all  # noqa: E402
from src.datasets.schema import get_cicids_schema  # noqa: E402
from src.models.tree_models import NIDSEnsembleTrainer  # noqa: E402

SEED = 42
MODEL_CONFIGS = {
    'xgboost': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
                'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED},
    'lightgbm': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': SEED},
    # the CatBoost configuration of experiments/run_transferability.py
    'catboost': {'iterations': 60, 'depth': 6, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': False},
}
CACHE = os.environ.get('FFEV2_CACHE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ffev2_cache')))


def get_models(name: str):
    """(data, trained {xgboost, lightgbm}); trained on the train split exactly like run_full_benchmark."""
    data = load_arrays(name)
    path = os.path.join(CACHE, f'models_{name}.pkl')
    if os.path.exists(path):
        trained = pickle.load(open(path, 'rb'))
        trained['xgboost'].model.get_booster().set_param({'device': 'cpu'})   # the pickle restores cuda
    else:
        X_tr, y_tr, _ = data['train']
        trained = {}
        for m_type, cfg in MODEL_CONFIGS.items():
            t = NIDSEnsembleTrainer(m_type, cfg)
            t.train(X_tr, y_tr)
            if m_type == 'xgboost':
                t.model.get_booster().set_param({'device': 'cpu'})
            trained[m_type] = t
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump(trained, open(path, 'wb'))
    return data, trained


def pick_cohort(models, X, y, fam, n: int, seed: int = SEED):
    """Identical to run_full_benchmark.pick_cohort (flows detected by all models)."""
    rng = np.random.RandomState(seed)
    mal = np.where(y == 1)[0]
    detected = mal[detected_by_all(models, X[mal])]
    take = rng.choice(len(detected), size=min(n, len(detected)), replace=False)
    idx = detected[take]
    return idx, X[idx], fam[idx]


class SingleThread:
    """Wraps a trainer so that scoring uses one thread (many worker processes share the machine)."""

    def __init__(self, trainer):
        self.trainer = trainer
        inner = trainer.model
        if trainer.model_type == 'catboost':
            self._kw = {'thread_count': 1}
        else:
            self._kw = {}
            inner.set_params(n_jobs=1)

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.trainer.model.predict_proba(X, **self._kw)

    def get_split_thresholds(self, feature_names=None):
        return self.trainer.get_split_thresholds(feature_names)


def load_trained(name: str, single_thread: bool = False):
    """Trained models only (no data load); requires ``get_models(name)`` to have been run once."""
    trained = pickle.load(open(os.path.join(CACHE, f'models_{name}.pkl'), 'rb'))
    trained['xgboost'].model.get_booster().set_param({'device': 'cpu'})
    return {k: SingleThread(v) for k, v in trained.items()} if single_thread else trained


def cohorts_for(trained, X, y, fam, n: int):
    """Main-benchmark cohort (flows detected by XGBoost and LightGBM) and, per target, the sub-cohort
    that the target itself detects (only those can be genuine evasions of that target)."""
    idx, Xc, fc = pick_cohort([trained['xgboost'], trained['lightgbm']], X, y, fam, n)
    out = {}
    for mt, model in trained.items():
        keep = model.predict_proba(Xc)[:, 1] >= 0.5
        out[mt] = {'idx': idx[keep], 'X': Xc[keep], 'family': fc[keep]}
    return out
