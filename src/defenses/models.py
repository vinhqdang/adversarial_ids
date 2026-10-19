"""Library-agnostic wrapper for tree ensembles used by the hardening recipe.

One class (``RobustTreeModel``) trains XGBoost, LightGBM, CatBoost or a random forest with the
same recipe knobs: hidden columns (feature-set hardening), monotone constraints on the columns an
attacker can only increase, and per-sample weights. It exposes the interface the attacks expect
(``predict_proba``, ``get_split_thresholds`` in ORIGINAL 12-feature indices) plus the raw ensemble
margin that a white-box attacker reads.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.tree_models import NIDSEnsembleTrainer  # imports xgboost/lightgbm before torch

N_FEATURES = 12
LIBRARIES = ('xgboost', 'lightgbm', 'catboost', 'random_forest')
SEED = 42

BASE_PARAMS: Dict[str, dict] = {
    'xgboost': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1, 'tree_method': 'hist',
                'device': 'cuda', 'random_state': SEED},
    'lightgbm': {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': SEED,
                 'n_jobs': 8},
    'catboost': {'iterations': 60, 'depth': 6, 'learning_rate': 0.1, 'verbose': False,
                 'random_seed': SEED, 'thread_count': 8},
    'random_forest': {'n_estimators': 100, 'max_depth': 12, 'random_state': SEED, 'n_jobs': 8},
}


# Monotone-safe engineered columns: each is a ratio that is non-increasing in exactly one coordinate
# the attacker can only increase (fwd packets, fwd bytes, duration) and unaffected by the others, so
# a score that is non-increasing in it is non-decreasing in that coordinate.  (index, sign)
EXTRA_SIGN = {12: -1, 13: -1, 14: -1, 15: -1, 16: -1}
ALL_EXTRAS = tuple(EXTRA_SIGN)


def extra_columns(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    dur_s = X[:, 6] / 1e6 + 1e-3
    return np.column_stack([X[:, 3] / np.maximum(1.0, X[:, 2]), X[:, 5] / (X[:, 4] + 1.0),
                            X[:, 5] / np.maximum(1.0, X[:, 2]), X[:, 3] / dur_s, X[:, 5] / dur_s])


def _sigmoid_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p))}


class RobustTreeModel:
    """Tree ensemble with hidden columns and monotone constraints (original feature indices)."""

    def __init__(self, library: str, hidden: Sequence[int] = (), monotone: Optional[Dict[int, int]] = None,
                 params: Optional[dict] = None, extras: Sequence[int] = ()):
        if library not in LIBRARIES:
            raise ValueError(f'unknown library {library!r}')
        self.library = library
        self.hidden = tuple(sorted(hidden))
        self.extras = tuple(extras)
        self.keep = [j for j in range(N_FEATURES) if j not in set(self.hidden)]
        self.columns = self.keep + list(self.extras)
        mono = dict(monotone or {})
        mono.update({j: EXTRA_SIGN[j] for j in self.extras if monotone})
        self.monotone = {j: s for j, s in mono.items() if j in self.columns}
        self.params = {**BASE_PARAMS[library], **(params or {})}
        self.train_seconds = 0.0
        self.predict_threads = 0
        self.trainer: Optional[NIDSEnsembleTrainer] = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.extras:
            return X[:, self.keep]
        E = extra_columns(X)[:, [j - N_FEATURES for j in self.extras]]
        return np.hstack([X[:, self.keep], E])

    def _model_params(self) -> dict:
        p = dict(self.params)
        if not self.monotone:
            return p
        vec = [int(self.monotone.get(j, 0)) for j in self.columns]
        if self.library == 'xgboost':
            p['monotone_constraints'] = tuple(vec)
        elif self.library == 'lightgbm':
            p.update(monotone_constraints=vec, monotone_constraints_method='intermediate')
        elif self.library == 'catboost':
            p['monotone_constraints'] = vec
        else:
            p['monotonic_cst'] = vec
        return p

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'RobustTreeModel':
        t0 = time.time()
        self.trainer = NIDSEnsembleTrainer(self.library, self._model_params())
        Xk = self._design(X)
        if sample_weight is None:
            self.trainer.model.fit(Xk, y)
        else:
            self.trainer.model.fit(Xk, y, sample_weight=sample_weight)
        if self.library == 'xgboost':
            self.trainer.model.get_booster().set_param({'device': 'cpu'})  # one device for metrics and attacks
        self.train_seconds = time.time() - t0
        self.__dict__.pop('_cached_splits', None)
        return self

    def set_predict_threads(self, n: int) -> None:
        """Threads used at inference (1 for single-flow attack queries, many for batch metrics)."""
        self.predict_threads = n
        if self.library == 'xgboost':
            self.trainer.model.get_booster().set_param({'nthread': n})
        elif self.library in ('lightgbm', 'random_forest'):
            self.trainer.model.set_params(n_jobs=n)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xk = self._design(X)
        if self.library == 'catboost' and self.predict_threads:
            return self.trainer.model.predict_proba(Xk, thread_count=self.predict_threads)
        return self.trainer.predict_proba(Xk)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return _metrics(y, self.predict_proba(X)[:, 1])

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Raw ensemble score (log-odds for boosting; probability logit for the forest)."""
        Xk = self._design(X)
        m = self.trainer.model
        if self.library == 'xgboost':
            return np.asarray(m.predict(Xk, output_margin=True), dtype=float)
        if self.library == 'lightgbm':
            return np.asarray(m.predict(Xk, raw_score=True), dtype=float)
        if self.library == 'catboost':
            return np.asarray(m.predict(Xk, prediction_type='RawFormulaVal',
                                        thread_count=self.predict_threads or -1), dtype=float)
        return _sigmoid_logit(m.predict_proba(Xk)[:, 1])

    def get_split_thresholds(self, feature_names=None) -> List[Tuple[int, float]]:
        return [(self.columns[f], t) for f, t in self.trainer.get_split_thresholds()
                if self.columns[f] < N_FEATURES]

    def thresholds_by_feature(self) -> List[np.ndarray]:
        splits = self.get_split_thresholds()
        return [np.unique(np.array([t for j, t in splits if j == f], dtype=np.float32))
                for f in range(N_FEATURES)]


class OffsetMargin:
    """A model whose decision margin is shifted so that it fires at a chosen score threshold."""

    def __init__(self, model, offset: float):
        self.model, self.offset = model, float(offset)

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.model.margin(X) - self.offset

    def get_split_thresholds(self, feature_names=None):
        return self.model.get_split_thresholds(feature_names)

    @property
    def train_seconds(self) -> float:
        return self.model.train_seconds


class FusedModel:
    """OR-fusion of components: margin = max over components (flagged iff any component flags)."""

    def __init__(self, *components):
        self.components = components

    def margin(self, X: np.ndarray) -> np.ndarray:
        return np.max([c.margin(X) for c in self.components], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(self.margin(np.atleast_2d(X)), -50, 50)))
        return np.column_stack([1 - p, p])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return _metrics(y, self.predict_proba(X)[:, 1])

    def set_predict_threads(self, n: int) -> None:
        for c in self.components:
            inner = getattr(c, 'model', c)
            if hasattr(inner, 'set_predict_threads'):
                inner.set_predict_threads(n)
            elif hasattr(c, 'set_predict_threads'):
                c.set_predict_threads(n)

    def get_split_thresholds(self, feature_names=None):
        return [t for c in self.components for t in c.get_split_thresholds(feature_names)]

    def thresholds_by_feature(self) -> List[np.ndarray]:
        splits = self.get_split_thresholds()
        return [np.unique(np.array([t for j, t in splits if j == f], dtype=np.float32))
                for f in range(N_FEATURES)]

    @property
    def train_seconds(self) -> float:
        return float(sum(c.train_seconds for c in self.components))
