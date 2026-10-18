"""Shared full-data loading, cohort selection and model wrappers for the experiment scripts."""

import os
import sys
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.full_data import FEATURES, build_full_dataset, stratified_split
from src.datasets.schema import project_network_invariants_batch

SEED = 42
XGB_CFG = {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
           'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED}


def load_arrays(name: str, seed: int = SEED) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Full deduplicated dataset -> {'train'|'val'|'test': (X, y, family)}, projected onto M_flow."""
    splits = stratified_split(build_full_dataset(name), seed=seed)
    return {k: (project_network_invariants_batch(p[FEATURES].values.astype(float)),
                p['label'].values.astype(int), p['attack_family'].values)
            for k, p in splits.items()}


def detected_by_all(models: Union[object, Sequence], X: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows that every model classifies as malicious."""
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    mask = np.ones(len(X), dtype=bool)
    for m in models:
        mask &= m.predict_proba(X)[:, 1] >= 0.5
    return mask


def detected_cohort(models, X, y, fam, n_per_family: int, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Malicious flows detected by every given model, up to ``n_per_family`` per attack family."""
    rng = np.random.RandomState(seed)
    mal = np.where(y == 1)[0]
    det = mal[detected_by_all(models, X[mal])]
    picks = []
    for f in sorted(set(fam[det])):
        idx = det[fam[det] == f]
        picks.append(rng.choice(idx, size=min(n_per_family, len(idx)), replace=False))
    sel = np.concatenate(picks)
    return X[sel], fam[sel]


class ColumnMaskedModel:
    """A tree model that never sees the ``masked`` columns (the alpha -> infinity limit of MPS).

    The attacker still works on, and perturbs, the full 12-feature flow; the model simply ignores
    the masked features. Split thresholds are reported in the original feature indices so that
    tree-split boundary snapping keeps working.
    """

    def __init__(self, trainer, masked: Sequence[int], n_features: int = len(FEATURES)):
        self.trainer = trainer
        self.keep = [j for j in range(n_features) if j not in set(masked)]

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trainer.train(X[:, self.keep], y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.trainer.predict_proba(np.atleast_2d(X)[:, self.keep])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self.trainer.evaluate(X[:, self.keep], y)

    def get_split_thresholds(self, feature_names: List[str] = None):
        return [(self.keep[f], t) for f, t in self.trainer.get_split_thresholds()]
