"""Manipulation-signature detector (feasibility-anomaly idea), fused with the classifier by OR.

A small tree ensemble is trained to separate natural flows from flows that went through random
compositions of the four physical actions (``orbit_augment``), using only the 12 schema features.
It is calibrated on VALIDATION so that at most ``max_fpr`` of natural benign flows are flagged, and
its margin is shifted so that it fires at that operating point. The false-positive cost is therefore
bounded by construction and reported by the experiment.
"""

from typing import List, Optional

import numpy as np

from src.defenses.augment import orbit_augment
from src.defenses.models import N_FEATURES, RobustTreeModel


class ManipulationDetector:
    def __init__(self, library: str = 'xgboost', max_fpr: float = 0.002, n_train: int = 400_000,
                 seed: int = 42):
        self.library, self.max_fpr, self.n_train, self.seed = library, max_fpr, n_train, seed
        self.model: Optional[RobustTreeModel] = None
        self.offset = 0.0
        self.train_seconds = 0.0

    def fit(self, X: np.ndarray, X_val_benign: np.ndarray) -> 'ManipulationDetector':
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(X), min(self.n_train, len(X)), replace=False)
        nat = X[idx]
        pool = X[X[:, 1] == 6, 7][:100000]
        half = len(nat) // 2
        data = np.vstack([nat[:half], orbit_augment(nat[half:], rng, win_pool=pool)])
        label = np.concatenate([np.zeros(half, dtype=int), np.ones(len(nat) - half, dtype=int)])
        self.model = RobustTreeModel(self.library).train(data, label)
        scores = self.model.margin(X_val_benign)
        self.offset = float(np.quantile(scores, 1.0 - self.max_fpr))
        self.train_seconds = self.model.train_seconds
        return self

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.model.margin(X) - self.offset

    def get_split_thresholds(self, feature_names=None) -> List:
        return self.model.get_split_thresholds(feature_names)

    def set_predict_threads(self, n: int) -> None:
        self.model.set_predict_threads(n)
