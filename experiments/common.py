"""Shared full-data loading and cohort selection for the experiment scripts."""

import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.full_data import FEATURES, build_full_dataset, stratified_split
from src.datasets.schema import project_network_invariants_batch

SEED = 42
XGB_CFG = {'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.1,
           'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED}


def load_arrays(name: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Full deduplicated dataset -> {'train'|'val'|'test': (X, y, family)}, projected onto M_flow."""
    splits = stratified_split(build_full_dataset(name), seed=SEED)
    return {k: (project_network_invariants_batch(p[FEATURES].values.astype(float)),
                p['label'].values.astype(int), p['attack_family'].values)
            for k, p in splits.items()}


def detected_cohort(model, X, y, fam, n_per_family: int, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Detected malicious flows, up to ``n_per_family`` per attack family (balanced)."""
    rng = np.random.RandomState(seed)
    mal = np.where(y == 1)[0]
    det = mal[model.predict_proba(X[mal])[:, 1] >= 0.5]
    picks = []
    for f in sorted(set(fam[det])):
        idx = det[fam[det] == f]
        picks.append(rng.choice(idx, size=min(n_per_family, len(idx)), replace=False))
    sel = np.concatenate(picks)
    return X[sel], fam[sel]
