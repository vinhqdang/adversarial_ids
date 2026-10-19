"""Library-agnostic white-box threshold-crossing attack (adaptive attacker for any tree ensemble).

Same algorithm as ``src/attacks/whitebox_tree.WhiteBoxLeafTupleAttack`` (greedy single-feature moves
across split thresholds that the attacker reads from the trees, every candidate projected onto the
flow manifold in the loop), but the ensemble margin comes from the library's raw score rather than a
hand-written XGBoost tree walker, so it works for XGBoost, LightGBM, CatBoost and random forests,
for models with hidden columns, and for fused models. Crossing values are the threshold and one
step either side of it, so both split conventions (x < t and x <= t) are crossed.
"""

import numpy as np

from src.attacks.whitebox_tree import WhiteBoxLeafTupleAttack


class ThresholdView:
    """Adapter that gives a model the (thresholds, margin) interface the white-box attack reads."""

    def __init__(self, model):
        self.model = model
        self.thresholds = model.thresholds_by_feature()

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.model.margin(X)


class GenericWhiteBoxAttack(WhiteBoxLeafTupleAttack):
    def __init__(self, model, schema, max_steps: int = 20, project_in_loop: bool = True):
        super().__init__(model, schema, max_steps=max_steps, project_in_loop=project_in_loop,
                         tree_dump=ThresholdView(model))

    def _candidates(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for j, ths in enumerate(self.dump.thresholds):
            if len(ths) == 0:
                continue
            th = ths.astype(np.float64)
            eps = np.maximum(1e-6 * np.abs(th), 1e-6)
            vals = np.concatenate([th, th - eps, th + eps])
            vals = np.unique(vals[(vals >= 0) & (vals != x[j])])
            if len(vals) == 0:
                continue
            cand = np.repeat(x[None, :], len(vals), axis=0)
            cand[:, j] = vals
            rows.append(cand)
        return np.concatenate(rows) if rows else np.empty((0, len(x)))
