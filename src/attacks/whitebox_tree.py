"""White-box leaf-tuple-style attack on an XGBoost ensemble (control for FFE).

This is NOT the official LT-Attack code (Zhang et al., NeurIPS 2020, chong-z/tree-ensemble-attack,
a Bazel/C++ tool that was not buildable in our environment). It is a re-implementation in the same
spirit: the attacker reads every tree, so the ensemble's decision function is a piecewise-constant
function over axis-aligned cells (tuples of leaves). Starting from the flow's own leaf tuple, each
step moves ONE feature to the far side of ONE split threshold of the ensemble (exactly to the
threshold, or one float32 ulp below it), choosing the move that lowers the exact ensemble margin
the most, until the margin is negative (flow classified benign) or no single move helps.
Differences to LT-Attack: greedy single-feature moves instead of its leaf-tuple search with
an L-inf norm objective and no optimality bound, and the margin, not the L-inf distance, drives the
choice.

Two feasibility treatments (both followed by the same schema projection as FFE):

* ``project_in_loop=False`` (default, "project at end"): the attack is an unconstrained feature-space
  white-box attack; its final flow is projected onto the flow manifold with
  ``project_network_invariants(x_adv, schema, x_orig)`` and success is re-measured on the
  projected flow. This is the fair "unconstrained attack + schema repair" control.
* ``project_in_loop=True``: every candidate move is projected before its margin is evaluated, so the
  search itself sees the manifold (a constrained white-box attacker).

Accounting: ``queries_used`` counts calls to the model's ``predict_proba`` (initial check and final
check of the returned flow). Reading the trees costs no queries; ``internal_evals`` counts the
candidate flows whose margin was computed from the tree dump (not queries, but work done with
white-box access).
"""

import json
from typing import List, Optional, Tuple

import numpy as np

from .base import AttackResult, BaseNIDSAttack
from ..datasets.schema import (
    NetworkFeatureSchema, project_network_invariants, project_network_invariants_batch,
    verify_physical_feasibility,
)


class TreeDump:
    """Padded array form of an XGBoost binary ensemble, evaluated in float32 like XGBoost."""

    def __init__(self, model, n_features: int = 12):
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        trees = [self._flatten(json.loads(t)) for t in booster.get_dump(dump_format='json')]
        n_nodes = max(len(t[0]) for t in trees)
        self.n_trees = len(trees)
        self.feat = np.full((self.n_trees, n_nodes), -1, dtype=np.int64)
        self.thr = np.zeros((self.n_trees, n_nodes), dtype=np.float32)
        self.left = np.zeros((self.n_trees, n_nodes), dtype=np.int64)
        self.right = np.zeros((self.n_trees, n_nodes), dtype=np.int64)
        self.leaf = np.zeros((self.n_trees, n_nodes), dtype=np.float64)
        for i, (f, th, le, ri, lf) in enumerate(trees):
            k = len(f)
            self.feat[i, :k], self.thr[i, :k], self.left[i, :k] = f, th, le
            self.right[i, :k], self.leaf[i, :k] = ri, lf
        self.max_depth = int(self._depth())
        self.n_features = n_features
        self.thresholds: List[np.ndarray] = [
            np.unique(self.thr[(self.feat == j)]) for j in range(n_features)]
        self.base_margin = self._calibrate(booster, n_features)

    @staticmethod
    def _flatten(root: dict) -> Tuple[list, list, list, list, list]:
        feat, thr, left, right, leaf = [], [], [], [], []

        def visit(node: dict) -> int:
            idx = len(feat)
            feat.append(-1), thr.append(0.0), left.append(idx), right.append(idx), leaf.append(0.0)
            if 'leaf' in node:
                leaf[idx] = float(node['leaf'])
                return idx
            feat[idx] = int(node['split'][1:])
            thr[idx] = float(node['split_condition'])
            by_id = {c['nodeid']: c for c in node['children']}
            left[idx] = visit(by_id[node['yes']])
            right[idx] = visit(by_id[node['no']])
            return idx

        visit(root)
        return feat, thr, left, right, leaf

    def _depth(self) -> int:
        depth = 0
        for t in range(self.n_trees):
            stack = [(0, 0)]
            while stack:
                node, d = stack.pop()
                depth = max(depth, d)
                if self.feat[t, node] >= 0:
                    stack += [(self.left[t, node], d + 1), (self.right[t, node], d + 1)]
        return depth

    def _calibrate(self, booster, n_features: int) -> float:
        import xgboost as xgb
        rng = np.random.RandomState(0)
        probe = np.abs(rng.standard_normal((8, n_features))) * np.array([1e3] * n_features)
        margin = booster.predict(xgb.DMatrix(probe.astype(np.float32)), output_margin=True)
        return float(np.mean(margin - self.tree_sum(probe)))

    def tree_sum(self, X: np.ndarray) -> np.ndarray:
        X32 = np.atleast_2d(X).astype(np.float32)
        n = len(X32)
        node = np.zeros((n, self.n_trees), dtype=np.int64)
        tree_ix = np.arange(self.n_trees)[None, :]
        rows = np.arange(n)[:, None]
        for _ in range(self.max_depth + 1):
            f = self.feat[tree_ix, node]
            leafmask = f < 0
            val = X32[rows, np.maximum(f, 0)]
            go_left = val < self.thr[tree_ix, node]
            nxt = np.where(go_left, self.left[tree_ix, node], self.right[tree_ix, node])
            node = np.where(leafmask, node, nxt)
        return self.leaf[tree_ix, node].sum(axis=1)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Raw log-odds of the malicious class (prob >= 0.5 iff margin >= 0)."""
        return self.tree_sum(X) + self.base_margin


def project_batch_with_origin(X: np.ndarray, x_orig: np.ndarray, schema: NetworkFeatureSchema) -> np.ndarray:
    """Row-wise ``project_network_invariants(x, schema, x_orig)`` for a matrix of candidates."""
    P = np.array(X, dtype=float, copy=True)
    for idx in schema.immutable_indices:
        P[:, idx] = x_orig[idx]
    for idx in schema.monotonic_increasing_indices:
        P[:, idx] = np.maximum(x_orig[idx], P[:, idx])
    for idx in (4, 6, 11):  # the scalar projection floors these at the original, rounded
        P[:, idx] = np.maximum(x_orig[idx], np.round(P[:, idx]))
    return project_network_invariants_batch(P)


class WhiteBoxLeafTupleAttack(BaseNIDSAttack):
    """Greedy threshold-crossing attack on an XGBoost ensemble with full tree access."""

    def __init__(self, model, schema: NetworkFeatureSchema, max_steps: int = 20,
                 project_in_loop: bool = False, tree_dump: Optional[TreeDump] = None):
        super().__init__(model, schema)
        self.max_steps = max_steps
        self.project_in_loop = project_in_loop
        self.dump = tree_dump or TreeDump(getattr(model, 'model', model))
        self.internal_evals = 0
        self.raw_success = False  # margin < 0 before the final projection

    def _candidates(self, x: np.ndarray) -> np.ndarray:
        """One candidate per (feature, threshold) that lies on the other side of the threshold."""
        rows = []
        for j, ths in enumerate(self.dump.thresholds):
            if len(ths) == 0:
                continue
            xj = np.float32(x[j])
            below = np.nextafter(ths, np.float32(-np.inf)).astype(np.float32)
            vals = np.where(xj < ths, ths, below)  # cross to the opposite side of each threshold
            vals = vals[(vals >= 0) & (np.float32(xj) != vals)]
            cand = np.repeat(x[None, :], len(vals), axis=0)
            cand[:, j] = vals
            rows.append(cand)
        return np.concatenate(rows) if rows else np.empty((0, len(x)))

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        self.internal_evals = 0
        self.raw_success = False
        x_orig = x.copy()
        x_curr = project_network_invariants(x_orig, self.schema, x_orig)
        if self.predict_malicious_prob(x_curr) < 0.5:
            self.raw_success = True
            return self._result(x_orig, x_curr)

        margin = float(self.dump.margin(x_curr)[0])
        for _ in range(self.max_steps):
            if margin < 0:
                break
            cand = self._candidates(x_curr)
            if len(cand) == 0:
                break
            if self.project_in_loop:
                cand = project_batch_with_origin(cand, x_orig, self.schema)
            margins = self.dump.margin(cand)
            self.internal_evals += len(cand)
            best = int(np.argmin(margins))
            if margins[best] >= margin - 1e-12:
                break
            x_curr, margin = cand[best].copy(), float(margins[best])

        self.raw_success = margin < 0
        if not self.project_in_loop:
            x_curr = project_network_invariants(x_curr, self.schema, x_orig)
        return self._result(x_orig, x_curr)

    def _result(self, x_orig: np.ndarray, x_final: np.ndarray) -> AttackResult:
        p_final = self.predict_malicious_prob(x_final)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_final - x_orig)))
        return AttackResult(
            x_original=x_orig, x_adversarial=x_final, success=bool(p_final < 0.5),
            physically_valid=verify_physical_feasibility(x_orig, x_final, self.schema),
            queries_used=self.query_count, weighted_cost=cost, feature_shifts=x_final - x_orig)
