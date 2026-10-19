"""Physical-transformation augmentation and iterative adversarial training (library-agnostic).

``orbit_augment`` applies random compositions of the four schema-feasible actions (temporal
dilation, payload padding, chaff injection, window morphing) with log-uniform magnitudes to flows
of EITHER class; labels never change. Training a model on a flow together with its transformed copies
teaches it that these actions are not evidence of class (invariance by augmentation).
``iterative_adversarial_training`` additionally crafts flows with real attacks against the current
model each round and labels them with their true class.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.datasets.schema import project_network_invariants_batch

MAX_DURATION_US = 1.1e8
MAX_CHAFF = 1000
BENIGN_WINDOW = 64240.0
MONOTONE_IDX = (2, 4, 6, 11)


def orbit_augment(X: np.ndarray, rng: np.random.RandomState, p_action: float = 0.5,
                  win_pool: Optional[np.ndarray] = None) -> np.ndarray:
    """Random feasible transformation of each row of ``X`` (labels are unchanged by the caller)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    on = rng.random_sample((n, 4)) < p_action
    none = ~on.any(axis=1)
    on[none, rng.randint(0, 4, none.sum())] = True
    P = X.copy()
    tcp = X[:, 1] == 6
    # chaff: N packets of l0 bytes with h0 header bytes each (v_chaff)
    h0 = np.where(tcp, np.where(X[:, 11] / np.maximum(1.0, X[:, 2]) >= 26.0, 32.0, 20.0), 8.0)
    l0 = 20.0 + h0
    n_chaff = np.floor(np.exp(rng.uniform(0.0, np.log1p(MAX_CHAFF), n))) * on[:, 2]
    P[:, 2] += n_chaff
    P[:, 4] += n_chaff * l0
    P[:, 11] += n_chaff * h0
    # padding: extra payload bytes on the forward direction
    pad = np.exp(rng.uniform(0.0, np.log(1.0 + 1500.0 * np.maximum(1.0, P[:, 2])), n)) * on[:, 1]
    P[:, 4] += pad
    # dilation: stretch the duration by a log-uniform factor up to the 110 s cap
    ratio_max = np.maximum(MAX_DURATION_US / np.maximum(1.0, X[:, 6]), 1.0)
    P[:, 6] = X[:, 6] * np.exp(rng.uniform(0.0, 1.0, n) * np.log(ratio_max)) * on[:, 0] + X[:, 6] * ~on[:, 0]
    # window morphing (TCP only): benign OS profile, a value from the empirical pool, or uniform
    if on[:, 3].any():
        pick = rng.random_sample(n)
        pool = win_pool if win_pool is not None and len(win_pool) else np.array([BENIGN_WINDOW])
        new_win = np.where(pick < 0.5, BENIGN_WINDOW, np.where(pick < 0.8, rng.choice(pool, n),
                                                              rng.uniform(0.0, 65535.0, n)))
        P[:, 7] = np.where(on[:, 3] & tcp, new_win, X[:, 7])
    out = project_network_invariants_batch(P)
    for j in MONOTONE_IDX:
        out[:, j] = np.maximum(out[:, j], X[:, j])
    out = project_network_invariants_batch(out)
    out[:, [0, 1, 3, 5]] = X[:, [0, 1, 3, 5]]
    return out


def make_orbit_set(X: np.ndarray, y: np.ndarray, n: int, rng: np.random.RandomState,
                   balanced: bool = True, win_pool: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` transformed copies of random training flows; half per class when ``balanced``."""
    if balanced:
        idx = np.concatenate([rng.choice(np.where(y == c)[0], n // 2, replace=True) for c in (0, 1)])
    else:
        idx = rng.choice(len(X), n, replace=True)
    return orbit_augment(X[idx], rng, win_pool=win_pool), y[idx]


def iterative_adversarial_training(
        train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], object],
        X: np.ndarray, y: np.ndarray, craft_fn: Callable[[object, np.ndarray, int], np.ndarray],
        rounds: int = 2, n_craft: int = 4000, n_orbit: int = 0, aug_weight: float = 1.0,
        seed: int = 42, base_model: Optional[object] = None) -> Tuple[object, List[Dict[str, float]]]:
    """Adversarial training: attack the current model, add the flows with true labels, retrain.

    ``craft_fn(model, X_malicious, round)`` returns one adversarial flow per input row. Each round
    crafts against the model of the previous round, so flows that evade the latest model are kept
    and the attack families rotate with the round index. ``n_orbit`` orbit-augmented copies of both
    classes are added once. Returns the final model and a per-round log.
    """
    rng = np.random.RandomState(seed)
    mal = np.where(y == 1)[0]
    parts_x, parts_y, log = [X], [y], []
    if n_orbit:
        xo, yo = make_orbit_set(X, y, n_orbit, rng, win_pool=X[X[:, 1] == 6, 7][:100000])
        parts_x.append(xo), parts_y.append(yo)
    model = base_model
    for r in range(rounds):
        if model is None:
            model = train_fn(np.vstack(parts_x), np.concatenate(parts_y), _weights(parts_x, aug_weight))
        sel = rng.choice(mal, min(n_craft, len(mal)), replace=False)
        adv = craft_fn(model, X[sel], r)
        parts_x.append(adv), parts_y.append(np.ones(len(adv), dtype=int))
        log.append({'round': r, 'crafted': int(len(adv))})
        model = None
    final = train_fn(np.vstack(parts_x), np.concatenate(parts_y), _weights(parts_x, aug_weight))
    return final, log


def _weights(parts_x: Sequence[np.ndarray], aug_weight: float) -> Optional[np.ndarray]:
    if aug_weight == 1.0:
        return None
    w = [np.ones(len(parts_x[0]))] + [np.full(len(p), aug_weight) for p in parts_x[1:]]
    return np.concatenate(w)
