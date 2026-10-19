"""Pure helpers for the multi-seed FFE-v2 evaluation: outcome summaries, exact McNemar, pooling over seeds."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


def summarise_outcomes(outcomes: Sequence[dict]) -> Dict:
    """Real ASR / PFR / mean queries / median operational cost of the real evasions, from per-flow outcomes."""
    n = len(outcomes)
    if n == 0:
        return {'n': 0, 'real_asr': None, 'asr': None, 'pfr': None, 'avg_queries': None, 'median_op_cost_real': None}
    real = [o for o in outcomes if o['success'] and o['valid']]
    return {'n': n,
            'real_asr': 100 * len(real) / n,
            'asr': 100 * sum(o['success'] for o in outcomes) / n,
            'pfr': 100 * sum(o['valid'] for o in outcomes) / n,
            'avg_queries': float(np.mean([o['queries'] for o in outcomes])),
            'median_op_cost_real': float(np.median([o['op_cost'] for o in real])) if real else None}


def real_flags(outcomes: Sequence[dict]) -> np.ndarray:
    return np.array([bool(o['success'] and o['valid']) for o in outcomes], dtype=bool)


def mcnemar_counts(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    """(flows only ``a`` evades, flows only ``b`` evades)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return int(np.sum(a & ~b)), int(np.sum(~a & b))


def exact_p(only_a: int, only_b: int) -> float:
    return 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def pooled_mcnemar(pairs: Sequence[Tuple[int, int]]) -> Dict:
    """Exact McNemar on discordant counts summed over seeds (flows differ per seed, so counts simply add)."""
    a = sum(p[0] for p in pairs)
    b = sum(p[1] for p in pairs)
    return {'v2_only': a, 'other_only': b, 'p': exact_p(a, b)}


def win_tie_loss(v2: Sequence[float], other: Sequence[float], tol: float = 1e-9) -> Tuple[int, int, int]:
    w = sum(x > y + tol for x, y in zip(v2, other))
    l = sum(x < y - tol for x, y in zip(v2, other))
    return w, len(v2) - w - l, l


def mean_std(vals: List[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float('nan'), float('nan'))
