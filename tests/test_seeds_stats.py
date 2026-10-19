import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments')))

from seeds_stats import mcnemar_counts, pooled_mcnemar, summarise_outcomes, win_tie_loss  # noqa: E402


def test_summary_counts_only_valid_successes():
    out = [{'success': True, 'valid': True, 'queries': 10, 'op_cost': 2.0},
           {'success': True, 'valid': False, 'queries': 20, 'op_cost': 9.0},
           {'success': False, 'valid': True, 'queries': 30, 'op_cost': 1.0},
           {'success': True, 'valid': True, 'queries': 40, 'op_cost': 4.0}]
    s = summarise_outcomes(out)
    assert s['n'] == 4 and s['real_asr'] == 50.0 and s['asr'] == 75.0 and s['pfr'] == 75.0
    assert s['avg_queries'] == 25.0 and s['median_op_cost_real'] == 3.0


def test_mcnemar_and_pooling():
    a = np.array([1, 1, 1, 0, 0], bool)
    b = np.array([1, 0, 0, 1, 0], bool)
    assert mcnemar_counts(a, b) == (2, 1)
    pooled = pooled_mcnemar([(2, 1), (8, 0)])
    assert (pooled['v2_only'], pooled['other_only']) == (10, 1) and pooled['p'] < 0.05
    assert pooled_mcnemar([(0, 0)])['p'] == 1.0


def test_win_tie_loss():
    assert win_tie_loss([5, 3, 1], [4, 3, 2]) == (1, 1, 1)
