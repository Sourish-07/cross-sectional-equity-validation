"""
True Combinatorial Purged Cross-Validation (Lopez de Prado, "Advances in
Financial Machine Learning", ch. 12), with embargo and row-count-balanced
blocks (see validation.py for why row-count balancing matters here — the
equity universe grows ~20x from 2000 to 2016).

Unlike validation.py's get_cpcv_splits (purged K-fold: N sequential,
single-path folds), this partitions the data into N blocks and forms every
C(N, k) combination of test blocks. Each combination yields one train/test
split; the set of all combinations traces multiple, overlapping backtest
PATHS through time. This is what lets PBO estimate a genuine distribution
of Sharpe ratios rather than compare N independent point estimates.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def _row_balanced_blocks(dates, n_blocks):
    dates = pd.to_datetime(dates)
    date_counts = dates.value_counts().sort_index()
    cum_rows = date_counts.cumsum()
    total_rows = cum_rows.iloc[-1]
    boundaries = np.linspace(0, total_rows, n_blocks + 1)
    unique_dates = date_counts.index.values
    block_edges = np.searchsorted(cum_rows.values, boundaries[1:-1])
    return np.split(unique_dates, block_edges)


def get_combinatorial_cpcv_splits(dates, n_blocks=10, k_test_blocks=2, embargo_days=5 * 21):
    """
    Yields (train_idx, test_idx, combo_id) for every C(n_blocks, k_test_blocks)
    combination of test blocks.

    n_blocks=10, k_test_blocks=2 -> C(10,2) = 45 combinations, a common choice
    in the original CPCV paper balancing combinatorial coverage against
    per-split test-block size. Adjust k_test_blocks up for more paths (more
    compute), down for fewer.
    """
    dates = pd.to_datetime(dates)
    blocks = _row_balanced_blocks(dates, n_blocks)
    embargo = pd.Timedelta(days=embargo_days)

    for combo_id, test_block_idxs in enumerate(combinations(range(n_blocks), k_test_blocks)):
        test_dates = np.concatenate([blocks[i] for i in test_block_idxs])
        test_mask = dates.isin(test_dates)

        embargo_mask = pd.Series(False, index=dates.index)
        for i in test_block_idxs:
            min_d, max_d = blocks[i].min(), blocks[i].max()
            embargo_mask |= (dates >= min_d - embargo) & (dates <= max_d + embargo)

        train_mask = ~test_mask & ~embargo_mask
        yield np.where(train_mask)[0], np.where(test_mask)[0], combo_id


def count_combinations(n_blocks=10, k_test_blocks=2):
    from math import comb
    return comb(n_blocks, k_test_blocks)