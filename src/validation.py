import numpy as np
import pandas as pd

def get_cpcv_splits(dates, n_splits=20, embargo_days=5 * 21):
    """
    Purged K-Fold Cross-Validation with embargo, split by row count
    (not unique date count) so folds carry comparable sample weight
    despite the U.S. equity universe growing ~20x in ticker count
    from 2000 to 2016.
    """
    dates = pd.to_datetime(dates)
    date_counts = dates.value_counts().sort_index()   # rows per unique date
    cum_rows = date_counts.cumsum()
    total_rows = cum_rows.iloc[-1]
    boundaries = np.linspace(0, total_rows, n_splits + 1)

    unique_dates = date_counts.index.values
    fold_edges = np.searchsorted(cum_rows.values, boundaries[1:-1])
    folds = np.split(unique_dates, fold_edges)

    embargo = pd.Timedelta(days=embargo_days)

    for i in range(n_splits):
        test_dates = folds[i]
        min_d, max_d = test_dates.min(), test_dates.max()

        test_mask = dates.isin(test_dates)
        embargo_mask = (dates >= min_d - embargo) & (dates <= max_d + embargo)

        train_mask = ~test_mask & ~embargo_mask

        yield np.where(train_mask)[0], np.where(test_mask)[0]