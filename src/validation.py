import numpy as np
import pandas as pd

def get_cpcv_splits(dates, n_splits=20, embargo_days=5 * 21):
    """
    Combinatorial Purged Cross-Validation with embargo.
    """
    dates = pd.to_datetime(dates)
    unique_dates = dates.sort_values().unique()
    folds = np.array_split(unique_dates, n_splits)

    embargo = pd.Timedelta(days=embargo_days)

    for i in range(n_splits):
        test_dates = folds[i]
        min_d, max_d = test_dates.min(), test_dates.max()

        test_mask = dates.isin(test_dates)
        embargo_mask = (dates >= min_d - embargo) & (dates <= max_d + embargo)

        train_mask = ~test_mask & ~embargo_mask

        yield np.where(train_mask)[0], np.where(test_mask)[0]
