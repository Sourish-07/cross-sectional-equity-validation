import numpy as np
import pandas as pd

MAX_WEIGHT = 0.05

def build_weights(df, pred_col="pred", max_weight=MAX_WEIGHT):
    """Dollar-neutral, leverage-normalized cross-sectional weights. Canonical version — use everywhere."""
    demeaned = df.groupby("date")[pred_col].transform(lambda x: x - x.mean())
    denom = df.groupby("date")[pred_col].transform(lambda x: (x - x.mean()).abs().sum() + 1e-12)
    return (demeaned / denom).clip(-max_weight, max_weight)

def compute_turnover(df, weight_col="w"):
    return (
        df.sort_values(["ticker", "date"])
          .groupby("ticker")[weight_col]
          .diff()
          .abs()
          .groupby(df["date"])
          .sum()
          .fillna(0)
    )

def compute_daily_returns(df, weight_col="w", ret_col="ret_fwd", cost_per_turnover=0.0, turnover=None):
    gross = df.groupby("date").apply(
        lambda x: np.sum(x[weight_col] * x[ret_col]), include_groups=False
    ).sort_index()
    if cost_per_turnover > 0:
        if turnover is None:
            turnover = compute_turnover(df, weight_col)
        gross = gross - cost_per_turnover * turnover
    return gross

def sharpe(daily_returns, trading_days=252):
    x = daily_returns.dropna()
    if len(x) < 5 or x.std() == 0:
        return np.nan
    return np.sqrt(trading_days) * x.mean() / x.std()