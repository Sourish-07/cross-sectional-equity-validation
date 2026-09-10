import numpy as np
import pandas as pd
from scipy.stats import norm
from portfolio import build_weights, compute_daily_returns, sharpe
from trials import TRIALS

PRED_PATH = "data/processed/predictions.parquet"
FEATURE_PATH = "data/processed/features.parquet"

def deflated_sharpe(sr, n, trials):
    z = sr * np.sqrt(n)
    penalty = norm.ppf(1 - 1 / trials)
    return (z - penalty) / np.sqrt(n)

def main():
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"])
    df["w"] = build_weights(df)
    daily = compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

    sr = sharpe(daily)
    n_trials = len(TRIALS)  # derived from the actual trial grid, not hardcoded
    dsr = deflated_sharpe(sr, len(daily.dropna()), trials=n_trials)

    print("===== DEFLATED SHARPE =====")
    print(f"Sharpe: {sr:.3f}")
    print(f"Trials used for penalty: {n_trials}")
    print(f"Deflated Sharpe: {dsr:.3f}")

if __name__ == "__main__":
    main()