import numpy as np
import pandas as pd
from validation import get_cpcv_splits
from portfolio import build_weights, compute_daily_returns, sharpe
import os

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"
OUT_PATH = "data/processed/pbo_results.parquet"

def compute_portfolio_returns(df):
    df = df.copy()
    df["w"] = build_weights(df)
    return compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

def main():
    os.makedirs("data/processed", exist_ok=True)
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner").sort_values("date").reset_index(drop=True)

    is_sharpes, oos_sharpes = [], []
    for tr_idx, te_idx in get_cpcv_splits(df["date"]):
        is_sharpes.append(sharpe(compute_portfolio_returns(df.iloc[tr_idx])))
        oos_sharpes.append(sharpe(compute_portfolio_returns(df.iloc[te_idx])))

    is_sharpes = np.array(is_sharpes)
    oos_sharpes = np.array(oos_sharpes)

    ranks = pd.Series(is_sharpes).rank(ascending=False)
    top_half = ranks <= len(ranks) / 2
    failures = oos_sharpes[top_half] < np.nanmedian(oos_sharpes)
    pbo = np.nanmean(failures)

    pd.DataFrame({"is_sharpe": is_sharpes, "oos_sharpe": oos_sharpes}).to_parquet(OUT_PATH)

    print("===== PBO ANALYSIS =====")
    print(f"Mean IS Sharpe:  {np.nanmean(is_sharpes):.2f}")
    print(f"Mean OOS Sharpe: {np.nanmean(oos_sharpes):.2f}")
    print(f"PBO: {pbo:.3f}")
    print("========================")

if __name__ == "__main__":
    main()