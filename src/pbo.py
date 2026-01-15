import numpy as np
import pandas as pd
from validation import get_cpcv_splits
import os

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"
OUT_PATH = "data/processed/pbo_results.parquet"

def sharpe(x):
    x = x.dropna()
    if x.std() == 0 or len(x) < 5:
        return np.nan
    return np.sqrt(252) * x.mean() / x.std()

def compute_portfolio_returns(df):
    df = df.copy()

    # Dollar-neutral, leverage-normalized portfolio
    df["w"] = (
        df.groupby("date", observed=True)["pred"]
        .transform(lambda x: (x - x.mean()) / (x.abs().sum() + 1e-12))
        .clip(-0.05, 0.05)
    )

    daily = (
        df.groupby("date", observed=True)
        .apply(lambda x: np.sum(x["w"] * x["ret_fwd"]))
    )

    return daily

def main():
    os.makedirs("data/processed", exist_ok=True)

    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]

    df = (
        preds.merge(feats, on=["date", "ticker"], how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    is_sharpes = []
    oos_sharpes = []

    for tr_idx, te_idx in get_cpcv_splits(df["date"]):
        is_df = df.iloc[tr_idx]
        oos_df = df.iloc[te_idx]

        is_daily = compute_portfolio_returns(is_df)
        oos_daily = compute_portfolio_returns(oos_df)

        is_sharpes.append(sharpe(is_daily))
        oos_sharpes.append(sharpe(oos_daily))

    is_sharpes = np.array(is_sharpes)
    oos_sharpes = np.array(oos_sharpes)

    # PBO calculation (Bailey et al.)
    ranks = pd.Series(is_sharpes).rank(ascending=False)
    top_half = ranks <= len(ranks) / 2
    failures = oos_sharpes[top_half] < np.nanmedian(oos_sharpes)
    pbo = np.nanmean(failures)

    # Save fold-level results (CRITICAL for figures + appendix)
    out = pd.DataFrame({
        "is_sharpe": is_sharpes,
        "oos_sharpe": oos_sharpes
    })
    out.to_parquet(OUT_PATH)

    print("===== PBO ANALYSIS =====")
    print(f"Mean IS Sharpe:  {np.nanmean(is_sharpes):.2f}")
    print(f"Mean OOS Sharpe: {np.nanmean(oos_sharpes):.2f}")
    print(f"PBO: {pbo:.3f}")
    print("========================")

if __name__ == "__main__":
    main()
