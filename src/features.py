import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/master_stock_data_paper2_snapshot.csv"
OUT_PATH = Path("data/processed/features.parquet")

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required = {"date", "ticker", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["ticker", "date"])

    bad = df["close"] <= 0
    if bad.any():
        print(f"Removed {bad.sum():,} invalid price rows")
        df = df.loc[~bad]

    # ---------------------------------------------------------------
    # Liquidity filter: excludes thinly-traded instruments (warrants,
    # illiquid micro-caps) whose stale/discretized pricing structurally
    # distorts predictability, using price/volume thresholds rather than
    # ticker-name heuristics to avoid false positives on legitimate
    # low-priced tickers.
    # ---------------------------------------------------------------
    df["dollar_vol"] = df["close"] * df["volume"]
    median_dollar_vol = (
        df.groupby("ticker")["dollar_vol"]
        .transform(lambda x: x.rolling(20, min_periods=10).median())
    )
    liquid = (df["close"] >= 1.0) & (median_dollar_vol >= 250_000)
    n_excluded = (~liquid).sum()
    print(f"Excluding {n_excluded:,} illiquid rows ({n_excluded/len(df):.1%} of data)")
    df = df[liquid].drop(columns=["dollar_vol"])

    # Returns
    df["ret"] = df.groupby("ticker")["close"].pct_change()

    # ---------------------------------------------------------------
    # Sanity filter: remove rows with implausible single-day returns.
    # Confirmed via manual inspection (e.g. GOOGL 2017-11-10 -> 2017-11-13,
    # price drops ~20.2x with no corresponding real corporate action) that
    # the raw source data contains a market-wide price-convention/rebasing
    # discontinuity around 2017-11-10, affecting ~24% of tickers on that
    # date alone. A liquid common stock does not move >75% in one day
    # absent a real corporate action already reflected in split-adjusted
    # pricing, so this bound is a conservative, principled cut that removes
    # this artifact (and any similar ones elsewhere in the file) without
    # touching genuine high-volatility days (e.g. March 2020).
    # NOTE / limitation: this only catches rebasing events with a factor
    # >1.75x; smaller undetected rebasing on the same date boundary may
    # remain. Documented as a known data limitation.
    # ---------------------------------------------------------------
    extreme_ret = df["ret"].abs() > 0.75
    n_extreme = extreme_ret.sum()
    print(f"Flagging {n_extreme:,} rows with |1-day return| > 75% as data artifacts")
    df = df[~extreme_ret]

    df["ret_fwd"] = df.groupby("ticker")["ret"].shift(-1)

    # Volatility
    df["vol_20"] = (
        df.groupby("ticker")["ret"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Signals
    df["mom_5"] = df.groupby("ticker")["close"].pct_change(5)
    df["mr_5"] = -df["mom_5"]
    df["rel_volume"] = df["volume"] / (
        df.groupby("ticker")["volume"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base = ["mom_5", "mr_5", "vol_20", "rel_volume"]
    for f in base:
        df[f"{f}_z"] = df.groupby("date")[f].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
        df[f"{f}_rank"] = df.groupby("date")[f].rank(pct=True)

    # Volatility-scaled label (Lopez de Prado–style)
    df["target"] = (df["ret_fwd"] / df["vol_20"] > 0).astype("int8")

    feature_cols = [c for c in df.columns if c.endswith("_z") or c.endswith("_rank")]
    keep = ["date", "ticker", "ret_fwd", "target"] + feature_cols

    df = df[keep].dropna(subset=feature_cols + ["target"])

    # Downcast
    for c in feature_cols + ["ret_fwd"]:
        df[c] = df[c].astype("float32")
    df["ticker"] = df["ticker"].astype("category")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)

    print("Feature generation complete")
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Saved → {OUT_PATH}")

if __name__ == "__main__":
    main()