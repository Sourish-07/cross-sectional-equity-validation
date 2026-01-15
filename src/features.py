import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/master_stock_data_2000_2026.csv"
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

    # Returns
    df["ret"] = df.groupby("ticker")["close"].pct_change()
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
