# src/debug_fold14.py -- extend with this

import pandas as pd
from validation import get_cpcv_splits

df = pd.read_parquet("data/processed/features.parquet").sort_values("date").reset_index(drop=True)

print(f"{'Fold':<6}{'Rows':<12}{'Tickers':<10}{'Date range'}")
for k, (tr, te) in enumerate(get_cpcv_splits(df["date"])):
    d = df["date"].iloc[te]
    t = df["ticker"].iloc[te]
    print(f"{k+1:<6}{len(te):<12}{t.nunique():<10}{d.min().date()} to {d.max().date()}")

for k, (tr, te) in enumerate(get_cpcv_splits(df["date"])):
    if k == 13:
        sub = df.iloc[te]
        dupes = sub.duplicated(subset=["date", "ticker"]).sum()
        train_dates = set(df["date"].iloc[tr])
        test_dates = set(sub["date"])
        overlap = train_dates & test_dates
        print(f"\nFold 14 duplicate (date,ticker) rows: {dupes}")
        print(f"Fold 14 train/test date overlap (should be 0): {len(overlap)} dates")