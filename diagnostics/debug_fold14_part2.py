# src/debug_fold14_part2.py
import pandas as pd
import numpy as np
from validation import get_cpcv_splits

df = pd.read_parquet("data/processed/features.parquet").sort_values("date").reset_index(drop=True)
splits = list(get_cpcv_splits(df["date"]))
sub14 = df.iloc[splits[13][1]]
sub13 = df.iloc[splits[12][1]]   # neighbor fold for contrast, same rough size, normal AUC

print("Fold 14 ret_fwd == 0 fraction:", (sub14["ret_fwd"] == 0).mean())
print("Fold 13 ret_fwd == 0 fraction:", (sub13["ret_fwd"] == 0).mean())

print("\nFold 14 rows per ticker, top 10 most frequent:")
print(sub14["ticker"].value_counts().head(10))

feat_cols = [c for c in df.columns if c.endswith("_z") or c.endswith("_rank")]
print("\nFold 14 max abs feature values (look for extreme/inf-like outliers):")
print(sub14[feat_cols].abs().max().sort_values(ascending=False))
print("\nFold 13 max abs feature values (for contrast):")
print(sub13[feat_cols].abs().max().sort_values(ascending=False))