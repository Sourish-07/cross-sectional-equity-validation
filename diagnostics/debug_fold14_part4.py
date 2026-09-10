# src/debug_fold14_part4.py -- faster version, no raw CSV read
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from validation import get_cpcv_splits

df = pd.read_parquet("data/processed/features.parquet").sort_values("date").reset_index(drop=True)
preds = pd.read_parquet("data/processed/predictions.parquet")

splits = list(get_cpcv_splits(df["date"]))
te_idx = splits[13][1]
sub = df.iloc[te_idx][["date", "ticker", "target", "vol_20_z"]].merge(
    preds, on=["date", "ticker"], how="left"
).dropna(subset=["pred"])

rows = []
for tkr, g in sub.groupby("ticker", observed=True):
    if g["target"].nunique() < 2 or len(g) < 20:
        continue
    rows.append((tkr, roc_auc_score(g["target"], g["pred"]), g["vol_20_z"].abs().mean()))

merged = pd.DataFrame(rows, columns=["ticker", "auc", "mean_abs_vol20z"]).set_index("ticker")
print("Correlation between per-ticker AUC and mean |vol_20_z|:", merged["auc"].corr(merged["mean_abs_vol20z"]))
print(merged.sort_values("mean_abs_vol20z", ascending=False).head(10))