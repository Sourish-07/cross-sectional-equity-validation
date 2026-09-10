# diagnostics/debug_fold12.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from validation import get_cpcv_splits

df = pd.read_parquet("data/processed/features.parquet").sort_values("date").reset_index(drop=True)
preds = pd.read_parquet("data/processed/predictions.parquet")

splits = list(get_cpcv_splits(df["date"]))
te_idx = splits[11][1]  # fold 12
d = df["date"].iloc[te_idx]
print("Fold 12 date range:", d.min(), "to", d.max())
print("Rows:", len(te_idx), "| Tickers:", df["ticker"].iloc[te_idx].nunique())

sub = df.iloc[te_idx][["date", "ticker", "target"]].merge(preds, on=["date", "ticker"], how="left").dropna(subset=["pred"])
rows = []
for tkr, g in sub.groupby("ticker", observed=True):
    if g["target"].nunique() < 2 or len(g) < 20:
        continue
    rows.append((tkr, len(g), roc_auc_score(g["target"], g["pred"])))
pt = pd.DataFrame(rows, columns=["ticker", "n", "auc"]).sort_values("auc", ascending=False)
print("\nTop 15 tickers by AUC:")
print(pt.head(15).to_string(index=False))
print("\nMedian per-ticker AUC:", pt["auc"].median())