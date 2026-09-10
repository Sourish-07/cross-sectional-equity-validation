# src/debug_fold14_part3.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from validation import get_cpcv_splits

df = pd.read_parquet("data/processed/features.parquet").sort_values("date").reset_index(drop=True)
preds = pd.read_parquet("data/processed/predictions.parquet")

splits = list(get_cpcv_splits(df["date"]))
te_idx = splits[13][1]  # fold 14 test indices

sub = df.iloc[te_idx][["date", "ticker", "target"]].copy()
merged = sub.merge(preds, on=["date", "ticker"], how="left").dropna(subset=["pred"])
print("Fold 14 merged rows with predictions:", len(merged))

rows = []
for tkr, g in merged.groupby("ticker"):
    if g["target"].nunique() < 2 or len(g) < 20:
        continue
    rows.append((tkr, len(g), roc_auc_score(g["target"], g["pred"])))

pt = pd.DataFrame(rows, columns=["ticker", "n", "auc"]).sort_values("auc", ascending=False)
print("\nTop 15 tickers by per-ticker AUC in fold 14:")
print(pt.head(15).to_string(index=False))
print("\nBottom 15 tickers by per-ticker AUC in fold 14:")
print(pt.tail(15).to_string(index=False))
print("\nMedian per-ticker AUC:", pt["auc"].median())
print("How many tickers have AUC > 0.90:", (pt["auc"] > 0.90).sum(), "out of", len(pt))
print("Pooled AUC recomputed (sanity check vs original 0.8185):", roc_auc_score(merged["target"], merged["pred"]))