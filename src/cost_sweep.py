import os
import pandas as pd
import numpy as np

os.makedirs("results", exist_ok=True)

pred = pd.read_parquet("data/processed/predictions.parquet")
rets = pd.read_parquet("data/processed/features.parquet")[["date","ticker","ret_fwd"]]

df = pred.merge(rets, on=["date","ticker"], how="inner")

df["w"] = df.groupby("date")["pred"].transform(
    lambda x: (x - x.mean()) / (x.abs().sum() + 1e-12)
)
df["dw"] = df.groupby("ticker")["w"].diff().abs().fillna(0)

rows = []
for c in [0, 5e-4, 1e-3, 2e-3]:
    daily = df.groupby("date", observed=True).apply(
        lambda x: np.sum(x["w"] * x["ret_fwd"]) - c * x["dw"].sum()
    )
    sharpe = np.sqrt(252) * daily.mean() / daily.std()
    rows.append({"cost": c, "sharpe": sharpe})

out = pd.DataFrame(rows)
out.to_csv("results/cost_sweep.csv", index=False)
print(out)
