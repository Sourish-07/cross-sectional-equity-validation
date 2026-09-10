import os
import pandas as pd
import numpy as np
from portfolio import build_weights, compute_turnover, sharpe

os.makedirs("results", exist_ok=True)

pred = pd.read_parquet("data/processed/predictions.parquet")
rets = pd.read_parquet("data/processed/features.parquet")[["date", "ticker", "ret_fwd"]]
df = pred.merge(rets, on=["date", "ticker"], how="inner")

df["w"] = build_weights(df)
turnover_daily = compute_turnover(df, weight_col="w")
gross_daily = df.groupby("date").apply(
    lambda x: np.sum(x["w"] * x["ret_fwd"]), include_groups=False
).sort_index()

rows = []
for c in [0, 5e-4, 1e-3, 2e-3]:
    net_daily = gross_daily - c * turnover_daily
    rows.append({
        "cost": c,
        "sharpe": sharpe(net_daily),
        "mean_daily_cost_drag": c * turnover_daily.mean()  # sanity check — must increase with c
    })

out = pd.DataFrame(rows)
out.to_csv("results/cost_sweep.csv", index=False)
print(out)