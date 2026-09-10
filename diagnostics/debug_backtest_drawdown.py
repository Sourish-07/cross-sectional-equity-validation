# diagnostics/debug_backtest_drawdown.py
import pandas as pd
import numpy as np
from portfolio import build_weights, compute_turnover, compute_daily_returns

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"
TARGET_DAILY_VOL = 0.01
TCOST = 0.0005

preds = pd.read_parquet(PRED_PATH)
feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]
df = preds.merge(feats, on=["date", "ticker"], how="inner")

df["w"] = build_weights(df)
turnover = compute_turnover(df, weight_col="w")
daily = compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

rolling_vol = daily.rolling(20).std()
vol_scale = (TARGET_DAILY_VOL / rolling_vol).clip(0, 3)
daily_vol_targeted = daily * vol_scale
daily_net = daily_vol_targeted - TCOST * turnover

print("Worst 15 single-day daily_net returns:")
print(daily_net.sort_values().head(15))
print("\nCorresponding raw daily (pre-vol-scaling) returns on those dates:")
print(daily.loc[daily_net.sort_values().head(15).index])
print("\nCorresponding vol_scale on those dates:")
print(vol_scale.loc[daily_net.sort_values().head(15).index])
print("\nrolling_vol stats:", rolling_vol.describe())