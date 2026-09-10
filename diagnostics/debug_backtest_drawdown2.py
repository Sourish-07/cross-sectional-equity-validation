# diagnostics/debug_backtest_drawdown2.py
import pandas as pd
import numpy as np
from src.portfolio import build_weights, compute_turnover, compute_daily_returns

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

print("daily (unscaled) return stats:")
print(daily.describe())
print("\nWorst 10 raw daily returns:")
print(daily.sort_values().head(10))

rolling_vol_noshift = daily.rolling(20).std()
rolling_vol_shift = daily.rolling(20).std().shift(1)

vol_scale_noshift = (TARGET_DAILY_VOL / rolling_vol_noshift).clip(0, 3)
vol_scale_shift = (TARGET_DAILY_VOL / rolling_vol_shift).clip(0, 3)

print("\nvol_scale (no shift) hitting the 3x cap, fraction of days:", (vol_scale_noshift == 3).mean())
print("vol_scale (shifted) hitting the 3x cap, fraction of days:", (vol_scale_shift == 3).mean())