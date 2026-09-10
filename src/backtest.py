import pandas as pd
import numpy as np
from portfolio import build_weights, compute_turnover, compute_daily_returns, sharpe

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"

TRADING_DAYS = 252
TARGET_DAILY_VOL = 0.01
TCOST = 0.0005

def main():
    preds = pd.read_parquet(PRED_PATH)
    if "pred" not in preds.columns:
        raise ValueError("Prediction column 'pred' missing.")

    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner")

    df["w"] = build_weights(df)
    turnover = compute_turnover(df, weight_col="w")
    daily = compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

    rolling_vol = daily.rolling(20).std().shift(1)  # lagged: no lookahead into same-day sizing
    vol_scale = (TARGET_DAILY_VOL / rolling_vol).clip(0, 3)
    daily_vol_targeted = daily * vol_scale
    daily_net = daily_vol_targeted - TCOST * turnover

    sr = sharpe(daily_net, TRADING_DAYS)

    cum = (1 + daily_net.fillna(0)).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    ic = df.groupby("date").apply(
        lambda x: np.corrcoef(x["pred"], x["ret_fwd"])[0, 1], include_groups=False
    ).dropna()
    ic_mean = ic.mean()
    ic_t = ic_mean / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan

    print("\n===== BACKTEST RESULTS (dollar-neutral) =====")
    print(f"Annualized Sharpe (net): {sr:.3f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Mean IC: {ic_mean:.4f}")
    print(f"IC t-stat: {ic_t:.2f}")
    print("==============================================\n")

if __name__ == "__main__":
    main()