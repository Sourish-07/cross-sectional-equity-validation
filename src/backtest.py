import pandas as pd
import numpy as np

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"

TRADING_DAYS = 252
TARGET_DAILY_VOL = 0.01     # 1% daily vol target
TCOST = 0.0005              # 5 bps per unit turnover
MAX_WEIGHT = 0.05           # leverage cap per asset


def normalize(x):
    """L1-normalize cross-section"""
    s = np.sum(np.abs(x))
    if s == 0 or np.isnan(s):
        return np.zeros_like(x)
    return x / s


def main():
    # ---------------------------
    # Load predictions
    # ---------------------------
    preds = pd.read_parquet(PRED_PATH)

    if "pred" not in preds.columns:
        raise ValueError("Prediction column 'pred' missing.")

    # ---------------------------
    # Load realized forward returns
    # ---------------------------
    feats = pd.read_parquet(FEATURE_PATH)[
        ["date", "ticker", "ret_fwd"]
    ]

    # ---------------------------
    # Merge predictions + returns
    # ---------------------------
    df = preds.merge(
        feats,
        on=["date", "ticker"],
        how="inner"
    )

    # ---------------------------
    # Cross-sectional portfolio
    # ---------------------------
    df["weight"] = (
        df.groupby("date", group_keys=False)["pred"]
          .transform(normalize)
          .clip(-MAX_WEIGHT, MAX_WEIGHT)
    )

    # ---------------------------
    # Raw daily portfolio returns
    # ---------------------------
    daily = (
        df.groupby("date", group_keys=False)
          .apply(lambda x: np.sum(x["weight"] * x["ret_fwd"]))
          .sort_index()
    )

    # ---------------------------
    # Volatility targeting
    # ---------------------------
    rolling_vol = daily.rolling(20).std()
    vol_scale = (TARGET_DAILY_VOL / rolling_vol).clip(0, 3)
    daily_vol_targeted = daily * vol_scale

    # ---------------------------
    # Transaction costs
    # ---------------------------
    turnover = (
        df.sort_values(["ticker", "date"])
          .groupby("ticker")["weight"]
          .diff()
          .abs()
          .groupby(df["date"])
          .sum()
          .fillna(0)
    )

    daily_net = daily_vol_targeted - TCOST * turnover

    # ---------------------------
    # Performance metrics
    # ---------------------------
    mean = daily_net.mean()
    std = daily_net.std()

    sharpe = mean / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan

    cum = (1 + daily_net).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    # ---------------------------
    # Information Coefficient (IC)
    # ---------------------------
    ic = (
        df.groupby("date", group_keys=False)
          .apply(lambda x: np.corrcoef(x["pred"], x["ret_fwd"])[0, 1])
          .dropna()
    )

    ic_mean = ic.mean()
    ic_t = ic_mean / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan

    # ---------------------------
    # Print results
    # ---------------------------
    print("\n===== BACKTEST RESULTS =====")
    print(f"Annualized Sharpe (net): {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Mean IC: {ic_mean:.4f}")
    print(f"IC t-stat: {ic_t:.2f}")
    print("============================\n")


if __name__ == "__main__":
    main()
