import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"

df = pd.read_parquet(PRED_PATH).merge(
    pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]],
    on=["date", "ticker"]
)

df["w"] = (
    df.groupby("date")["pred"]
    .transform(lambda x: x / np.sum(np.abs(x)))
    .clip(-0.05, 0.05)
)

daily = df.groupby("date").apply(lambda x: np.sum(x["w"] * x["ret_fwd"]))

# --- Cumulative ---
cum = (1 + daily).cumprod()
plt.figure()
cum.plot()
plt.title("Cumulative Returns")
plt.savefig("fig_1_cumulative.png")

# --- Drawdown ---
dd = cum / cum.cummax() - 1
plt.figure()
dd.plot()
plt.title("Drawdown")
plt.savefig("fig_2_drawdown.png")

# --- Rolling Sharpe ---
roll = daily.rolling(252).mean() / daily.rolling(252).std() * np.sqrt(252)
plt.figure()
roll.plot()
plt.title("Rolling Sharpe (252d)")
plt.savefig("fig_3_rolling_sharpe.png")
