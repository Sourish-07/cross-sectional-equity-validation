import numpy as np
import pandas as pd
from scipy.stats import norm

PRED_PATH = "data/processed/predictions.parquet"
FEATURE_PATH = "data/processed/features.parquet"

def deflated_sharpe(sr, n, trials):
    z = (sr * np.sqrt(n))
    penalty = norm.ppf(1 - 1 / trials)
    return (z - penalty) / np.sqrt(n)

def main():
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]

    df = preds.merge(feats, on=["date", "ticker"])

    df["w"] = (
        df.groupby("date")["pred"]
        .transform(lambda x: x / np.sum(np.abs(x)))
        .clip(-0.05, 0.05)
    )

    daily = (
        df.groupby("date", observed=True)
        .apply(lambda x: np.sum(x["w"] * x["ret_fwd"]))
    )

    sr = daily.mean() / daily.std() * np.sqrt(252)
    dsr = deflated_sharpe(sr, len(daily), trials=20)

    print("===== DEFLATED SHARPE =====")
    print(f"Sharpe: {sr:.3f}")
    print(f"Deflated Sharpe: {dsr:.3f}")

if __name__ == "__main__":
    main()
