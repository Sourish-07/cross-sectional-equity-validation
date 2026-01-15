import pandas as pd
import numpy as np

FEATURE_PATH = "data/processed/features.parquet"

BASELINES = {
    "momentum": "mom_5_rank",
    "mean_reversion": "mr_5_rank",
}

def normalize(x):
    s = np.sum(np.abs(x))
    return x * 0 if s == 0 else x / s

def run(signal_col):
    df = pd.read_parquet(FEATURE_PATH)[
        ["date", "ticker", signal_col, "ret_fwd"]
    ].dropna()

    df["weight"] = (
        df.groupby("date")[signal_col]
        .transform(normalize)
        .clip(-0.05, 0.05)
    )

    daily = (
        df.groupby("date")
        .apply(lambda x: np.sum(x["weight"] * x["ret_fwd"]))
    )

    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    dd = ((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()

    return sharpe, dd

def main():
    print("===== BASELINES =====")
    for name, col in BASELINES.items():
        s, d = run(col)
        print(f"{name:15s} Sharpe: {s:.2f} | MaxDD: {d:.2%}")

if __name__ == "__main__":
    main()
