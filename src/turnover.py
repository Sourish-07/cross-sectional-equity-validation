import pandas as pd
import numpy as np

PRED_PATH = "data/processed/predictions.parquet"
FEAT_PATH = "data/processed/features.parquet"

COST_PER_TURNOVER = 0.001  

def main():
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEAT_PATH)[["date", "ticker", "ret_fwd"]]

    df = preds.merge(feats, on=["date", "ticker"], how="inner")

    # Portfolio weights
    df["w"] = df.groupby("date")["pred"].transform(lambda x: x - x.mean())
    df["w"] /= df.groupby("date")["w"].transform(lambda x: x.abs().sum())

    df = df.sort_values(["ticker", "date"])
    df["dw"] = df.groupby("ticker")["w"].diff().abs().fillna(0)

    daily_turnover = df.groupby("date")["dw"].sum()
    daily_cost = daily_turnover * COST_PER_TURNOVER

    gross = df.groupby("date").apply(
        lambda x: np.sum(x["w"] * x["ret_fwd"])
    )

    net = gross - daily_cost

    sharpe_gross = np.sqrt(252) * gross.mean() / gross.std()
    sharpe_net = np.sqrt(252) * net.mean() / net.std()

    print("===== TURNOVER ANALYSIS =====")
    print("Gross Sharpe:", round(sharpe_gross, 2))
    print("Net Sharpe (10bps):", round(sharpe_net, 2))
    print("Avg daily turnover:", round(daily_turnover.mean(), 2))
    print("=============================")

if __name__ == "__main__":
    main()
