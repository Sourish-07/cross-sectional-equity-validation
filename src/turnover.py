import pandas as pd
import numpy as np
from portfolio import build_weights, compute_turnover, sharpe

PRED_PATH = "data/processed/predictions.parquet"
FEAT_PATH = "data/processed/features.parquet"
COST_PER_TURNOVER = 0.001

def main():
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEAT_PATH)[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner")

    df["w"] = build_weights(df)
    turnover_daily = compute_turnover(df, weight_col="w")
    gross = df.groupby("date").apply(lambda x: np.sum(x["w"] * x["ret_fwd"]), include_groups=False)
    net = gross - COST_PER_TURNOVER * turnover_daily

    print("===== TURNOVER ANALYSIS =====")
    print("Gross Sharpe:", round(sharpe(gross), 2))
    print("Net Sharpe (10bps):", round(sharpe(net), 2))
    print("Avg daily turnover:", round(turnover_daily.mean(), 2))
    print("=============================")

if __name__ == "__main__":
    main()