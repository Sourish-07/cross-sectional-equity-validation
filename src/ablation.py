import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.makedirs("results", exist_ok=True)

FEATURE_GROUPS = {
    "all": [
        "mom_5_z","mom_5_rank",
        "mr_5_z","mr_5_rank",
        "vol_20_z","vol_20_rank",
        "rel_volume_z","rel_volume_rank"
    ],
    "momentum": ["mom_5_z","mom_5_rank"],
    "mean_reversion": ["mr_5_z","mr_5_rank"],
    "volume": ["vol_20_z","vol_20_rank","rel_volume_z","rel_volume_rank"]
}

def backtest(df):
    df = df.copy()
    df["w"] = df.groupby("date")["pred"].transform(
        lambda x: (x - x.mean()) / (x.abs().sum() + 1e-12)
    )
    daily = df.groupby("date", observed=True).apply(
        lambda x: np.sum(x["w"] * x["ret_fwd"])
    )
    sharpe = np.sqrt(252) * daily.mean() / daily.std()
    ic = df.groupby("date", observed=True).apply(
        lambda x: np.corrcoef(x["pred"], x["ret_fwd"])[0,1]
    )
    return sharpe, ic.mean()

def main():
    df = pd.read_parquet("data/processed/features.parquet").dropna()
    results = []

    for name, feats in FEATURE_GROUPS.items():
        X = df[feats]
        y = (df["ret_fwd"] > 0).astype(int)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        df["pred"] = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, df["pred"])
        sharpe, mean_ic = backtest(df)

        results.append({
            "model": name,
            "auc": auc,
            "gross_sharpe": sharpe,
            "mean_ic": mean_ic
        })

    res = pd.DataFrame(results)
    res.to_csv("results/ablation.csv", index=False)
    print(res)

if __name__ == "__main__":
    main()
