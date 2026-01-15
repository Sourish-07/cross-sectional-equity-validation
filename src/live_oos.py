import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = "data/processed/features.parquet"

TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"

FEATURES = [
    "mom_5_z", "mom_5_rank",
    "mr_5_z", "mr_5_rank",
    "vol_20_z", "vol_20_rank",
    "rel_volume_z", "rel_volume_rank"
]

def main():
    df = pd.read_parquet(DATA).dropna(subset=FEATURES + ["ret_fwd"])

    train = df[df["date"] <= TRAIN_END]
    test = df[df["date"] >= TEST_START]

    X_train = train[FEATURES]
    y_train = (train["ret_fwd"] > 0).astype(int)

    X_test = test[FEATURES]
    y_test = test["ret_fwd"]

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    test["pred"] = model.predict_proba(X_test)[:, 1]

    test.to_parquet("data/processed/live_oos_predictions.parquet")
    print("Saved → live OOS predictions")

if __name__ == "__main__":
    main()