import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from validation import get_cpcv_splits
import os

FEATURES_PATH = "data/processed/features.parquet"
OUT_PATH = "data/processed/predictions.parquet"

CHUNK_SIZE = 2_000_000

EXCLUDE_COLS = {
        "date", "ticker", "target", "open", "high", "low", "close", "volume", "source"
    }

def main():
        df = pd.read_parquet(FEATURES_PATH)
        df = df.sort_values("date").reset_index(drop=True)

        # ---- Feature detection ----
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not feature_cols:
            raise ValueError("No usable feature columns detected.")

        print(f"Using {len(feature_cols)} features:")
        print(feature_cols)

        X = df[feature_cols].astype("float32")
        y = df["target"].astype("int8")

        preds = np.full(len(df), np.nan, dtype="float32")

        # ---- Cross-validation ----
        for k, (tr, te) in enumerate(get_cpcv_splits(df["date"])):
            print(f"Fold {k+1}")

            scaler = StandardScaler()
            imputer = SimpleImputer(strategy="median")

            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1,
                learning_rate="optimal",
                warm_start=True
            )

            # ---- FIT IMPUTER ON FULL TRAIN (ONCE) ----
            imputer.fit(X.iloc[tr])

            # ---- FIT SCALER (CHUNKED) ----
            for i in range(0, len(tr), CHUNK_SIZE):
                idx = tr[i:i + CHUNK_SIZE]
                X_imp = imputer.transform(X.iloc[idx])
                scaler.partial_fit(X_imp)

            # ---- FIT MODEL (CHUNKED) ----
            for i in range(0, len(tr), CHUNK_SIZE):
                idx = tr[i:i + CHUNK_SIZE]
                X_imp = imputer.transform(X.iloc[idx])
                X_scaled = scaler.transform(X_imp)

                model.partial_fit(
                    X_scaled,
                    y.iloc[idx],
                    classes=np.array([0, 1])
                )

            # ---- PREDICT ----
            X_te = scaler.transform(imputer.transform(X.iloc[te]))
            p = model.predict_proba(X_te)[:, 1]
            preds[te] = p

            auc = roc_auc_score(y.iloc[te], p)
            print(f"  AUC: {auc:.4f}")

            # ---- SAVE INCREMENTALLY ----
            out = df.loc[~np.isnan(preds), ["date", "ticker"]].copy()
            out["pred"] = preds[~np.isnan(preds)]
            out.to_parquet(OUT_PATH, index=False)

        print(f"Saved predictions → {OUT_PATH}")

if __name__ == "__main__":
        main()