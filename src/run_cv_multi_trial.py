"""
Runs a specified number of trials (from trials.get_trials(n)) through every
combinatorial CPCV split.

FIX vs. previous version: the IS holdout is now a TIME-ORDERED, embargoed
slice of the training block, not a random row sample. The previous random
split placed IS-holdout rows immediately adjacent in time to fit-set rows
within the same training block, with no embargo between them -- given
serial autocorrelation in daily returns, this could let information leak
across the fit/IS-holdout boundary and inflate the apparent IS signal used
to rank trials. The corrected version takes the chronologically LAST
~20% of rows (by cumulative row count, for balance) within the training
block as the IS holdout, with an explicit embargo gap removed from the
fit set immediately preceding it -- mirroring the same purge/embargo logic
already used between train and the true OOS test block.

Usage: set N_TRIALS below (7, 15, or 30 for the sensitivity sweep) and
OUT_DIR is automatically tagged with the trial count so runs at different
N do not overwrite each other.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from cpcv_combinatorial import get_combinatorial_cpcv_splits, count_combinations
from trials import get_trials, RANDOM_STATE
from portfolio import build_weights, compute_daily_returns, sharpe

FEATURES_PATH = "data/processed/features.parquet"

N_TRIALS = 30  # <-- CHANGE THIS to 7, 15, or 30 for each sweep run
OUT_DIR = f"data/processed/multi_trial_summaries_n{N_TRIALS}"

CHUNK_SIZE = 2_000_000
TREE_TRAIN_SAMPLE_CAP = 500_000
IS_HOLDOUT_FRAC = 0.20
IS_EMBARGO_DAYS = 5 * 21  # same embargo convention used elsewhere in this project

EXCLUDE_COLS = {
    "date", "ticker", "target", "ret_fwd", "open", "high", "low", "close", "volume", "source"
}
N_BLOCKS = 10
K_TEST_BLOCKS = 2


def carve_time_ordered_is_holdout(dates_full, tr_idx, is_frac=IS_HOLDOUT_FRAC, embargo_days=IS_EMBARGO_DAYS):
    """Splits tr_idx into (fit_idx, is_idx): is_idx is the chronologically
    last is_frac of rows (by cumulative row count) within the training
    block; fit_idx excludes an embargo window immediately preceding it."""
    tr_idx = np.asarray(tr_idx)
    sub_dates = dates_full.iloc[tr_idx]
    order = np.argsort(sub_dates.values, kind="stable")
    tr_idx_sorted = tr_idx[order]
    sub_dates_sorted = pd.to_datetime(sub_dates.values[order])

    unique_dates, counts = np.unique(sub_dates_sorted.values, return_counts=True)
    cum = np.cumsum(counts)
    total = cum[-1]
    cutoff_target = total * (1 - is_frac)
    cutoff_pos = min(np.searchsorted(cum, cutoff_target), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_pos])

    embargo = pd.Timedelta(days=embargo_days)
    is_mask = sub_dates_sorted >= cutoff_date
    fit_mask = sub_dates_sorted < (cutoff_date - embargo)

    is_idx = tr_idx_sorted[is_mask]
    fit_idx = tr_idx_sorted[fit_mask]
    return fit_idx, is_idx


def fit_predict_both(trial, X, y, fit_idx, is_idx, oos_idx, rng):
    scaler = StandardScaler()

    if trial["supports_partial_fit"]:
        model = trial["build"]()
        for i in range(0, len(fit_idx), CHUNK_SIZE):
            idx = fit_idx[i:i + CHUNK_SIZE]
            scaler.partial_fit(X.iloc[idx])
        for i in range(0, len(fit_idx), CHUNK_SIZE):
            idx = fit_idx[i:i + CHUNK_SIZE]
            X_scaled = scaler.transform(X.iloc[idx])
            model.partial_fit(X_scaled, y.iloc[idx], classes=np.array([0, 1]))
    else:
        fit_sample = fit_idx if len(fit_idx) <= TREE_TRAIN_SAMPLE_CAP else rng.choice(fit_idx, TREE_TRAIN_SAMPLE_CAP, replace=False)
        scaler.partial_fit(X.iloc[fit_sample])
        X_scaled_fit = scaler.transform(X.iloc[fit_sample])
        model = trial["build"]()
        model.fit(X_scaled_fit, y.iloc[fit_sample])

    p_is = model.predict_proba(scaler.transform(X.iloc[is_idx]))[:, 1]
    p_oos = model.predict_proba(scaler.transform(X.iloc[oos_idx]))[:, 1]
    return p_is, p_oos


def sharpe_from_preds(df_full, idx, preds):
    d = df_full.iloc[idx][["date", "ticker", "ret_fwd"]].copy()
    d["pred"] = preds
    d["w"] = build_weights(d, pred_col="pred")
    daily = compute_daily_returns(d, weight_col="w", ret_col="ret_fwd")
    return sharpe(daily)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    os.makedirs(OUT_DIR, exist_ok=True)

    trials = get_trials(N_TRIALS)
    print(f"Running with N_TRIALS={N_TRIALS} -> {len(trials)} trials")

    df = pd.read_parquet(FEATURES_PATH).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    X = df[feature_cols].astype("float32")
    y = df["target"].astype("int8")
    dates_full = df["date"]

    n_combos = count_combinations(N_BLOCKS, K_TEST_BLOCKS)
    print(f"{n_combos} combinatorial splits x {len(trials)} trials = {n_combos * len(trials)} total fits")

    for tr_idx, te_idx, combo_id in get_combinatorial_cpcv_splits(df["date"], N_BLOCKS, K_TEST_BLOCKS):
        combo_path = os.path.join(OUT_DIR, f"combo_{combo_id:03d}.parquet")
        if os.path.exists(combo_path):
            print(f"combo {combo_id:>3} | already saved, skipping")
            continue

        fit_idx, is_idx = carve_time_ordered_is_holdout(dates_full, tr_idx)

        rows = []
        for trial in trials:
            p_is, p_oos = fit_predict_both(trial, X, y, fit_idx, is_idx, te_idx, rng)

            auc_is = roc_auc_score(y.iloc[is_idx], p_is)
            auc_oos = roc_auc_score(y.iloc[te_idx], p_oos)
            is_sharpe = sharpe_from_preds(df, is_idx, p_is)
            oos_sharpe = sharpe_from_preds(df, te_idx, p_oos)

            print(f"combo {combo_id:>3} | {trial['trial_id']:<22} | "
                  f"IS AUC {auc_is:.4f} OOS AUC {auc_oos:.4f} | "
                  f"IS Sharpe {is_sharpe:.3f} OOS Sharpe {oos_sharpe:.3f}")

            rows.append({
                "combo_id": combo_id, "trial_id": trial["trial_id"],
                "is_auc": auc_is, "oos_auc": auc_oos,
                "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
            })

        pd.DataFrame(rows).to_parquet(combo_path)
        print(f"combo {combo_id:>3} | saved -> {combo_path}")

    print(f"All combinatorial splits complete for N_TRIALS={N_TRIALS}.")


if __name__ == "__main__":
    main()