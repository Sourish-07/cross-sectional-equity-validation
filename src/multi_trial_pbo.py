"""
Computes multi-trial PBO from a given trial-count's summary directory.
Set N_TRIALS to match whichever run_cv_multi_trial.py output you want to
score (must match the N_TRIALS used when that data was generated).
"""
import glob
import os
import pandas as pd

N_TRIALS = 30  # <-- CHANGE THIS to match the run you want to score: 7, 15, or 30
IN_DIR = f"data/processed/multi_trial_summaries_n{N_TRIALS}"
OUT_PATH = f"data/processed/multi_trial_pbo_results_n{N_TRIALS}.parquet"


def compute_pbo(sh):
    logits = []
    for combo_id, g in sh.groupby("combo_id"):
        g = g.dropna(subset=["is_sharpe", "oos_sharpe"])
        if len(g) < 2:
            continue
        winner = g.sort_values("is_sharpe", ascending=False).iloc[0]
        median_oos = g["oos_sharpe"].median()
        failure = winner["oos_sharpe"] < median_oos
        logits.append({
            "combo_id": combo_id, "winner_trial": winner["trial_id"],
            "winner_is_sharpe": winner["is_sharpe"], "winner_oos_sharpe": winner["oos_sharpe"],
            "median_oos_sharpe": median_oos, "failure": failure
        })
    logit_df = pd.DataFrame(logits)
    return logit_df["failure"].mean(), logit_df


def main():
    combo_files = sorted(glob.glob(os.path.join(IN_DIR, "combo_*.parquet")))
    if not combo_files:
        raise FileNotFoundError(f"No combo files found in {IN_DIR} -- run run_cv_multi_trial.py with N_TRIALS={N_TRIALS} first.")

    print(f"Found {len(combo_files)} combo files in {IN_DIR}")
    sh = pd.concat([pd.read_parquet(p) for p in combo_files], ignore_index=True)
    pbo, logit_df = compute_pbo(sh)

    sh.to_parquet(OUT_PATH)
    print(f"===== MULTI-TRIAL PBO (N_TRIALS={N_TRIALS}) =====")
    print(f"Combinatorial splits evaluated: {logit_df['combo_id'].nunique()}")
    print(f"Trials per split: {sh['trial_id'].nunique()}")
    print(f"PBO: {pbo:.4f}")
    print(f"Mean IS-winner Sharpe: {logit_df['winner_is_sharpe'].mean():.3f}")
    print(f"Mean IS-winner's OOS Sharpe: {logit_df['winner_oos_sharpe'].mean():.3f}")
    print("=================================================")


if __name__ == "__main__":
    main()