"""
Direct test of the reconciliation mechanism: are trials within the same
combinatorial split more correlated with each other than a single trial
is correlated with itself across different splits?

Uses the multi_trial_summaries_nX data already generated -- no retraining
needed.

Two numbers:
1. Within-split correlation: for each split, take the OOS Sharpe of every
   trial; compute the average pairwise correlation across splits (i.e.
   do trials that do well in one split tend to also do well in another,
   as a group?) -- more precisely, we measure how tightly trials cluster
   within a split relative to the spread across splits, via a simple
   variance-decomposition: how much of the total variance in OOS Sharpe
   is "between splits" (regime-driven) vs "within splits" (trial-driven).
2. If between-split variance >> within-split variance, this confirms
   split/regime, not trial identity, dominates -- supporting the
   reconciliation argument directly.
"""
import glob
import pandas as pd
import numpy as np

N_TRIALS = 30  # change to 15 or 30 to check at other grid sizes
IN_DIR = f"data/processed/multi_trial_summaries_n{N_TRIALS}"


def main():
    combo_files = sorted(glob.glob(f"{IN_DIR}/combo_*.parquet"))
    sh = pd.concat([pd.read_parquet(p) for p in combo_files], ignore_index=True)

    # Pivot: rows = combo_id, columns = trial_id, values = oos_sharpe
    pivot = sh.pivot(index="combo_id", columns="trial_id", values="oos_sharpe")

    # Variance decomposition: total variance = between-split + within-split
    grand_mean = pivot.values.mean()
    split_means = pivot.mean(axis=1)  # mean OOS sharpe per split, averaged across trials
    between_split_var = ((split_means - grand_mean) ** 2).mean()

    within_split_var = pivot.sub(split_means, axis=0).pow(2).values.mean()

    total_var = between_split_var + within_split_var
    pct_between = 100 * between_split_var / total_var

    print(f"N_TRIALS = {N_TRIALS}")
    print(f"Between-split variance (driven by which split/regime): {between_split_var:.4f}")
    print(f"Within-split variance (driven by which trial): {within_split_var:.4f}")
    print(f"Share of total variance explained by split/regime: {pct_between:.1f}%")

    # Average pairwise correlation across trials, computed across splits
    corr_matrix = pivot.corr()
    n = corr_matrix.shape[0]
    avg_pairwise_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
    print(f"Average pairwise correlation between trials (across the {len(pivot)} splits): {avg_pairwise_corr:.3f}")


if __name__ == "__main__":
    main()