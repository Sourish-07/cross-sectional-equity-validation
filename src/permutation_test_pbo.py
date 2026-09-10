"""
Permutation test: is the observed multi-trial PBO (computed at N_TRIALS=7)
meaningfully low, or is a value this low what you'd expect from noise given
7 correlated trials and 45 non-independent combinatorial splits?

Null hypothesis: within each combinatorial split, the mapping between
"which trial is IS-ranked as the winner" and "which trial's OOS Sharpe gets
compared against" carries no real information -- i.e., the winner's OOS
rank among its peers is exchangeable.

Procedure: for each of N_PERMUTATIONS iterations, independently shuffle the
trial_id <-> oos_sharpe pairing WITHIN each combo (keeping is_sharpe fixed
to preserve which trial is "selected" as winner, but randomizing which
OOS outcome that selection is credited with), recompute PBO exactly as in
multi_trial_pbo.py, and record it. The empirical p-value is the fraction
of permuted PBOs that are <= the observed PBO -- i.e., how often does
random chance alone produce a PBO this low or lower.
"""
import glob
import os
import numpy as np
import pandas as pd

N_TRIALS = 30  # match whichever run you're testing
IN_DIR = f"data/processed/multi_trial_summaries_n{N_TRIALS}"
N_PERMUTATIONS = 2000
RANDOM_STATE = 42


def compute_pbo_from_arrays(is_sharpe_by_combo, oos_sharpe_by_combo):
    failures = []
    for combo_id in is_sharpe_by_combo:
        is_vals = is_sharpe_by_combo[combo_id]
        oos_vals = oos_sharpe_by_combo[combo_id]
        if len(is_vals) < 2:
            continue
        winner_pos = np.argmax(is_vals)
        median_oos = np.median(oos_vals)
        failures.append(oos_vals[winner_pos] < median_oos)
    return np.mean(failures)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    combo_files = sorted(glob.glob(os.path.join(IN_DIR, "combo_*.parquet")))
    if not combo_files:
        raise FileNotFoundError(f"No combo files found in {IN_DIR}")

    sh = pd.concat([pd.read_parquet(p) for p in combo_files], ignore_index=True)

    is_by_combo, oos_by_combo = {}, {}
    for combo_id, g in sh.groupby("combo_id"):
        g = g.dropna(subset=["is_sharpe", "oos_sharpe"])
        if len(g) < 2:
            continue
        is_by_combo[combo_id] = g["is_sharpe"].values
        oos_by_combo[combo_id] = g["oos_sharpe"].values

    observed_pbo = compute_pbo_from_arrays(is_by_combo, oos_by_combo)
    print(f"Observed PBO (N_TRIALS={N_TRIALS}): {observed_pbo:.4f}")

    permuted_pbos = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        oos_shuffled = {}
        for combo_id, vals in oos_by_combo.items():
            oos_shuffled[combo_id] = rng.permutation(vals)
        permuted_pbos[p] = compute_pbo_from_arrays(is_by_combo, oos_shuffled)

    p_value = np.mean(permuted_pbos <= observed_pbo)

    print(f"Permutation null distribution: mean={permuted_pbos.mean():.4f}, "
          f"std={permuted_pbos.std():.4f}, "
          f"2.5/97.5 pct=[{np.percentile(permuted_pbos, 2.5):.4f}, {np.percentile(permuted_pbos, 97.5):.4f}]")
    print(f"Empirical p-value (P[permuted PBO <= observed]): {p_value:.4f}")

    out = pd.DataFrame({"permuted_pbo": permuted_pbos})
    out.to_csv(f"results/permutation_test_n{N_TRIALS}.csv", index=False)
    print(f"Saved permutation samples -> results/permutation_test_n{N_TRIALS}.csv")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()