# diagnostics/debug_combo_correlation.py
import pandas as pd
sh = pd.read_parquet("data/processed/multi_trial_pbo_results.parquet")
# For sgd_alpha_0.0001 specifically, how much does OOS sharpe vary by combo?
sub = sh[sh["trial_id"] == "sgd_alpha_0.0001"]
print(sub[["combo_id", "oos_sharpe"]].describe())
print("\nFraction of combos with positive OOS sharpe for this trial:", (sub["oos_sharpe"] > 0).mean())