"""
Block bootstrap CIs for the single-model diagnostic's Sharpe ratio and PBO.
Uses a moving-block bootstrap (not naive i.i.d. resampling) because daily
returns are serially correlated; naive bootstrap would understate the true
variance of the Sharpe estimate.
"""
import numpy as np
import pandas as pd
from portfolio import build_weights, compute_daily_returns, sharpe
from validation import get_cpcv_splits

FEATURE_PATH = "data/processed/features.parquet"
PRED_PATH = "data/processed/predictions.parquet"

BLOCK_SIZE = 20          # ~1 trading month; matches the vol_20 window elsewhere in this project
N_BOOTSTRAP = 2000
RANDOM_STATE = 42


def moving_block_bootstrap_sample(x, block_size, rng):
    n = len(x)
    n_blocks_needed = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks_needed)
    sample = np.concatenate([x[s:s + block_size] for s in starts])[:n]
    return sample


def bootstrap_sharpe_ci(daily_returns, block_size=BLOCK_SIZE, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    x = daily_returns.dropna().values
    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = moving_block_bootstrap_sample(x, block_size, rng)
        boot_sharpes[i] = np.sqrt(252) * sample.mean() / sample.std() if sample.std() > 0 else np.nan
    boot_sharpes = boot_sharpes[~np.isnan(boot_sharpes)]
    return {
        "point_estimate": np.sqrt(252) * x.mean() / x.std(),
        "boot_mean": boot_sharpes.mean(),
        "ci_2.5": np.percentile(boot_sharpes, 2.5),
        "ci_97.5": np.percentile(boot_sharpes, 97.5),
    }


def main():
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEATURE_PATH)[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner").sort_values("date").reset_index(drop=True)

    df["w"] = build_weights(df)
    daily = compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

    sharpe_ci = bootstrap_sharpe_ci(daily)
    print("===== BLOCK BOOTSTRAP: FULL-SAMPLE SHARPE =====")
    print(f"Point estimate: {sharpe_ci['point_estimate']:.3f}")
    print(f"Bootstrap mean: {sharpe_ci['boot_mean']:.3f}")
    print(f"95% CI: [{sharpe_ci['ci_2.5']:.3f}, {sharpe_ci['ci_97.5']:.3f}]")
    print("================================================")

    # Bootstrap CI for the fold-level PBO: resample which folds are included, with replacement
    rng = np.random.default_rng(RANDOM_STATE)
    is_sh, oos_sh = [], []
    for tr_idx, te_idx in get_cpcv_splits(df["date"]):
        for idx, bucket in [(tr_idx, is_sh), (te_idx, oos_sh)]:
            d = df.iloc[idx].copy()
            d["w"] = build_weights(d)
            daily_fold = compute_daily_returns(d, weight_col="w", ret_col="ret_fwd")
            bucket.append(sharpe(daily_fold))
    is_sh, oos_sh = np.array(is_sh), np.array(oos_sh)
    n_folds = len(is_sh)

    boot_pbos = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_folds, size=n_folds)
        is_b, oos_b = is_sh[idx], oos_sh[idx]
        ranks = pd.Series(is_b).rank(ascending=False)
        top_half = ranks <= len(ranks) / 2
        failures = oos_b[top_half] < np.nanmedian(oos_b)
        boot_pbos[i] = np.nanmean(failures)

    print("\n===== BLOCK BOOTSTRAP: SINGLE-MODEL PBO (fold resampling) =====")
    print(f"Point estimate: {(np.nanmean(oos_sh[pd.Series(is_sh).rank(ascending=False) <= n_folds/2] < np.nanmedian(oos_sh))):.3f}")
    print(f"Bootstrap mean: {boot_pbos.mean():.3f}")
    print(f"95% CI: [{np.percentile(boot_pbos, 2.5):.3f}, {np.percentile(boot_pbos, 97.5):.3f}]")
    print("================================================================")


if __name__ == "__main__":
    main()