"""
Generates the three figures for Paper 2:
  1. equity_curve_p2.png   -- cumulative returns of the vol-targeted backtest (backtest.py's output)
  2. pbo_scatter_p2.png    -- IS vs OOS Sharpe across the 20 row-balanced purged folds (single-model)
  3. multi_trial_scatter_p2.png -- IS vs OOS Sharpe across all 45 combinatorial splits x 7 trials,
                                    color-coded by trial, illustrating the reconciliation argument
                                    in Section "Reconciling the Two Diagnostics"

Run after the full pipeline (backtest.py, pbo.py, run_cv_multi_trial.py, multi_trial_pbo.py)
has already produced their output parquet/csv files.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio import build_weights, compute_turnover, compute_daily_returns, sharpe
from validation import get_cpcv_splits

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

TARGET_DAILY_VOL = 0.01
TCOST = 0.0005


# ---------------------------------------------------------------
# Figure 1: Equity curve (vol-targeted, cost-adjusted backtest)
# ---------------------------------------------------------------
def make_equity_curve():
    preds = pd.read_parquet("data/processed/predictions.parquet")
    feats = pd.read_parquet("data/processed/features.parquet")[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner")

    df["w"] = build_weights(df)
    turnover = compute_turnover(df, weight_col="w")
    daily = compute_daily_returns(df, weight_col="w", ret_col="ret_fwd")

    rolling_vol = daily.rolling(20).std().shift(1)
    vol_scale = (TARGET_DAILY_VOL / rolling_vol).clip(0, 3)
    daily_net = (daily * vol_scale) - TCOST * turnover

    cum = (1 + daily_net.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cum.index, cum.values, color="#1f4e79", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Cumulative Return (Growth of \\$1)")
    ax.set_xlabel("Date")
    ax.set_title("Figure 1: Cumulative Returns, Vol-Targeted Dollar-Neutral Backtest")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "equity_curve_p2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------
# Figure 2: IS vs OOS Sharpe, single-model, 20 row-balanced folds
# ---------------------------------------------------------------
def make_single_model_scatter():
    preds = pd.read_parquet("data/processed/predictions.parquet")
    feats = pd.read_parquet("data/processed/features.parquet")[["date", "ticker", "ret_fwd"]]
    df = preds.merge(feats, on=["date", "ticker"], how="inner").sort_values("date").reset_index(drop=True)

    is_sh, oos_sh = [], []
    for tr_idx, te_idx in get_cpcv_splits(df["date"]):
        for idx, bucket in [(tr_idx, is_sh), (te_idx, oos_sh)]:
            d = df.iloc[idx].copy()
            d["w"] = build_weights(d)
            daily = compute_daily_returns(d, weight_col="w", ret_col="ret_fwd")
            bucket.append(sharpe(daily))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(is_sh, oos_sh, color="#c0392b", alpha=0.75, edgecolor="black", linewidth=0.4)
    lims = [min(is_sh + oos_sh) - 0.2, max(is_sh + oos_sh) + 0.2]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, label="IS = OOS")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("In-Sample Sharpe")
    ax.set_ylabel("Out-of-Sample Sharpe")
    ax.set_title("Figure 2: IS vs. OOS Sharpe, Single-Model\nRow-Balanced Purged CV (20 folds)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pbo_scatter_p2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------
# Figure 3: IS vs OOS Sharpe, combinatorial multi-trial (45 splits x 7 trials)
# ---------------------------------------------------------------
def make_multi_trial_scatter():
    combo_files = sorted(glob.glob("data/processed/multi_trial_summaries/combo_*.parquet"))
    sh = pd.concat([pd.read_parquet(p) for p in combo_files], ignore_index=True)

    trials = sorted(sh["trial_id"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {t: cmap(i % 10) for i, t in enumerate(trials)}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for t in trials:
        sub = sh[sh["trial_id"] == t]
        ax.scatter(sub["is_sharpe"], sub["oos_sharpe"], label=t,
                   color=color_map[t], alpha=0.7, s=28, edgecolor="black", linewidth=0.3)

    all_vals = pd.concat([sh["is_sharpe"], sh["oos_sharpe"]]).dropna()
    lims = [all_vals.min() - 0.2, all_vals.max() + 0.2]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("In-Sample Sharpe (held-out IS slice)")
    ax.set_ylabel("Out-of-Sample Sharpe (true test block)")
    ax.set_title("Figure 3: IS vs. OOS Sharpe Across 45 Combinatorial Splits x 7 Trials")
    ax.legend(fontsize=8, loc="best", ncol=1)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multi_trial_scatter_p2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    make_equity_curve()
    make_single_model_scatter()
    make_multi_trial_scatter()