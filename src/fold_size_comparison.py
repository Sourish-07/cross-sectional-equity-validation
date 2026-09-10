"""
Generates a bar chart comparing fold row-counts under date-balanced
splitting (the scheme in Paper 1) vs row-balanced splitting (Paper 2),
BOTH computed on the same final cleaned dataset, for a true apples-to-apples
comparison of the fix's effect in isolation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_PATH = "data/processed/features.parquet"
N_SPLITS = 20


def date_balanced_fold_sizes(dates, n_splits=N_SPLITS):
    dates = pd.to_datetime(dates)
    unique_dates = dates.sort_values().unique()
    folds = np.array_split(unique_dates, n_splits)
    return [dates.isin(f).sum() for f in folds]


def row_balanced_fold_sizes(dates, n_splits=N_SPLITS):
    dates = pd.to_datetime(dates)
    date_counts = dates.value_counts().sort_index()
    cum_rows = date_counts.cumsum()
    total_rows = cum_rows.iloc[-1]
    boundaries = np.linspace(0, total_rows, n_splits + 1)
    unique_dates = date_counts.index.values
    fold_edges = np.searchsorted(cum_rows.values, boundaries[1:-1])
    folds = np.split(unique_dates, fold_edges)
    return [dates.isin(f).sum() for f in folds]


def main():
    df = pd.read_parquet(FEATURES_PATH)
    date_sizes = date_balanced_fold_sizes(df["date"])
    row_sizes = row_balanced_fold_sizes(df["date"])

    print("Fold  Date-balanced rows  Row-balanced rows")
    for i, (d, r) in enumerate(zip(date_sizes, row_sizes)):
        print(f"{i+1:<6}{d:<20,}{r:,}")

    x = np.arange(1, N_SPLITS + 1)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, date_sizes, width, label="Date-balanced (Paper 1 scheme)", color="#c0392b", alpha=0.8)
    ax.bar(x + width/2, row_sizes, width, label="Row-balanced (Paper 2 scheme)", color="#1f4e79", alpha=0.8)
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Row Count")
    ax.set_title("Fold Size: Date-Balanced vs. Row-Balanced Splitting\n(Same Final Cleaned Dataset)")
    ax.legend()
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig("results/figures/fold_size_comparison.png", dpi=150)
    print("Saved results/figures/fold_size_comparison.png")


if __name__ == "__main__":
    main()