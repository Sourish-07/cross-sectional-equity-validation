import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_parquet("data/processed/pbo_results.parquet")

plt.figure(figsize=(7,7))
plt.scatter(df["is_sharpe"], df["oos_sharpe"], alpha=0.6)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("In-Sample Sharpe")
plt.ylabel("Out-of-Sample Sharpe")
plt.title("IS vs OOS Sharpe (CPCV)")
plt.tight_layout()
plt.savefig("figures/pbo_scatter.png")
plt.close()
