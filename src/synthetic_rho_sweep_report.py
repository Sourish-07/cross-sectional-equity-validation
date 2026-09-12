"""
Run this AFTER synthetic_rho_sweep.py (reads results/synthetic/sweep_results.json,
which that script writes). Produces:
  - results/synthetic/synthetic_rho_sweep.png   (figure)
  - results/synthetic/synthetic_sweep_results.csv (flat table, one row per rho)

Run both from your repo root:
    python src/synthetic_rho_sweep.py
    python src/synthetic_rho_sweep_report.py
"""
import json
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = "results/synthetic"

with open(os.path.join(IN_DIR, "sweep_results.json")) as f:
    d = json.load(f)

rhos = d["rho"]
grid_sizes = [str(m) for m in d["grid_sizes"]]

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"7": "#4C72B0", "15": "#DD8452", "30": "#55A868"}
for M in grid_sizes:
    means = d["multi_trial"][M]["pbo_mean"]
    lo = d["multi_trial"][M]["pbo_ci_lo"]
    hi = d["multi_trial"][M]["pbo_ci_hi"]
    ax.plot(rhos, means, label=f"Multi-trial PBO (M={M})", color=colors.get(M), linewidth=2)
    ax.fill_between(rhos, lo, hi, color=colors.get(M), alpha=0.12)

sm_mean = d["single_model"]["mean"]
sm_lo, sm_hi = d["single_model"]["ci95"]
ax.axhline(sm_mean, color="black", linestyle="--", linewidth=2, label="Single-model PBO (reference)")
ax.axhspan(sm_lo, sm_hi, color="gray", alpha=0.08)

ax.set_xlabel(r"Correlation dial $\rho$")
ax.set_ylabel("PBO")
ax.set_title("Synthetic sweep: multi-trial PBO vs. single-model PBO\n(no real data; $\\rho$ = share of score variance from a shared factor)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 0.55)
ax.grid(alpha=0.25)
fig.tight_layout()
fig_path = os.path.join(IN_DIR, "synthetic_rho_sweep.png")
fig.savefig(fig_path, dpi=200)
print(f"Saved {fig_path}")

# ---------------- CSV ----------------
csv_path = os.path.join(IN_DIR, "synthetic_sweep_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["rho", "single_model_pbo_flat_reference"]
    for M in grid_sizes:
        header += [f"multi_trial_pbo_M{M}", f"multi_trial_pbo_ci_lo_M{M}",
                   f"multi_trial_pbo_ci_hi_M{M}", f"measured_pairwise_corr_M{M}"]
    w.writerow(header)
    for i, rho in enumerate(rhos):
        row = [rho, sm_mean]
        for M in grid_sizes:
            dd = d["multi_trial"][M]
            row += [dd["pbo_mean"][i], dd["pbo_ci_lo"][i], dd["pbo_ci_hi"][i], dd["measured_pairwise_corr"][i]]
        w.writerow(row)
print(f"Saved {csv_path}")