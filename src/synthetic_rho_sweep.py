"""
Synthetic controlled experiment: reproduces the single-model vs multi-trial
PBO disagreement under a known, dial-able correlation parameter -- no real
data, no finance-specific mechanism.

MECHANISM (why a naive design fails, and what this one does instead)
----------------------------------------------------------------------
A common shock added IDENTICALLY to every model within a split cancels out
of any *within-split ranking* comparison (argmax for selection, or
above/below-median for scoring) -- so it can't move a rank-based PBO at all.
For the correlation dial to actually matter, models must differ in their
EXPOSURE to the shared factor, not just share it identically.

So: each model m has a fixed sensitivity b_m to a per-split regime factor
Z_s. Model m's score (both its in-sample slice and its held-out slice,
which are close in time / share the same embargoed regime) is:

    score_{s,m} = b_m * Z_s + noise_{s,m}

noise is drawn FRESH and independently for the IS slice and the OOS slice
(so a model's own IS score does not trivially equal its own OOS score).
No model has any genuine cross-period predictive skill -- b_m is just a
fixed, arbitrary sensitivity to a factor that happens to persist over the
short embargo window, not a real edge.

rho = share of a model's score variance coming from the shared factor,
      i.e. rho = Var(b_m*Z_s) / (Var(b_m*Z_s) + 1)   [noise variance = 1]

At rho -> 0: scores are pure independent noise. Whichever model wins
in-sample is a uniformly random draw, so it lands below the out-of-sample
median about half the time. Multi-trial PBO -> ~0.5 (uninformative, as it
should be, since there is no real skill anywhere).

At rho -> 1: scores are dominated by b_m*Z_s. The IS winner is (almost)
always the model with the highest b_m*Z_s that split -- and since OOS is
dominated by the SAME b_m*Z_s, it is also the OOS winner. Multi-trial PBO
-> ~0, even though nothing here is "skill": it's leverage to a shared,
transient factor.

The single-model diagnostic is a separate, parallel computation with NO
b_m/Z_s machinery at all: one fixed "model" whose IS and OOS scores are
literally independent draws (the textbook null of "no generalizable
relationship"). It has nothing to get fooled by, so it should sit flat
around chance (~0.5) regardless of rho.
"""

import numpy as np
import json
import os

# Run this from your repo root (same place you run the other src/ scripts from).
# Follows the same convention as plots_paper2.py: paper-2-only outputs live
# under results/<subfolder>, separate from paper 1's top-level figures/.
OUT_DIR = "results/synthetic"

RNG_SEED = 20260910
N_SPLITS = 45          # matches the real study's 45 combinatorial splits
N_REPS = 500            # independent replications of the whole 45-split study, per (rho, M)
GRID_SIZES = [7, 15, 30]  # matches the real study's trial-grid sizes
RHOS = np.round(np.linspace(0.0, 0.95, 20), 3)


def run_multi_trial_once(rng, M, rho, n_splits=N_SPLITS):
    """One replication: n_splits splits, M models. Returns (pbo, mean_pairwise_corr)."""
    b = rng.normal(loc=1.0, scale=0.35, size=M)          # fixed per-model sensitivities
    sigma_z = np.sqrt(rho / (1 - rho)) if rho < 1 else 1e6  # so Var(b*Z)/(Var(b*Z)+1) = rho on average

    Z = rng.normal(0, sigma_z, size=n_splits)              # shared per-split regime factor
    noise_is = rng.normal(0, 1, size=(n_splits, M))
    noise_oos = rng.normal(0, 1, size=(n_splits, M))

    IS = np.outer(Z, b) + noise_is
    OOS = np.outer(Z, b) + noise_oos

    winners = np.argmax(IS, axis=1)
    oos_winner = OOS[np.arange(n_splits), winners]
    oos_median = np.median(OOS, axis=1)
    pbo = np.mean(oos_winner < oos_median)

    corr = np.corrcoef(OOS, rowvar=False)
    iu = np.triu_indices(M, k=1)
    mean_pairwise_corr = np.mean(corr[iu])

    return pbo, mean_pairwise_corr


def run_single_model_once(rng, n_splits=N_SPLITS):
    """No b_m, no Z_s: literally independent IS/OOS draws for one fixed model.
    'Trials' here are resampled paths of that SAME single model, standard PBO recipe."""
    IS = rng.normal(0, 1, size=n_splits)
    OOS = rng.normal(0, 1, size=n_splits)
    order = np.argsort(-IS)                 # best IS-ranked path first
    top_half = order[: n_splits // 2]
    below_median = OOS[top_half] < np.median(OOS)
    return np.mean(below_median)


def main():
    rng = np.random.default_rng(RNG_SEED)
    results = {"rho": list(map(float, RHOS)), "grid_sizes": GRID_SIZES, "multi_trial": {}, "single_model": None}

    # single-model reference (independent of rho and M by construction)
    sm_pbos = [run_single_model_once(rng) for _ in range(N_REPS)]
    results["single_model"] = {
        "mean": float(np.mean(sm_pbos)),
        "ci95": [float(np.percentile(sm_pbos, 2.5)), float(np.percentile(sm_pbos, 97.5))],
    }

    for M in GRID_SIZES:
        pbo_means, pbo_los, pbo_his, corr_means = [], [], [], []
        for rho in RHOS:
            pbos, corrs = [], []
            for _ in range(N_REPS):
                p, c = run_multi_trial_once(rng, M, rho)
                pbos.append(p)
                corrs.append(c)
            pbo_means.append(float(np.mean(pbos)))
            pbo_los.append(float(np.percentile(pbos, 2.5)))
            pbo_his.append(float(np.percentile(pbos, 97.5)))
            corr_means.append(float(np.mean(corrs)))
        results["multi_trial"][str(M)] = {
            "pbo_mean": pbo_means, "pbo_ci_lo": pbo_los, "pbo_ci_hi": pbo_his,
            "measured_pairwise_corr": corr_means,
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    print("Single-model PBO (flat reference):", results["single_model"])
    print()
    for M in GRID_SIZES:
        d = results["multi_trial"][str(M)]
        print(f"M={M}")
        for i, rho in enumerate(RHOS):
            print(f"  rho={rho:.2f}  multi-trial PBO={d['pbo_mean'][i]:.3f}  "
                  f"[{d['pbo_ci_lo'][i]:.3f},{d['pbo_ci_hi'][i]:.3f}]  "
                  f"measured pairwise corr={d['measured_pairwise_corr'][i]:.3f}")
        print()


if __name__ == "__main__":
    main()