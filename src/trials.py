"""
Expandable trial grid. The original 7-trial grid is preserved as an exact
PREFIX of this list, so that get_trials(7) reproduces the original trial
set exactly -- this makes the trial-count sensitivity sweep (7 -> 15 -> 30)
a genuine apples-to-apples nested comparison rather than a different
random set at each size.
"""
from itertools import product
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42


def _sgd_trial(alpha):
    return {
        "trial_id": f"sgd_alpha_{alpha}",
        "family": "linear",
        "build": lambda a=alpha: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=a, max_iter=1,
            learning_rate="optimal", warm_start=True, random_state=RANDOM_STATE
        ),
        "supports_partial_fit": True,
    }


def _hgb_trial(depth, n_iter):
    return {
        "trial_id": f"hgb_depth_{depth}_iter_{n_iter}",
        "family": "nonlinear",
        "build": lambda d=depth, it=n_iter: HistGradientBoostingClassifier(
            max_depth=d, max_iter=it, random_state=RANDOM_STATE
        ),
        "supports_partial_fit": False,
    }


def build_full_trial_grid():
    trials = []
    # --- Original 7 trials, kept as an exact prefix ---
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
        trials.append(_sgd_trial(alpha))
    for depth, n_iter in [(3, 50), (5, 50), (3, 100)]:
        trials.append(_hgb_trial(depth, n_iter))

    # --- Additional SGD alphas (extends linear family) ---
    for alpha in [1e-6, 3e-5, 3e-4, 3e-3, 3e-2, 1e-1]:
        trials.append(_sgd_trial(alpha))

    # --- Additional HGB configs (extends nonlinear family) ---
    extra_hgb_configs = [
        (2, 50), (4, 50), (6, 50),
        (3, 25), (3, 75), (3, 150),
        (5, 25), (5, 75), (5, 100),
        (2, 100), (4, 100), (6, 100),
        (2, 25), (4, 25), (6, 25),
        (4, 75),(6, 150),
    ]
    for depth, n_iter in extra_hgb_configs:
        trials.append(_hgb_trial(depth, n_iter))

    return trials


_FULL_GRID = build_full_trial_grid()


def get_trials(n):
    """Returns the first n trials from the full grid. n=7 reproduces the
    original grid exactly. Raises if n exceeds the number of defined trials."""
    if n > len(_FULL_GRID):
        raise ValueError(f"Only {len(_FULL_GRID)} trials defined; requested {n}")
    return _FULL_GRID[:n]


# Backwards-compatible default (matches original 7-trial behavior)
TRIALS = get_trials(7)

if __name__ == "__main__":
    for n in [7, 15, 30]:
        print(f"--- get_trials({n}) ---")
        for t in get_trials(n):
            print(f"  {t['trial_id']} ({t['family']})")