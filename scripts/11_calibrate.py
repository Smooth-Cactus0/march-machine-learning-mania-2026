"""
11_calibrate.py -- March Machine Learning Mania 2026

Platt scaling calibration applied to ensemble OOF predictions.
Loads results/oof_preds_{M,W}.npz produced by 10_ensemble.py.

Platt scaling fits LogisticRegression on logit(ensemble_pred) -> y_true.
This learns the optimal linear rescaling:
    p_cal = sigmoid(a * logit(p_raw) + b)
With ~670 OOF samples and only 2 parameters it is very robust.

Outputs:
  results/calibrator_{M,W}.pkl  -- {"platt": fitted LogisticRegression}
  results/benchmarks.csv / BENCHMARKS.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression


def platt_calibrate(gender: str) -> tuple:
    """
    Fit Platt scaling on ensemble OOF predictions.
    Returns (brier_before, brier_after).
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Calibration -- {label}")
    print(f"{'='*52}")

    # Load OOF arrays produced by 10_ensemble.py
    npz      = np.load(utils.RESULTS / f"oof_preds_{gender}.npz")
    y_true   = npz["y_true"]
    ens_pred = npz["y_ensemble"]

    brier_before = utils.brier_score(y_true, ens_pred)
    print(f"  Pre-calibration  Brier = {brier_before:.4f}  (N={len(y_true)} OOF games)")

    # Platt scaling: logistic regression on logit(p) -> labels
    eps      = 1e-7
    log_odds = np.log(
        np.clip(ens_pred, eps, 1 - eps) / np.clip(1 - ens_pred, eps, 1 - eps)
    )
    platt = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    platt.fit(log_odds.reshape(-1, 1), y_true)

    cal_pred    = platt.predict_proba(log_odds.reshape(-1, 1))[:, 1]
    brier_after = utils.brier_score(y_true, cal_pred)

    delta = brier_after - brier_before
    print(f"  Post-calibration Brier = {brier_after:.4f}  (delta = {delta:+.4f})")
    print(f"  Platt params: slope={platt.coef_[0][0]:.4f}  "
          f"intercept={platt.intercept_[0]:.4f}")

    # Save calibrator
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    cal_path = utils.RESULTS / f"calibrator_{gender}.pkl"
    joblib.dump({"platt": platt}, cal_path)
    print(f"  Saved {cal_path.name}")

    return brier_before, brier_after


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    for gender in ["M", "W"]:
        brier_before, brier_after = platt_calibrate(gender)
        results[gender] = (brier_before, brier_after)
        utils.log_benchmark(
            "calibrated_v1", gender, brier_after,
            f"Platt-scaled ensemble (pre={brier_before:.4f} post={brier_after:.4f})"
        )

    print("\n" + "="*52)
    print("CALIBRATION SUMMARY")
    print("="*52)
    for gender, (before, after) in results.items():
        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label:<10} {before:.4f} -> {after:.4f}  (delta={after-before:+.4f})")
    print("Saved results/calibrator_{{M,W}}.pkl")
    print("Benchmarks updated.")
