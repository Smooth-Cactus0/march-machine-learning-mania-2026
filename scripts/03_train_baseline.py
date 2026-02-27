"""
03_train_baseline.py — March Machine Learning Mania 2026

Seed-difference logistic regression baseline.

Inputs:
  features/team_features_{M,W}.parquet  (for SeedNum column)
  tournament results via utils.load_tourney()

Outputs:
  models/baseline_M.pkl
  models/baseline_W.pkl
  results/benchmarks.csv  (appended)
  BENCHMARKS.md           (regenerated)
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ── Main training loop ────────────────────────────────────────────────────────

def train_baseline(gender: str) -> float:
    """
    Run leave-one-season-out CV on seed-diff logistic regression.
    Returns mean CV Brier score.
    """
    label = "M" if gender == "M" else "W"

    # 1. Load features — use only SeedNum so make_matchup_df keeps all games
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    seed_only = features[["Season", "TeamID", "SeedNum"]].copy()

    # 2. Load tournament results
    tourney = utils.load_tourney(gender)

    # 3. Build matchup dataframe (SeedNum_diff will be the only feature column)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df(tourney, seed_only)

    # Safety: drop any remaining NaN in SeedNum_diff (should be zero)
    matchups = matchups.dropna(subset=["SeedNum_diff"]).reset_index(drop=True)

    FEATURE_COLS = ["SeedNum_diff"]
    X_all = matchups[FEATURE_COLS].values
    y_all = matchups["Label"].values

    # 4. Leave-one-season-out CV on last 10 seasons
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    print(f"\n[{label}] Leave-one-season-out CV (last {len(cv_seasons)} seasons):")

    fold_briers = []
    for season in cv_seasons:
        train_mask = matchups["Season"] != season
        test_mask  = matchups["Season"] == season

        X_train = matchups.loc[train_mask, FEATURE_COLS].values
        y_train = matchups.loc[train_mask, "Label"].values
        X_test  = matchups.loc[test_mask,  FEATURE_COLS].values
        y_test  = matchups.loc[test_mask,  "Label"].values

        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        brier  = utils.brier_score(y_test, y_pred)
        fold_briers.append(brier)

        print(f"  Season {season}: Brier = {brier:.4f}  (N={len(y_test)} games)")

    mean_brier = float(np.mean(fold_briers))
    std_brier  = float(np.std(fold_briers))
    print(f"  -> Mean Brier: {mean_brier:.4f} +/- {std_brier:.4f}")

    # 5. Train final model on ALL data
    final_model = LogisticRegression(C=1.0, max_iter=1000)
    final_model.fit(X_all, y_all)

    # 6. Save final model
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = utils.MODELS / f"baseline_{gender}.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nSaved {model_path.relative_to(utils.ROOT)}")

    return mean_brier


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Baseline: seed-diff logistic regression")
    print("=" * 60)

    mean_brier_m = train_baseline("M")
    mean_brier_w = train_baseline("W")

    # 7. Log benchmarks
    utils.log_benchmark("baseline_logreg", "M", mean_brier_m, "seed-diff only")
    utils.log_benchmark("baseline_logreg", "W", mean_brier_w, "seed-diff only")

    print("\nBenchmarks updated.")
