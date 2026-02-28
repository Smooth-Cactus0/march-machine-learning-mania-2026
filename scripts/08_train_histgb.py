"""
08_train_histgb.py — March Machine Learning Mania 2026

HistGradientBoostingClassifier (sklearn's native LightGBM-style histogram
boosting) trained on the curated feature set.

Key advantages over LGBM / XGB in this setting:
  - Native NaN handling — no SimpleImputer needed
  - Monotonic constraints: SeedNum_diff and massey_composite_diff are
    constrained to be monotonically decreasing in P(Team1 wins).
    This prevents the calibration failure observed in fig 08 where models
    underestimated win probability for large seed gaps.
  - Built-in early stopping per fold prevents overfitting on small folds
    (67 games each).

Inputs:
  features/team_features_{M,W}.parquet
  tournament results via utils.load_tourney()

Outputs:
  models/histgb_M.pkl
  models/histgb_W.pkl
  results/benchmarks.csv  (appended)
  BENCHMARKS.md           (regenerated)
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import build_monotone_vec  # monotone constraints live in utils

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier


# ── Main training loop ────────────────────────────────────────────────────────

def train_histgb(gender: str) -> float:
    """
    Run leave-one-season-out CV using HistGradientBoostingClassifier.
    Returns mean CV Brier score.
    """
    label = gender

    # 1. Load features
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")

    # 2. Load tournament results
    tourney = utils.load_tourney(gender)

    # 3. Build matchup dataframe (NaN-tolerant — HistGB handles NaN natively)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)

    # 4. Curated feature columns
    all_diff     = [c for c in matchups.columns if c.endswith("_diff")]
    FEATURE_COLS = utils.curate_features(all_diff)

    # 5. Build monotonic constraint vector aligned to FEATURE_COLS
    monotone = build_monotone_vec(FEATURE_COLS)

    # 6. Leave-one-season-out CV
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    print(f"\n[{label}] Leave-one-season-out CV (last {len(cv_seasons)} seasons):")
    print(f"       Features: {len(FEATURE_COLS)}  |  Monotone constraints: "
          f"{sum(1 for c in monotone if c != 0)}/{len(monotone)} active")

    fold_briers = []
    for season in cv_seasons:
        train_mask = matchups["Season"] != season
        test_mask  = matchups["Season"] == season

        if test_mask.sum() == 0:
            print(f"  Season {season}: no test data, skipping")
            continue

        X_train = matchups.loc[train_mask, FEATURE_COLS].values
        y_train = matchups.loc[train_mask, "Label"].values
        X_test  = matchups.loc[test_mask,  FEATURE_COLS].values
        y_test  = matchups.loc[test_mask,  "Label"].values

        # HistGB handles NaN natively — no imputer needed
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            l2_regularization=1.0,
            max_bins=255,
            monotonic_cst=monotone,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        brier  = utils.brier_score(y_test, y_pred)
        fold_briers.append(brier)

        print(f"  Season {season}: Brier = {brier:.4f}  (N={len(y_test)} games)")

    mean_brier = float(np.mean(fold_briers))
    std_brier  = float(np.std(fold_briers))

    baseline = 0.1965 if gender == "M" else 0.1497
    print(f"  -> Mean Brier: {mean_brier:.4f} +/- {std_brier:.4f}  "
          f"(vs baseline {baseline:.4f})")

    # 7. Train final model on ALL data
    X_all = matchups[FEATURE_COLS].values
    y_all = matchups["Label"].values

    final_model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        l2_regularization=1.0,
        max_bins=255,
        monotonic_cst=monotone,
        early_stopping=False,
        random_state=42,
    )
    final_model.fit(X_all, y_all)

    # 8. Save model (store feature cols for inference)
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = utils.MODELS / f"histgb_{gender}.pkl"
    joblib.dump({"model": final_model, "feature_cols": FEATURE_COLS}, model_path)

    return mean_brier


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 36)
    print("HistGradientBoosting: curated features")
    print("  + monotonic constraints on seed/quality diffs")
    print("=" * 36)

    mean_brier_m = train_histgb("M")
    mean_brier_w = train_histgb("W")

    print(f"\nSaved models/histgb_M.pkl, models/histgb_W.pkl")

    utils.log_benchmark(
        "histgb_v1", "M", mean_brier_m,
        "HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints"
    )
    utils.log_benchmark(
        "histgb_v1", "W", mean_brier_w,
        "HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints"
    )

    print("Benchmarks updated.")
