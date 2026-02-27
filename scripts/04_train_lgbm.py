"""
04_train_lgbm.py — March Machine Learning Mania 2026

LightGBM classifier trained on full feature set (team-level diffs).

Inputs:
  features/team_features_{M,W}.parquet
  tournament results via utils.load_tourney()

Outputs:
  models/lgbm_M.pkl        (dict with 'model' and 'imputer' keys)
  models/lgbm_W.pkl
  results/benchmarks.csv   (appended)
  BENCHMARKS.md            (regenerated)
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier


# ── Main training loop ────────────────────────────────────────────────────────

def train_lgbm(gender: str) -> float:
    """
    Run leave-one-season-out CV using LightGBM on the full feature set.
    Returns mean CV Brier score.
    """
    label = "M" if gender == "M" else "W"

    # 1. Load full feature set
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")

    # 2. Load tournament results
    tourney = utils.load_tourney(gender)

    # 3. Build matchup dataframe (NaN-tolerant: keeps rows with partial NaN
    #    so seasons like 2024/2025 — missing massey_SAG — are not silently
    #    dropped; SimpleImputer handles the NaN values within each CV fold).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)

    # 4. Identify feature columns: all *_diff columns except neutral_games_diff
    diff_cols = [c for c in matchups.columns if c.endswith("_diff")]
    # Drop neutral_games_diff — it's a count, not a quality signal
    diff_cols = [c for c in diff_cols if c != "neutral_games_diff"]

    FEATURE_COLS = diff_cols

    # 5. Leave-one-season-out CV on last 10 seasons
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    print(f"\n[{label}] Leave-one-season-out CV (last {len(cv_seasons)} seasons):")

    fold_briers = []
    for season in cv_seasons:
        train_mask = matchups["Season"] != season
        test_mask  = matchups["Season"] == season

        # Skip seasons with no tournament data in the matchups
        if test_mask.sum() == 0:
            continue

        X_train_df = matchups.loc[train_mask, FEATURE_COLS]
        y_train    = matchups.loc[train_mask, "Label"].values
        X_test_df  = matchups.loc[test_mask,  FEATURE_COLS]
        y_test     = matchups.loc[test_mask,  "Label"].values

        # Impute NaN with median computed on training data only (no leakage)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_df)
        X_test  = imputer.transform(X_test_df)

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        brier  = utils.brier_score(y_test, y_pred)
        fold_briers.append(brier)

        n_feat = len(FEATURE_COLS)
        print(f"  Season {season}: Brier = {brier:.4f}  (N={len(y_test)} games, {n_feat} features)")

    mean_brier = float(np.mean(fold_briers))
    std_brier  = float(np.std(fold_briers))

    # Baselines from 03_train_baseline.py
    baseline_ref = {"M": 0.1965, "W": 0.1497}
    print(f"  -> Mean Brier: {mean_brier:.4f} +/- {std_brier:.4f}  (vs baseline {baseline_ref[gender]:.4f})")

    # 6. Train final model on ALL data
    X_all_df = matchups[FEATURE_COLS]
    y_all    = matchups["Label"].values

    final_imputer = SimpleImputer(strategy="median")
    X_all_imputed = final_imputer.fit_transform(X_all_df)

    final_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X_all_imputed, y_all)

    # 7. Save model + imputer together
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = utils.MODELS / f"lgbm_{gender}.pkl"
    joblib.dump({"model": final_model, "imputer": final_imputer}, model_path)

    return mean_brier


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 28)
    print("LightGBM: full feature set")
    print("=" * 28)

    mean_brier_m = train_lgbm("M")
    mean_brier_w = train_lgbm("W")

    print(f"\nSaved models/lgbm_M.pkl, models/lgbm_W.pkl")

    # Log benchmarks
    utils.log_benchmark("lgbm_v1", "M", mean_brier_m, "LGBMClassifier n=500 lr=0.05 md=4")
    utils.log_benchmark("lgbm_v1", "W", mean_brier_w, "LGBMClassifier n=500 lr=0.05 md=4")

    print("Benchmarks updated.")
