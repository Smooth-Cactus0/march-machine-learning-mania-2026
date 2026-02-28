"""
06_train_catboost.py — March Machine Learning Mania 2026

CatBoost classifier with full feature set.
CatBoost handles NaN natively via nan_mode="Min" — no imputation needed.

Inputs:
  features/team_features_{M,W}.parquet
  tournament results via utils.load_tourney()

Outputs:
  models/catboost_M.pkl
  models/catboost_W.pkl
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
from catboost import CatBoostClassifier


def make_matchup_df_catboost(
    tourney_df: pd.DataFrame, features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a matchup-level DataFrame for CatBoost training.

    Unlike utils.make_matchup_df, this version does NOT drop rows with NaN
    feature diffs — CatBoost handles NaN natively.  It does require at least
    one team to appear in features_df (games where both teams are missing are
    dropped).

    Columns that are entirely NaN across all matched rows (i.e., the feature
    was never populated for any tournament team) are removed so they don't
    consume a feature slot unnecessarily.
    """
    feat_cols = [c for c in features_df.columns if c not in ("Season", "TeamID")]

    df = tourney_df.copy()
    df["Team1ID"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["Team2ID"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["Label"]   = (df["WTeamID"] == df["Team1ID"]).astype(int)

    feat = features_df.set_index(["Season", "TeamID"])

    t1 = (
        df[["Season", "Team1ID"]]
        .rename(columns={"Team1ID": "TeamID"})
        .join(feat, on=["Season", "TeamID"])
        .drop(columns="TeamID")
    )
    t2 = (
        df[["Season", "Team2ID"]]
        .rename(columns={"Team2ID": "TeamID"})
        .join(feat, on=["Season", "TeamID"])
        .drop(columns="TeamID")
    )

    result = df[["Season", "Team1ID", "Team2ID", "Label"]].copy()
    for c in feat_cols:
        result[f"{c}_diff"] = t1[c].values - t2[c].values

    # Drop rows where BOTH teams are missing (diff is NaN because neither
    # team appeared in the feature set — no information at all).
    diff_cols = [f"{c}_diff" for c in feat_cols]
    all_nan_mask = result[diff_cols].isna().all(axis=1)
    n_dropped = all_nan_mask.sum()
    if n_dropped:
        warnings.warn(
            f"make_matchup_df_catboost: dropped {n_dropped} games where all "
            "feature diffs were NaN"
        )
    result = result[~all_nan_mask].reset_index(drop=True)

    # Remove columns that are entirely NaN after joining (feature never populated
    # for tournament teams in this dataset).
    always_nan = [c for c in diff_cols if result[c].isna().all()]
    if always_nan:
        result = result.drop(columns=always_nan)

    return result


# ── Main training loop ────────────────────────────────────────────────────────

def train_catboost(gender: str) -> float:
    """
    Run leave-one-season-out CV on CatBoost with full feature set.
    Returns mean CV Brier score.
    """
    label = gender

    # 1. Load features
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")

    # 2. Load tournament results
    tourney = utils.load_tourney(gender)

    # 3. Build matchup dataframe (NaN-tolerant version for CatBoost)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = make_matchup_df_catboost(tourney, features)

    # 4. Feature columns: curated set (see utils.CURATED_FEATURES)
    all_diff     = [c for c in matchups.columns if c.endswith("_diff")]
    FEATURE_COLS = utils.curate_features(all_diff)

    # 6. Leave-one-season-out CV on last 10 seasons
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    print(f"\n[{label}] Leave-one-season-out CV (last {len(cv_seasons)} seasons):")

    fold_briers = []
    for season in cv_seasons:
        train_mask = matchups["Season"] != season
        test_mask  = matchups["Season"] == season

        X_train = matchups.loc[train_mask, FEATURE_COLS]
        y_train = matchups.loc[train_mask, "Label"].values
        X_test  = matchups.loc[test_mask,  FEATURE_COLS]
        y_test  = matchups.loc[test_mask,  "Label"].values

        if len(y_test) == 0:
            print(f"  Season {season}: no test data, skipping")
            continue

        # CatBoost handles NaN natively — pass as numpy arrays to avoid column name issues
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=10,
            nan_mode="Min",
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train.values, y_train)

        y_pred = model.predict_proba(X_test.values)[:, 1]
        brier  = utils.brier_score(y_test, y_pred)
        fold_briers.append(brier)

        print(f"  Season {season}: Brier = {brier:.4f}  (N={len(y_test)} games)")

    mean_brier = float(np.mean(fold_briers))
    std_brier  = float(np.std(fold_briers))

    baseline = 0.1965 if gender == "M" else 0.1497
    print(f"  -> Mean Brier: {mean_brier:.4f} +/- {std_brier:.4f}  (vs baseline {baseline:.4f})")

    # 7. Train final model on ALL data — no imputer needed
    X_all = matchups[FEATURE_COLS]
    y_all = matchups["Label"].values

    final_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        min_data_in_leaf=10,
        nan_mode="Min",
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
    )
    final_model.fit(X_all.values, y_all)

    # 8. Save final model
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = utils.MODELS / f"catboost_{gender}.pkl"
    joblib.dump({"model": final_model}, model_path)

    return mean_brier


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 28)
    print("CatBoost: full feature set")
    print("=" * 28)

    mean_brier_m = train_catboost("M")
    mean_brier_w = train_catboost("W")

    print(f"\nSaved models/catboost_M.pkl, models/catboost_W.pkl")

    # Log benchmarks
    utils.log_benchmark(
        "catboost_v2", "M", mean_brier_m,
        "CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats"
    )
    utils.log_benchmark(
        "catboost_v2", "W", mean_brier_w,
        "CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats"
    )

    print("Benchmarks updated.")
