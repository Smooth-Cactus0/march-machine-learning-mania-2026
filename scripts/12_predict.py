"""
12_predict.py -- March Machine Learning Mania 2026

Generate Stage 1 (seasons 2022-2025, ~519k rows) and
Stage 2 (season 2026, ~132k rows) submission CSV files.

Pipeline per matchup row:
  parse ID -> gender split -> feature diff -> 4-model ensemble -> Platt calibration -> clip

Inputs:
  models/{model}_tuned_{M,W}.pkl  -- Optuna-tuned model artifacts
  results/calibrator_{M,W}.pkl    -- Platt scaling objects
  results/ensemble_config.json    -- strategy per gender
  features/team_features_{M,W}.parquet

Outputs:
  submissions/submission_stage1.csv
  submissions/submission_stage2.csv
"""

import sys
import json
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(sub_df: pd.DataFrame, features_df: pd.DataFrame,
                   feat_cols: list) -> pd.DataFrame:
    """
    Build feature-diff DataFrame for a submission slice.

    sub_df must have columns: Season, Team1ID, Team2ID (with original index).
    feat_cols is a list of '*_diff' column names (e.g. 'SeedNum_diff').

    Returns a DataFrame of shape (N, len(feat_cols)) aligned to sub_df's index.
    NaN values are preserved; models handle them via their saved imputer or
    natively (CatBoost/HistGB).
    """
    raw_cols = [c.replace("_diff", "") for c in feat_cols]
    feat     = features_df.set_index(["Season", "TeamID"])

    t1 = (sub_df[["Season", "Team1ID"]]
          .rename(columns={"Team1ID": "TeamID"})
          .join(feat, on=["Season", "TeamID"])
          .drop(columns="TeamID"))

    t2 = (sub_df[["Season", "Team2ID"]]
          .rename(columns={"Team2ID": "TeamID"})
          .join(feat, on=["Season", "TeamID"])
          .drop(columns="TeamID"))

    diff_df = pd.DataFrame(
        {fc: t1[rc].values - t2[rc].values
         for fc, rc in zip(feat_cols, raw_cols)},
        index=sub_df.index,
    )
    return diff_df


# ── Ensemble + calibration predictor ──────────────────────────────────────────

def predict_gender(sub_df: pd.DataFrame, gender: str,
                   ens_config: dict, calibrator: dict) -> np.ndarray:
    """
    Run ensemble prediction for one gender's slice of the submission.

    sub_df: DataFrame with Season, Team1ID, Team2ID (and original index).
    Returns calibrated probability array of shape (N,), clipped to [0.025, 0.975].
    """
    # Load features once for this gender
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    cfg      = ens_config[gender]
    models   = cfg["models"]    # ["lgbm", "xgb", "catboost", "histgb"]
    strategy = cfg["strategy"]  # "mean" or "rank_mean"

    # Get canonical feature cols from the first model artifact
    first_art = joblib.load(utils.MODELS / f"{models[0]}_tuned_{gender}.pkl")
    feat_cols = first_art["feature_cols"]

    # Build diff matrix once — shared across all models for this gender
    X_df = build_features(sub_df, features, feat_cols)

    # Collect per-model predictions
    preds = []
    for model_name in models:
        art            = joblib.load(utils.MODELS / f"{model_name}_tuned_{gender}.pkl")
        model_feat_cols = art["feature_cols"]  # re-select in case order differs
        X = X_df[model_feat_cols]

        if "imputer" in art:
            # LGBM / XGB: impute NaN with training medians (no leakage)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_imp = art["imputer"].transform(X)
            p = art["model"].predict_proba(X_imp)[:, 1]
        else:
            # CatBoost / HistGB: handle NaN natively
            p = art["model"].predict_proba(X.values)[:, 1]

        preds.append(p)

    # Combine predictions
    if strategy == "mean":
        ens_pred = np.mean(preds, axis=0)
    else:  # rank_mean
        rank_preds = [rankdata(p) / len(p) for p in preds]
        ens_pred   = np.mean(rank_preds, axis=0)

    # Platt calibration: apply learned logit rescaling
    eps      = 1e-7
    log_odds = np.log(
        np.clip(ens_pred, eps, 1 - eps) / np.clip(1 - ens_pred, eps, 1 - eps)
    )
    cal_pred = calibrator["platt"].predict_proba(log_odds.reshape(-1, 1))[:, 1]

    # Clip to competition-safe range (avoids extreme overconfidence)
    return np.clip(cal_pred, 0.025, 0.975)


# ── Submission generator ───────────────────────────────────────────────────────

def generate_submission(stage: int) -> pd.DataFrame:
    """
    Parse sample submission, predict all rows, return completed DataFrame.

    stage: 1 (seasons 2022-2025, ~519k rows) or 2 (season 2026, ~132k rows).
    """
    print(f"\nGenerating Stage {stage} submission...")

    # Load shared config + calibrators once
    ens_config  = json.load(open(utils.RESULTS / "ensemble_config.json"))
    calibrators = {
        "M": joblib.load(utils.RESULTS / "calibrator_M.pkl"),
        "W": joblib.load(utils.RESULTS / "calibrator_W.pkl"),
    }

    # Parse submission IDs into Season / Team1ID / Team2ID
    sub    = utils.load_sample_submission(stage).copy()
    parsed = sub["ID"].str.split("_", expand=True).astype(int)
    sub["Season"]  = parsed[0]
    sub["Team1ID"] = parsed[1]
    sub["Team2ID"] = parsed[2]
    # Gender: Men's TeamIDs are 1xxx (<3000), Women's are 3xxx (>=3000)
    sub["gender"]  = np.where(sub["Team1ID"] < 3000, "M", "W")
    sub["Pred"]    = 0.5  # default; overwritten below for all rows

    for gender in ["M", "W"]:
        mask  = sub["gender"] == gender
        sub_g = sub.loc[mask, ["Season", "Team1ID", "Team2ID"]].copy()

        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label}: {mask.sum():,} matchups...", end=" ", flush=True)

        preds = predict_gender(sub_g, gender, ens_config, calibrators[gender])
        sub.loc[mask, "Pred"] = preds

        brier_ref = ens_config[gender]["cv_brier"]
        print(f"done  (mean pred={preds.mean():.3f}, CV Brier ref={brier_ref:.4f})")

    return sub[["ID", "Pred"]]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    utils.SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    for stage in [1, 2]:
        result = generate_submission(stage)

        # Sanity checks before saving
        assert result["Pred"].between(0.025, 0.975).all(), \
            f"Stage {stage}: predictions out of [0.025, 0.975] range!"
        assert result["Pred"].isna().sum() == 0, \
            f"Stage {stage}: NaN predictions found!"

        out_path = utils.SUBMISSIONS / f"submission_stage{stage}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved {out_path.name}  ({len(result):,} rows)")

    print("\nDone. Submission files ready:")
    print("  submissions/submission_stage1.csv")
    print("  submissions/submission_stage2.csv")
