"""
13_meta_learner.py -- March Machine Learning Mania 2026

Stacking meta-learner: LogisticRegression on logit-transformed OOF predictions
from all 4 Optuna-tuned base models.

Why logit inputs? Base model outputs p in (0,1). Logistic regression operates
in log-odds space natively: sigmoid(w . logit(p) + b). Transforming inputs to
logit(p) = log(p/(1-p)) makes the combination linear in log-odds, which is the
natural parameterisation for combining calibrated probability estimates.

No Platt calibration applied after the meta-learner -- LogisticRegression
minimises log-loss and already produces well-calibrated probabilities.

Inputs:  results/oof_preds_{M,W}.npz  (from 10_ensemble.py)
         models/{model}_tuned_{M,W}.pkl

Outputs: results/meta_learner_{M,W}.pkl
         submissions/submission_meta_stage1.csv
         submissions/submission_meta_stage2.csv
         results/benchmarks.csv / BENCHMARKS.md
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

MODELS    = ["lgbm", "xgb", "catboost", "histgb"]
META_C    = 1.0    # L2 regularisation -- prevents multicollinearity instability
CLIP_LOW  = 0.025
CLIP_HIGH = 0.975
EPS       = 1e-7


def to_logit(p: np.ndarray) -> np.ndarray:
    """Transform probability array to log-odds. Clips to avoid ±inf."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ── Meta-learner training ─────────────────────────────────────────────────────

def train_meta(gender: str) -> dict:
    """
    Train LogisticRegression meta-learner on OOF predictions.
    Returns artifact dict with meta model + evaluation metrics.
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Meta-learner -- {label}")
    print(f"{'='*52}")

    # Load OOF arrays produced by 10_ensemble.py
    npz    = np.load(utils.RESULTS / f"oof_preds_{gender}.npz")
    y_true = npz["y_true"]
    P      = np.column_stack([npz[f"y_{m}"] for m in MODELS])  # (N, 4)
    L      = to_logit(P)                                        # logit-space features

    # Reference: simple mean Brier
    mean_pred  = P.mean(axis=1)
    brier_mean = utils.brier_score(y_true, mean_pred)

    # Train meta-learner on logit-space inputs
    meta = LogisticRegression(C=META_C, solver="lbfgs", max_iter=2000)
    meta.fit(L, y_true)

    meta_pred  = meta.predict_proba(L)[:, 1]
    brier_meta = utils.brier_score(y_true, meta_pred)

    delta = brier_meta - brier_mean
    print(f"  Mean ensemble Brier = {brier_mean:.4f}")
    print(f"  Meta-learner  Brier = {brier_meta:.4f}  (delta = {delta:+.4f})")

    w = meta.coef_[0]
    print(f"  Weights (logit space):")
    for model_name, wi in zip(MODELS, w):
        print(f"    {model_name:<12} {wi:+.4f}")
    print(f"    intercept    {meta.intercept_[0]:+.4f}")

    return {
        "meta":       meta,
        "models":     MODELS,
        "brier_meta": float(brier_meta),
        "brier_mean": float(brier_mean),
    }


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_features(sub_df: pd.DataFrame, features_df: pd.DataFrame,
                    feat_cols: list) -> pd.DataFrame:
    """
    Vectorised feature-diff join for submission rows.
    Mirrors build_features() in 12_predict.py.
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

    return pd.DataFrame(
        {fc: t1[rc].values - t2[rc].values
         for fc, rc in zip(feat_cols, raw_cols)},
        index=sub_df.index,
    )


def predict_meta_gender(sub_df: pd.DataFrame, gender: str,
                        artifact: dict) -> np.ndarray:
    """
    Predict for one gender's submission slice using the meta-learner.
    No Platt calibration -- LogisticRegression already produces calibrated probs.
    """
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    meta     = artifact["meta"]

    # Get feature cols from the first base model (all share the same curated set)
    first_art = joblib.load(utils.MODELS / f"{MODELS[0]}_tuned_{gender}.pkl")
    feat_cols = first_art["feature_cols"]
    X_df      = _build_features(sub_df, features, feat_cols)

    # Collect per-model base predictions
    base_preds = []
    for model_name in MODELS:
        art = joblib.load(utils.MODELS / f"{model_name}_tuned_{gender}.pkl")
        X   = X_df[art["feature_cols"]]
        if "imputer" in art:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_imp = art["imputer"].transform(X)
            p = art["model"].predict_proba(X_imp)[:, 1]
        else:
            p = art["model"].predict_proba(X.values)[:, 1]
        base_preds.append(p)

    # Stack → logit → meta-learner
    P   = np.column_stack(base_preds)   # (N, 4) probabilities
    L   = to_logit(P)                   # (N, 4) logit-space
    cal = meta.predict_proba(L)[:, 1]

    return np.clip(cal, CLIP_LOW, CLIP_HIGH)


# ── Submission generator ───────────────────────────────────────────────────────

def generate_meta_submission(stage: int, artifacts: dict) -> pd.DataFrame:
    """Parse sample submission, predict with meta-learner, return DataFrame."""
    print(f"\nGenerating meta Stage {stage} submission...")

    sub    = utils.load_sample_submission(stage).copy()
    parsed = sub["ID"].str.split("_", expand=True).astype(int)
    sub["Season"]  = parsed[0]
    sub["Team1ID"] = parsed[1]
    sub["Team2ID"] = parsed[2]
    sub["gender"]  = np.where(sub["Team1ID"] < 3000, "M", "W")
    sub["Pred"]    = 0.5

    for gender in ["M", "W"]:
        mask  = sub["gender"] == gender
        sub_g = sub.loc[mask, ["Season", "Team1ID", "Team2ID"]].copy()
        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label}: {mask.sum():,} matchups...", end=" ", flush=True)

        preds = predict_meta_gender(sub_g, gender, artifacts[gender])
        sub.loc[mask, "Pred"] = preds

        brier_ref = artifacts[gender]["brier_meta"]
        print(f"done  (mean={preds.mean():.3f}, meta OOF Brier={brier_ref:.4f})")

    return sub[["ID", "Pred"]]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    artifacts = {}

    for gender in ["M", "W"]:
        artifact = train_meta(gender)
        artifacts[gender] = artifact

        # Save artifact
        utils.RESULTS.mkdir(parents=True, exist_ok=True)
        path = utils.RESULTS / f"meta_learner_{gender}.pkl"
        joblib.dump(artifact, path)
        print(f"  Saved {path.name}")

        # Log benchmark
        utils.log_benchmark(
            "meta_v1", gender, artifact["brier_meta"],
            f"LogReg meta-learner on logit OOF "
            f"(mean={artifact['brier_mean']:.4f} -> meta={artifact['brier_meta']:.4f})"
        )

    # Generate submission files
    utils.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for stage in [1, 2]:
        result = generate_meta_submission(stage, artifacts)

        assert result["Pred"].between(CLIP_LOW, CLIP_HIGH).all(), \
            f"Stage {stage}: predictions out of [{CLIP_LOW}, {CLIP_HIGH}] range!"
        assert result["Pred"].isna().sum() == 0, \
            f"Stage {stage}: NaN predictions found!"

        out = utils.SUBMISSIONS / f"submission_meta_stage{stage}.csv"
        result.to_csv(out, index=False)
        print(f"  Saved {out.name}  ({len(result):,} rows)")

    print("\n" + "="*52)
    print("META-LEARNER SUMMARY")
    print("="*52)
    for gender, art in artifacts.items():
        label = "Men's" if gender == "M" else "Women's"
        delta = art["brier_meta"] - art["brier_mean"]
        print(f"  {label:<10} mean={art['brier_mean']:.4f}  "
              f"meta={art['brier_meta']:.4f}  delta={delta:+.4f}")
    print("Benchmarks updated.")
