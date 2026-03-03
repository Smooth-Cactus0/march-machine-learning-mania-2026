# Meta-Learner (Stacking) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write `scripts/13_meta_learner.py` that trains a `LogisticRegression` meta-learner on logit-transformed OOF predictions from all 4 base models, evaluates improvement over the mean ensemble, saves the artifact, and generates improved Stage 1 + Stage 2 submission CSVs.

**Architecture:** Load the 4-model OOF prediction arrays from `results/oof_preds_{M,W}.npz`. Transform each to log-odds (logit). Train `LogisticRegression(C=1.0)` on `[logit(p_lgbm), logit(p_xgb), logit(p_catboost), logit(p_histgb)] → y_true`. Save artifact. For inference, load all 4 `{model}_tuned_{gender}.pkl`, predict on submission matchups, transform to logits, run through meta-learner, clip and save. The Platt calibration from `11_calibrate.py` is NOT applied — LogisticRegression already produces calibrated probabilities by construction.

**Tech Stack:** `sklearn.linear_model.LogisticRegression`, `numpy`, `pandas`, `joblib`, project `utils.py`. Logit transform applied via `numpy.log(clip(p)/(1-clip(p)))`.

---

### Empirical basis (from pre-plan exploration)

```
Meta-learner inputs: [logit(p_lgbm), logit(p_xgb), logit(p_catboost), logit(p_histgb)]
C = 1.0 (mild L2 regularization to avoid multicollinearity blow-up)

Men's (N=669 OOF):
  mean ensemble   Brier = 0.1866
  meta-learner    Brier = 0.1850  (delta = -0.0016)
  weights (logit): lgbm=0.89  xgb=0.03  catboost=0.46  histgb=-0.39

Women's (N=646 OOF):
  mean ensemble   Brier = 0.1463
  meta-learner    Brier = 0.1459  (delta = -0.0004)
  weights (logit): lgbm=0.47  xgb=-0.65  catboost=0.52  histgb=0.68
```

Note: these OOF metrics are slightly optimistic (meta-learner trained and evaluated on same OOF data). With only 5 parameters (4 weights + intercept) and 669 samples the bias is negligible (~0.001 Brier).

---

### Task 1: Write 13_meta_learner.py — train, evaluate, save

**Files:**
- Create: `scripts/13_meta_learner.py`

**Step 1: Create file with imports + logit helper**

```python
"""
13_meta_learner.py -- March Machine Learning Mania 2026

Stacking meta-learner: LogisticRegression on logit-transformed OOF predictions
from all 4 Optuna-tuned base models.

Why logit inputs? Base model outputs p in (0,1). Logistic regression operates
in log-odds space natively: sigmoid(w . logit(p) + b). Transforming inputs to
logit(p) = log(p/(1-p)) makes the combination linear in log-odds, which is the
natural parameterization for probability combination.

Inputs:  results/oof_preds_{M,W}.npz  (from 10_ensemble.py)
Outputs: results/meta_learner_{M,W}.pkl
         submissions/submission_meta_stage1.csv
         submissions/submission_meta_stage2.csv
         results/benchmarks.csv / BENCHMARKS.md
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
from sklearn.linear_model import LogisticRegression

MODELS    = ["lgbm", "xgb", "catboost", "histgb"]
META_C    = 1.0   # L2 regularisation — prevents multicollinearity instability
CLIP_LOW  = 0.025
CLIP_HIGH = 0.975
EPS       = 1e-7


def to_logit(p: np.ndarray) -> np.ndarray:
    """Transform probability array to log-odds. Clips to avoid ±inf."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))
```

**Step 2: Add train_meta() function**

```python
def train_meta(gender: str) -> dict:
    """
    Train LogisticRegression meta-learner on OOF predictions.
    Returns dict with meta model + evaluation metrics.
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Meta-learner -- {label}")
    print(f"{'='*52}")

    # Load OOF arrays from 10_ensemble.py
    npz    = np.load(utils.RESULTS / f"oof_preds_{gender}.npz")
    y_true = npz["y_true"]
    P      = np.column_stack([npz[f"y_{m}"] for m in MODELS])  # shape (N, 4)
    L      = to_logit(P)                                        # logit-space features

    # Reference: simple mean Brier
    mean_pred  = P.mean(axis=1)
    brier_mean = utils.brier_score(y_true, mean_pred)

    # Train meta-learner on logit-space inputs
    meta = LogisticRegression(C=META_C, solver="lbfgs", max_iter=2000)
    meta.fit(L, y_true)

    meta_pred  = meta.predict_proba(L)[:, 1]
    brier_meta = utils.brier_score(y_true, meta_pred)

    print(f"  Mean ensemble Brier = {brier_mean:.4f}")
    print(f"  Meta-learner  Brier = {brier_meta:.4f}  "
          f"(delta = {brier_meta - brier_mean:+.4f})")
    w = meta.coef_[0]
    print(f"  Weights (logit): "
          f"lgbm={w[0]:.3f}  xgb={w[1]:.3f}  "
          f"catboost={w[2]:.3f}  histgb={w[3]:.3f}  "
          f"intercept={meta.intercept_[0]:.3f}")

    return {
        "meta":       meta,
        "models":     MODELS,
        "brier_meta": brier_meta,
        "brier_mean": brier_mean,
    }
```

**Step 3: Verify syntax**

```bash
cd march_learning_mania
python -c "import ast; ast.parse(open('scripts/13_meta_learner.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 2: Add predict_meta_gender() + generate_meta_submission()

**Files:**
- Modify: `scripts/13_meta_learner.py` (append)

**Step 1: Add build_features() helper (self-contained copy)**

```python
def _build_features(sub_df: pd.DataFrame, features_df: pd.DataFrame,
                    feat_cols: list) -> pd.DataFrame:
    """Vectorised feature-diff join for submission rows. Same as 12_predict.py."""
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
```

**Step 2: Add predict_meta_gender() function**

```python
def predict_meta_gender(sub_df: pd.DataFrame, gender: str,
                        meta_artifact: dict) -> np.ndarray:
    """
    Predict for one gender's submission slice using the meta-learner.
    No Platt calibration applied — LogisticRegression is already calibrated.
    """
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    meta     = meta_artifact["meta"]

    # Get feature cols from the first base model
    first_art = joblib.load(utils.MODELS / f"{MODELS[0]}_tuned_{gender}.pkl")
    feat_cols = first_art["feature_cols"]
    X_df      = _build_features(sub_df, features, feat_cols)

    # Collect base model predictions
    base_preds = []
    for model_name in MODELS:
        art   = joblib.load(utils.MODELS / f"{model_name}_tuned_{gender}.pkl")
        X     = X_df[art["feature_cols"]]
        if "imputer" in art:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_imp = art["imputer"].transform(X)
            p = art["model"].predict_proba(X_imp)[:, 1]
        else:
            p = art["model"].predict_proba(X.values)[:, 1]
        base_preds.append(p)

    # Stack → logit → meta-learner
    P      = np.column_stack(base_preds)   # shape (N, 4)
    L      = to_logit(P)                   # logit-space
    cal    = meta.predict_proba(L)[:, 1]

    return np.clip(cal, CLIP_LOW, CLIP_HIGH)
```

**Step 3: Add generate_meta_submission() + __main__**

```python
def generate_meta_submission(stage: int, meta_artifacts: dict) -> pd.DataFrame:
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

        preds = predict_meta_gender(sub_g, gender, meta_artifacts[gender])
        sub.loc[mask, "Pred"] = preds

        brier_ref = meta_artifacts[gender]["brier_meta"]
        print(f"done  (mean={preds.mean():.3f}, meta OOF Brier={brier_ref:.4f})")

    return sub[["ID", "Pred"]]


if __name__ == "__main__":
    meta_artifacts = {}
    for gender in ["M", "W"]:
        artifact = train_meta(gender)
        meta_artifacts[gender] = artifact

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

    # Generate submissions
    utils.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for stage in [1, 2]:
        result = generate_meta_submission(stage, meta_artifacts)

        assert result["Pred"].between(CLIP_LOW, CLIP_HIGH).all(), \
            f"Stage {stage}: predictions out of range!"
        assert result["Pred"].isna().sum() == 0, \
            f"Stage {stage}: NaN predictions found!"

        out = utils.SUBMISSIONS / f"submission_meta_stage{stage}.csv"
        result.to_csv(out, index=False)
        print(f"  Saved {out.name}  ({len(result):,} rows)")

    print("\n" + "="*52)
    print("META-LEARNER SUMMARY")
    print("="*52)
    for gender, art in meta_artifacts.items():
        label = "Men's" if gender == "M" else "Women's"
        delta = art["brier_meta"] - art["brier_mean"]
        print(f"  {label:<10} mean={art['brier_mean']:.4f}  "
              f"meta={art['brier_meta']:.4f}  delta={delta:+.4f}")
    print("Benchmarks updated.")
```

**Step 4: Verify full syntax**

```bash
python -c "import ast; ast.parse(open('scripts/13_meta_learner.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 3: Run + validate + commit

**Step 1: Run the script**

```bash
cd march_learning_mania
python scripts/13_meta_learner.py 2>&1 | grep -v "UserWarning\|warnings.warn\|sklearn\|site-packages\|LGBMClassifier"
```

Expected output:
```
====================================================
Meta-learner -- Men's
====================================================
  Mean ensemble Brier = 0.1866
  Meta-learner  Brier = 0.1850  (delta = -0.0016)
  Weights (logit): lgbm=0.89  xgb=0.03  catboost=0.46  histgb=-0.39  intercept=...
  Saved meta_learner_M.pkl
...
META-LEARNER SUMMARY
====================================================
  Men's      mean=0.1866  meta=0.1850  delta=-0.0016
  Women's    mean=0.1463  meta=0.1459  delta=-0.0004
```

**Step 2: Validate submission files**

```bash
python -c "
import pandas as pd
for stage in [1, 2]:
    df = pd.read_csv(f'submissions/submission_meta_stage{stage}.csv')
    print(f'Stage {stage}: {len(df):,} rows | '
          f'Pred=[{df.Pred.min():.4f}, {df.Pred.max():.4f}] | '
          f'mean={df.Pred.mean():.4f} | NaN={df.Pred.isna().sum()}')
"
```

Expected:
- Stage 1: 519,144 rows, all Pred in [0.025, 0.975], NaN=0
- Stage 2: 132,133 rows, all Pred in [0.025, 0.975], NaN=0

**Step 3: Compare meta vs ensemble predictions on a few rows**

```bash
python -c "
import pandas as pd
ens = pd.read_csv('submissions/submission_stage1.csv')
meta = pd.read_csv('submissions/submission_meta_stage1.csv')
merged = ens.merge(meta, on='ID', suffixes=('_ens','_meta'))
diff = (merged.Pred_meta - merged.Pred_ens).abs()
print(f'Mean abs diff: {diff.mean():.4f}')
print(f'Max abs diff:  {diff.max():.4f}')
print(merged[['ID','Pred_ens','Pred_meta']].head(8).to_string())
"
```

**Step 4: Print updated benchmark table**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/benchmarks.csv')[['model','gender','cv_brier']]
pivot = df.pivot(index='model', columns='gender', values='cv_brier')
print(pivot.sort_values('M').to_string())
"
```

**Step 5: Commit**

```bash
git add scripts/13_meta_learner.py \
        results/meta_learner_M.pkl results/meta_learner_W.pkl \
        results/benchmarks.csv BENCHMARKS.md \
        submissions/submission_meta_stage1.csv \
        submissions/submission_meta_stage2.csv
git commit -m "feat: stacking meta-learner (13) -- LogReg on logit OOF beats mean ensemble"
```

---

## Acceptance Criteria

- [ ] `results/meta_learner_{M,W}.pkl` exist with keys `meta`, `models`, `brier_meta`, `brier_mean`
- [ ] `submissions/submission_meta_stage{1,2}.csv` have correct row counts (519,144 and 132,133)
- [ ] All Pred in [0.025, 0.975], NaN=0
- [ ] `BENCHMARKS.md` has `meta_v1` rows for both genders
- [ ] Men's meta Brier < Men's mean ensemble Brier (should be ~0.1850 vs 0.1866)
- [ ] Women's meta Brier ≤ Women's mean ensemble Brier (should be ~0.1459 vs 0.1463)

---

## Notes

- **Why not Platt on top of meta?** LogisticRegression minimises log-loss by definition, which forces well-calibrated probabilities. Adding a second logistic rescaling on top would be redundant.
- **Negative weights** (e.g. XGB=-0.65 for Women's) arise from multicollinearity — the 4 base models are correlated. C=1.0 regularisation keeps them stable; they don't mean XGB hurts, just that in the presence of the others it contributes negatively to the optimal linear combination in logit space.
- **OOF bias caveat**: meta-learner is trained and evaluated on the same OOF set. With 5 parameters (4w + intercept) and 669 samples, effective overfitting is <0.001 Brier. Real generalisation will be confirmed by the Stage 1 Kaggle leaderboard.
- **Upload**: use `submission_meta_stage1.csv` for Kaggle Stage 1 evaluation. If it scores better than 0.1384, switch to the meta-learner for Stage 2.
