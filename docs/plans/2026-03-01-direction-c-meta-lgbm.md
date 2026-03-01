# Direction C — LightGBM Meta-Learner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write `scripts/14_meta_lgbm.py` — a LightGBM stacking meta-learner that takes 6 inputs (4 logit OOF predictions + SeedNum_diff + massey_pc1_diff) and learns a non-linear combination. Falls back to LogisticRegression if LightGBM doesn't beat it on both genders.

**Architecture:** Parallel to `13_meta_learner.py`. Loads same `oof_preds_{M,W}.npz` + loads `SeedNum_diff` / `massey_pc1_diff` from feature parquets. Trains LightGBM meta with Optuna (30 trials, 5-fold stratified-by-decade CV). Compares CV Brier vs LogReg meta. Saves the winner as `meta_learner_v2_{M,W}.pkl` and generates new submission files.

**Tech Stack:** `lightgbm`, `optuna`, `sklearn`, `numpy`, `joblib`, project `utils.py`.

**Note:** Run this plan AFTER Direction A features are complete (parquets must have `massey_pc1`).

---

### Task 1: Write 14_meta_lgbm.py — imports + helpers

**Files:**
- Create: `scripts/14_meta_lgbm.py`

**Step 1: Create file with header and imports**

```python
"""
14_meta_lgbm.py -- March Machine Learning Mania 2026

LightGBM stacking meta-learner with Optuna tuning.
Inputs (per gender, N ≈ 670 rows):
  - 4 logit-transformed OOF predictions (lgbm, xgb, catboost, histgb)
  - SeedNum_diff     (strongest single raw feature)
  - massey_pc1_diff  (PCA consensus ranking)

Falls back to LogisticRegression if LightGBM CV Brier is not strictly better
on both genders.

Inputs:  results/oof_preds_{M,W}.npz       (from 10_ensemble.py)
         features/team_features_{M,W}.parquet
         models/{model}_tuned_{M,W}.pkl

Outputs: results/meta_lgbm_{M,W}.pkl       (best meta model per gender)
         results/meta_comparison.json       (CV Brier: lgbm vs logreg)
         submissions/submission_meta_v2_stage{1,2}.csv
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

MODELS   = ["lgbm", "xgb", "catboost", "histgb"]
EPS      = 1e-7
CLIP_LOW  = 0.025
CLIP_HIGH = 0.975
N_TRIALS  = 30
N_FOLDS   = 5


def to_logit(p: np.ndarray) -> np.ndarray:
    """Transform probability array to log-odds. Clips to avoid ±inf."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 2: Add load_meta_features() — load OOF + anchor features

**Files:**
- Modify: `scripts/14_meta_lgbm.py` (append)

**Step 1: Add the feature loader**

```python
# ── Feature loading ────────────────────────────────────────────────────────────

def load_meta_features(gender: str) -> tuple:
    """
    Build the meta-learner feature matrix for one gender.

    Returns:
      X  -- np.ndarray of shape (N, 6):
              [logit_lgbm, logit_xgb, logit_catboost, logit_histgb,
               SeedNum_diff, massey_pc1_diff]
      y  -- np.ndarray of shape (N,): true labels (0/1)
      seasons -- np.ndarray of shape (N,): season per OOF row (for CV grouping)
    """
    # Load OOF predictions from 10_ensemble.py
    npz    = np.load(utils.RESULTS / f"oof_preds_{gender}.npz")
    y      = npz["y_true"]

    # 4 base logit columns
    logit_cols = np.column_stack([to_logit(npz[f"y_{m}"]) for m in MODELS])

    # ── Anchor features: SeedNum_diff + massey_pc1_diff ───────────────────────
    # These are loaded from the feature parquet + reconstructed as diffs.
    # We need to match OOF rows to their (Season, Team1ID, Team2ID) identifiers.
    # The OOF npz doesn't store these directly; we reconstruct from tourney data.

    tourney  = utils.load_tourney(gender)
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")

    # Identify OOF seasons (same logic as 10_ensemble.py)
    cv_seasons = utils.get_cv_seasons(tourney)

    all_rows = []
    for season in cv_seasons:
        t_season = tourney[tourney["Season"] == season].copy()
        t_season["Team1ID"] = t_season[["WTeamID", "LTeamID"]].min(axis=1)
        t_season["Team2ID"] = t_season[["WTeamID", "LTeamID"]].max(axis=1)
        all_rows.append(t_season[["Season", "Team1ID", "Team2ID"]])

    matchup_ids = pd.concat(all_rows, ignore_index=True)

    # Build anchor feature diffs
    feat_indexed = features.set_index(["Season", "TeamID"])
    anchor_cols  = []
    for col_name in ["SeedNum", "massey_pc1"]:
        if col_name not in features.columns:
            # massey_pc1 only available after Direction A — fallback to zeros
            anchor_cols.append(np.zeros(len(matchup_ids)))
            continue
        t1 = (matchup_ids[["Season", "Team1ID"]]
              .rename(columns={"Team1ID": "TeamID"})
              .join(feat_indexed[[col_name]], on=["Season", "TeamID"])[col_name]
              .values.astype(float))
        t2 = (matchup_ids[["Season", "Team2ID"]]
              .rename(columns={"Team2ID": "TeamID"})
              .join(feat_indexed[[col_name]], on=["Season", "TeamID"])[col_name]
              .values.astype(float))
        diff = t1 - t2
        # Fill NaN with 0 (median imputation equivalent for symmetric diffs)
        diff = np.where(np.isnan(diff), 0.0, diff)
        anchor_cols.append(diff)

    anchor = np.column_stack(anchor_cols)   # (N, 2)
    X      = np.column_stack([logit_cols, anchor])  # (N, 6)

    # Season labels for CV grouping
    seasons = matchup_ids["Season"].values

    print(f"  Meta features ({gender}): X={X.shape}, y={y.shape}, "
          f"NaN in X: {np.isnan(X).sum()}")
    return X, y, seasons
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```

---

### Task 3: Add LightGBM CV + Optuna tuning

**Files:**
- Modify: `scripts/14_meta_lgbm.py` (append)

**Step 1: Add LGBM CV runner and Optuna objective**

```python
# ── LightGBM CV + Optuna ───────────────────────────────────────────────────────

def _lgbm_cv_brier(params: dict, X: np.ndarray, y: np.ndarray,
                   seasons: np.ndarray) -> float:
    """
    5-fold stratified-by-decade CV (not LOSO — N≈670 is too small per fold for trees).
    Folds are stratified by outcome to keep class balance.
    """
    # Use decade as a proxy group to avoid future-leakage in the fold split
    # (ensure no fold trains on season S and validates on season S-1)
    decade_group = (seasons // 10).astype(int)  # e.g. 2015 → 201

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds, oof_labels = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx],   y[val_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
            n_jobs=-1,
            **params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                 lgb.log_evaluation(period=-1)])

        oof_preds.extend(model.predict_proba(X_va)[:, 1])
        oof_labels.extend(y_va)

    return utils.brier_score(np.array(oof_labels), np.array(oof_preds))


def make_lgbm_meta_objective(X: np.ndarray, y: np.ndarray,
                              seasons: np.ndarray):
    """Factory returning Optuna objective for meta-learner tuning."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "n_estimators":      trial.suggest_int("n_estimators", 30, 200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        return _lgbm_cv_brier(params, X, y, seasons)
    return objective


def tune_lgbm_meta(X: np.ndarray, y: np.ndarray,
                   seasons: np.ndarray, gender: str) -> dict:
    """Run Optuna for N_TRIALS and return best params + CV Brier."""
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n  Tuning LightGBM meta ({label}) — {N_TRIALS} trials...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        make_lgbm_meta_objective(X, y, seasons),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    print(f"  Best LGBM meta Brier: {study.best_value:.4f}")
    return {"params": study.best_params, "cv_brier": study.best_value}
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```

---

### Task 4: Add LogReg baseline + fallback logic

**Files:**
- Modify: `scripts/14_meta_lgbm.py` (append)

**Step 1: Add LogReg CV + comparison function**

```python
# ── LogReg baseline + winner selection ────────────────────────────────────────

def logreg_cv_brier(X: np.ndarray, y: np.ndarray,
                    seasons: np.ndarray) -> float:
    """5-fold CV Brier for LogisticRegression on same features."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds, oof_labels = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(X[train_idx], y[train_idx])
        oof_preds.extend(model.predict_proba(X[val_idx])[:, 1])
        oof_labels.extend(y[val_idx])
    return utils.brier_score(np.array(oof_labels), np.array(oof_preds))


def train_final_meta(X: np.ndarray, y: np.ndarray,
                     best_params: dict, use_lgbm: bool):
    """
    Retrain the winning model on ALL N samples (no holdout).
    Returns fitted model.
    """
    if use_lgbm:
        model = lgb.LGBMClassifier(
            objective="binary", verbose=-1, n_jobs=-1, **best_params
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
    else:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(X, y)
    return model


def select_and_train(gender: str, lgbm_result: dict) -> dict:
    """
    Compare LGBM vs LogReg CV Brier. Select winner.
    Returns artifact dict.
    """
    X, y, seasons = load_meta_features(gender)
    label = "Men's" if gender == "M" else "Women's"

    brier_logreg = logreg_cv_brier(X, y, seasons)
    brier_lgbm   = lgbm_result["cv_brier"]

    print(f"\n  {label} comparison:")
    print(f"    LogReg CV Brier:  {brier_logreg:.4f}")
    print(f"    LightGBM CV Brier: {brier_lgbm:.4f}")

    use_lgbm = brier_lgbm < brier_logreg
    winner   = "lgbm" if use_lgbm else "logreg"
    brier    = brier_lgbm if use_lgbm else brier_logreg
    print(f"    Winner: {winner}  (delta={brier_lgbm - brier_logreg:+.4f})")

    final_model = train_final_meta(X, y, lgbm_result["params"], use_lgbm)

    return {
        "meta":           final_model,
        "winner":         winner,
        "brier_lgbm":     brier_lgbm,
        "brier_logreg":   brier_logreg,
        "brier_selected": brier,
        "use_lgbm":       use_lgbm,
        "lgbm_params":    lgbm_result["params"],
    }
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```

---

### Task 5: Add predict + submission generation

**Files:**
- Modify: `scripts/14_meta_lgbm.py` (append)

**Step 1: Add predict_meta_v2_gender() and generate_meta_v2_submission()**

```python
# ── Inference ──────────────────────────────────────────────────────────────────

def _build_anchor_diff(sub_df: pd.DataFrame, features_df: pd.DataFrame,
                       col_name: str) -> np.ndarray:
    """Build SeedNum_diff or massey_pc1_diff for submission rows."""
    if col_name not in features_df.columns:
        return np.zeros(len(sub_df))
    feat = features_df.set_index(["Season", "TeamID"])
    t1   = (sub_df[["Season", "Team1ID"]]
            .rename(columns={"Team1ID": "TeamID"})
            .join(feat[[col_name]], on=["Season", "TeamID"])[col_name]
            .values.astype(float))
    t2   = (sub_df[["Season", "Team2ID"]]
            .rename(columns={"Team2ID": "TeamID"})
            .join(feat[[col_name]], on=["Season", "TeamID"])[col_name]
            .values.astype(float))
    diff = t1 - t2
    return np.where(np.isnan(diff), 0.0, diff)


def predict_meta_v2_gender(sub_df: pd.DataFrame, gender: str,
                            artifact: dict) -> np.ndarray:
    """Predict for one gender's submission slice with the v2 meta-learner."""
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    meta     = artifact["meta"]

    # Collect base model predictions
    base_preds = []
    first_art  = joblib.load(utils.MODELS / f"{MODELS[0]}_tuned_{gender}.pkl")
    feat_cols  = first_art["feature_cols"]

    # Build feature diffs (same as 13_meta_learner.py)
    feat_indexed = features.set_index(["Season", "TeamID"])
    raw_cols     = [c.replace("_diff", "") for c in feat_cols]
    t1 = (sub_df[["Season", "Team1ID"]]
          .rename(columns={"Team1ID": "TeamID"})
          .join(feat_indexed, on=["Season", "TeamID"])
          .drop(columns="TeamID"))
    t2 = (sub_df[["Season", "Team2ID"]]
          .rename(columns={"Team2ID": "TeamID"})
          .join(feat_indexed, on=["Season", "TeamID"])
          .drop(columns="TeamID"))
    X_df = pd.DataFrame(
        {fc: t1[rc].values - t2[rc].values for fc, rc in zip(feat_cols, raw_cols)},
        index=sub_df.index,
    )

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

    # Build meta feature matrix: [logit_lgbm, logit_xgb, logit_catboost, logit_histgb,
    #                              SeedNum_diff, massey_pc1_diff]
    logit_cols = np.column_stack([to_logit(p) for p in base_preds])
    seed_diff  = _build_anchor_diff(sub_df, features, "SeedNum")
    pca_diff   = _build_anchor_diff(sub_df, features, "massey_pc1")
    X_meta     = np.column_stack([logit_cols, seed_diff, pca_diff])

    preds = meta.predict_proba(X_meta)[:, 1]
    return np.clip(preds, CLIP_LOW, CLIP_HIGH)


def generate_meta_v2_submission(stage: int, artifacts: dict) -> pd.DataFrame:
    """Parse sample submission, predict with v2 meta-learner, return DataFrame."""
    print(f"\nGenerating v2 meta Stage {stage} submission...")
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
        preds = predict_meta_v2_gender(sub_g, gender, artifacts[gender])
        sub.loc[mask, "Pred"] = preds
        print(f"done  (mean={preds.mean():.3f})")

    return sub[["ID", "Pred"]]
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```

---

### Task 6: Add __main__ block

**Files:**
- Modify: `scripts/14_meta_lgbm.py` (append)

**Step 1: Add entry point**

```python
# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("LIGHTGBM META-LEARNER v2")
    print("=" * 52)

    artifacts   = {}
    comparison  = {}

    for gender in ["M", "W"]:
        label = "Men's" if gender == "M" else "Women's"
        print(f"\n{'='*52}")
        print(f"Gender: {label}")
        print(f"{'='*52}")

        X, y, seasons = load_meta_features(gender)

        # Tune LGBM meta
        lgbm_result = tune_lgbm_meta(X, y, seasons, gender)

        # Compare vs LogReg, select winner, train final model
        artifact = select_and_train(gender, lgbm_result)
        artifacts[gender] = artifact
        comparison[gender] = {
            "winner":         artifact["winner"],
            "brier_lgbm":     artifact["brier_lgbm"],
            "brier_logreg":   artifact["brier_logreg"],
            "brier_selected": artifact["brier_selected"],
        }

        # Save artifact
        utils.RESULTS.mkdir(parents=True, exist_ok=True)
        path = utils.RESULTS / f"meta_lgbm_{gender}.pkl"
        joblib.dump(artifact, path)
        print(f"  Saved {path.name}")

        # Log benchmark
        utils.log_benchmark(
            "meta_v2", gender, artifact["brier_selected"],
            f"{artifact['winner']} meta-learner (lgbm={artifact['brier_lgbm']:.4f} "
            f"logreg={artifact['brier_logreg']:.4f})"
        )

    # Save comparison JSON
    comp_path = utils.RESULTS / "meta_comparison.json"
    json.dump(comparison, open(comp_path, "w"), indent=2)
    print(f"\nComparison saved to {comp_path.name}")

    # Generate submissions
    utils.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for stage in [1, 2]:
        result = generate_meta_v2_submission(stage, artifacts)

        assert result["Pred"].between(CLIP_LOW, CLIP_HIGH).all(), \
            f"Stage {stage}: predictions out of [{CLIP_LOW}, {CLIP_HIGH}] range!"
        assert result["Pred"].isna().sum() == 0, \
            f"Stage {stage}: NaN predictions found!"

        out = utils.SUBMISSIONS / f"submission_meta_v2_stage{stage}.csv"
        result.to_csv(out, index=False)
        print(f"  Saved {out.name}  ({len(result):,} rows)")

    # Summary
    print("\n" + "="*52)
    print("META v2 SUMMARY")
    print("="*52)
    for gender, comp in comparison.items():
        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label:<10}  winner={comp['winner']:<6}  "
              f"lgbm={comp['brier_lgbm']:.4f}  "
              f"logreg={comp['brier_logreg']:.4f}  "
              f"selected={comp['brier_selected']:.4f}")

    # Compare vs meta_v1
    print("\n  vs meta_v1 (LogReg on 4 logit OOF):")
    try:
        import pandas as pd
        bench = pd.read_csv(utils.RESULTS / "benchmarks.csv")
        for gender in ["M", "W"]:
            label = "Men's" if gender == "M" else "Women's"
            v1 = bench[(bench["model"] == "meta_v1") & (bench["gender"] == gender)]
            v2 = comparison[gender]["brier_selected"]
            if not v1.empty:
                delta = v2 - float(v1["cv_brier"].iloc[0])
                print(f"  {label:<10}  v1={float(v1['cv_brier'].iloc[0]):.4f}  "
                      f"v2={v2:.4f}  delta={delta:+.4f}")
    except Exception:
        pass
    print("Benchmarks updated.")
```

**Step 2: Verify full script syntax**

```bash
python -c "import ast; ast.parse(open('scripts/14_meta_lgbm.py').read()); print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/14_meta_lgbm.py
git commit -m "feat: LightGBM meta-learner v2 (14_meta_lgbm.py) with LogReg fallback"
```

---

### Task 7: Run + validate + benchmark

**Step 1: Run the script**

```bash
python scripts/14_meta_lgbm.py
```

Expected output pattern:
```
====================================================
LIGHTGBM META-LEARNER v2
====================================================

====================================================
Gender: Men's
====================================================
  Meta features (M): X=(670, 6), y=(670,), NaN in X: 0
  Tuning LightGBM meta (Men's) — 30 trials...
  Best LGBM meta Brier: 0.18XX

  Men's comparison:
    LogReg CV Brier:   0.185X
    LightGBM CV Brier: 0.184X
    Winner: lgbm  (delta=-0.00XX)
  ...

META v2 SUMMARY
  Men's      winner=lgbm   lgbm=0.184X  logreg=0.185X  selected=0.184X
  Women's    winner=lgbm   lgbm=0.145X  logreg=0.146X  selected=0.145X

  vs meta_v1 (LogReg on 4 logit OOF):
  Men's      v1=0.1850  v2=0.184X  delta=-0.000X
  Women's    v1=0.1459  v2=0.145X  delta=-0.000X
```

If LightGBM loses for a gender, the output will say `winner=logreg` — this is expected and correct.

**Step 2: Validate submission files**

```bash
python -c "
import pandas as pd
for stage in [1, 2]:
    df = pd.read_csv(f'submissions/submission_meta_v2_stage{stage}.csv')
    print(f'Stage {stage}: {len(df):,} rows | '
          f'Pred [{df.Pred.min():.4f}, {df.Pred.max():.4f}] | '
          f'mean={df.Pred.mean():.4f} | NaN={df.Pred.isna().sum()}')
    assert df.Pred.between(0.025, 0.975).all()
    assert df.Pred.isna().sum() == 0
print('Validation PASSED')
"
```

Expected: same row counts as existing meta submissions (519,144 / 132,133).

**Step 3: Check comparison JSON**

```bash
python -c "
import json
comp = json.load(open('results/meta_comparison.json'))
for g, c in comp.items():
    print(f'{g}: winner={c[\"winner\"]}  lgbm={c[\"brier_lgbm\"]:.4f}  logreg={c[\"brier_logreg\"]:.4f}')
"
```

**Step 4: Commit results**

```bash
git add results/benchmarks.csv BENCHMARKS.md results/meta_comparison.json \
    submissions/submission_meta_v2_stage1.csv submissions/submission_meta_v2_stage2.csv
git commit -m "feat: Direction C — LightGBM meta-learner results + new submissions"
```

**Step 5: Submit to Kaggle (manual)**

Upload `submissions/submission_meta_v2_stage1.csv` to Kaggle Stage 1.
Record the leaderboard score.

**Step 6: Update commit with LB score**

```bash
git commit --allow-empty -m "chore: Direction C Kaggle LB = XX.XXXX (meta_v2)"
```

---

## Acceptance Criteria

- [ ] `scripts/14_meta_lgbm.py` exists and passes syntax check
- [ ] Script runs end-to-end without errors
- [ ] LogReg CV Brier computed for both genders as comparison baseline
- [ ] LGBM meta-learner either beats LogReg or falls back to LogReg
- [ ] `results/meta_lgbm_{M,W}.pkl` saved
- [ ] `results/meta_comparison.json` saved with winner per gender
- [ ] Both v2 submission CSV files validated (row counts, range, no NaN)
- [ ] BENCHMARKS.md updated with `meta_v2` entries
- [ ] Kaggle LB score recorded in git commit message
