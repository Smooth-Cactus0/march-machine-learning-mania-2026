# Pipeline Reference

Each numbered script reads specific inputs and writes specific outputs.
Run from the project root in ascending order.

---

| Script | Reads | Writes | Purpose |
|--------|-------|--------|---------|
| `01_eda.py` | Raw CSVs | `figures/` | Exploratory analysis — histograms, distributions, seed statistics |
| `02_feature_engineering.py` | Raw CSVs | `features/team_features_{M,W}.parquet` | Build per-(Season, TeamID) features: efficiency, Massey, seeds, location splits, conference, coach |
| `03_train_baseline.py` | `features/` | `results/benchmarks.csv` | Seed-diff logistic regression baseline (Brier ≈ 0.196 Men's) |
| `04_train_lgbm.py` | `features/` | `results/benchmarks.csv` | LightGBM with curated features and LOSO-CV (pre-Optuna reference) |
| `05_train_xgb.py` | `features/` | `results/benchmarks.csv` | XGBoost with curated features and LOSO-CV (pre-Optuna reference) |
| `06_train_catboost.py` | `features/` | `results/benchmarks.csv` | CatBoost with curated features and LOSO-CV (pre-Optuna reference) |
| `07_analysis.py` | `features/`, model artifacts | `figures/` | Feature importance, correlation heatmap, prediction diagnostics |
| `08_train_histgb.py` | `features/` | `results/benchmarks.csv` | HistGradientBoosting with monotonic constraints (pre-Optuna reference) |
| `09_tune_optuna.py` | `features/` | `models/{model}_tuned_{M,W}.pkl`, `results/best_params_*.json` | Optuna TPE hyperparameter search (50 trials per model×gender); retrains final model on full data |
| `10_ensemble.py` | `models/`, `features/` | `results/oof_preds_{M,W}.npz`, `results/ensemble_config.json` | Collects OOF predictions from all 4 models; selects mean vs rank-mean ensemble strategy |
| `11_calibrate.py` | `results/oof_preds_{M,W}.npz` | `results/calibrator_{M,W}.pkl` | Platt scaling (LogReg on logit(ensemble_pred) → labels) |
| `12_predict.py` | `models/`, `results/calibrator_*.pkl`, `results/ensemble_config.json`, `features/` | `submissions/submission_stage{1,2}.csv` | Mean ensemble + Platt calibration predictions for Stage 1 and Stage 2 (LB 0.1384) |
| `13_meta_learner.py` | `results/oof_preds_{M,W}.npz`, `models/` | `results/meta_learner_{M,W}.pkl`, `submissions/submission_meta_stage{1,2}.csv` | LogReg stacking meta-learner on logit OOF predictions; best submission (LB 0.125) |

---

## Key Shared Module: `scripts/utils.py`

| Symbol | Purpose |
|--------|---------|
| `utils.DATA` | Path to `march-machine-learning-mania-2026/` |
| `utils.FEATURES` | Path to `features/` |
| `utils.MODELS` | Path to `models/` |
| `utils.RESULTS` | Path to `results/` |
| `utils.CURATED_FEATURES` | Set of 18 feature names used by all models |
| `utils.build_monotone_vec(feat_cols)` | Returns monotonic constraint list for HistGB |
| `utils.make_matchup_df_nan_tolerant(tourney, features)` | Build matchup diff DataFrame, NaN-tolerant |
| `utils.brier_score(y_true, y_pred)` | MSE of probability predictions |
| `utils.log_benchmark(model, gender, brier, notes)` | Append to benchmarks.csv, regenerate BENCHMARKS.md |
| `utils.get_cv_seasons(tourney_df)` | Last 10 tournament seasons excluding 2020 (COVID) |

---

## CV Strategy

All models use **Leave-One-Season-Out (LOSO) cross-validation**:
- Fold: one full tournament season withheld for validation (~67 games)
- Train: all other available seasons
- Seasons used: last 10 tournament seasons, **excluding 2020** (tournament cancelled due to COVID)
- This matches the test distribution (predict a single future tournament year)

## Artifact Format

| Artifact | Keys |
|----------|------|
| `models/lgbm_tuned_{M,W}.pkl` | `model`, `imputer`, `feature_cols` |
| `models/xgb_tuned_{M,W}.pkl` | `model`, `imputer`, `feature_cols` |
| `models/catboost_tuned_{M,W}.pkl` | `model`, `feature_cols` |
| `models/histgb_tuned_{M,W}.pkl` | `model`, `feature_cols` |
| `results/calibrator_{M,W}.pkl` | `platt` (fitted LogisticRegression) |
| `results/oof_preds_{M,W}.npz` | `y_true`, `y_lgbm`, `y_xgb`, `y_catboost`, `y_histgb`, `y_ensemble` |
| `results/meta_learner_{M,W}.pkl` | `meta`, `models`, `brier_meta`, `brier_mean` |
