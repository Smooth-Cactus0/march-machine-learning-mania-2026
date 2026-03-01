# GitHub Repo Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `README.md`, `docs/SETUP.md`, and `docs/PIPELINE.md` to the existing project so the public repo tells a clear story of the ML pipeline and results progression.

**Architecture:** Three markdown files only — no code changes. README is the landing page with architecture diagram + LB table. SETUP.md is the reproduction guide. PIPELINE.md maps each numbered script to its role.

**Tech Stack:** Markdown, git.

---

### Task 1: Write README.md

**Files:**
- Create: `README.md` (project root, replacing the stub if one exists)

**Step 1: Write README.md**

Content sections in order:

```markdown
# March Machine Learning Mania 2026

Kaggle competition: predict win probability for every possible NCAA Men's and Women's
Basketball tournament matchup. Evaluation metric: Brier Score (lower is better).

**Best leaderboard score: 0.125** (Stage 1, LogReg meta-learner stacking)

---

## Results

| Submission | CV Brier (M) | CV Brier (W) | LB Stage 1 |
|------------|-------------|-------------|------------|
| Seed baseline (LogReg) | 0.1965 | 0.1497 | — |
| Optuna ensemble (mean) | 0.1866 | 0.1463 | 0.1384 |
| **Meta-learner (LogReg stacking)** | **0.1850** | **0.1459** | **0.125** |

CV = Leave-One-Season-Out cross-validation on last 10 tournament seasons.

---

## Pipeline Architecture

```
Raw CSVs
   │
   ▼
02_feature_engineering.py
   │  efficiency metrics (four factors, net eff)
   │  Massey ordinals composite (top-10 systems)
   │  tournament seeds
   │  neutral-site splits
   │  conference quality + SOS
   │  conference tournament performance
   │  coach continuity (Men's)
   ▼
features/team_features_{M,W}.parquet
   │
   ▼
09_tune_optuna.py  (50 Optuna trials × 4 models × 2 genders)
   │  LightGBM  ─┐
   │  XGBoost   ─┤─ LOSO-CV Brier per model
   │  CatBoost  ─┤
   │  HistGB    ─┘
   ▼
models/{model}_tuned_{M,W}.pkl
   │
   ▼
10_ensemble.py  →  results/oof_preds_{M,W}.npz
   │  mean ensemble (beats rank-mean post-Optuna)
   ▼
11_calibrate.py  →  results/calibrator_{M,W}.pkl
   │  Platt scaling (near-no-op; models already calibrated)
   ▼
13_meta_learner.py
   │  LogisticRegression on logit(OOF predictions)
   │  C=1.0 (L2 regularisation prevents multicollinearity)
   ▼
submissions/submission_meta_stage{1,2}.csv
```

---

## Kaggle Notebooks

Two pedagogical notebooks for the competition community:

- **[Notebook 1 — Feature Engineering](<kaggle_link_1>)**: data tour → efficiency metrics → Massey PCA → Elo ratings → momentum → correlation analysis
- **[Notebook 2 — Modelling Pipeline](<kaggle_link_2>)**: LOSO-CV → baseline → LightGBM+Optuna → ensemble → meta-learner → submission

---

## Reproducing Results

See [docs/SETUP.md](docs/SETUP.md) for installation and data download instructions.

Run scripts in order from the project root:

```bash
python scripts/02_feature_engineering.py   # build feature parquets
python scripts/03_train_baseline.py        # seed-diff baseline
python scripts/09_tune_optuna.py           # tune all 4 models (takes ~30 min)
python scripts/10_ensemble.py             # build OOF ensemble
python scripts/11_calibrate.py            # Platt scaling
python scripts/13_meta_learner.py         # meta-learner + generate submissions
```

---

## Directory Structure

```
march_learning_mania/
├── README.md                      ← this file
├── BENCHMARKS.md                  ← auto-updated CV scores
├── requirements.txt
├── docs/
│   ├── SETUP.md                   ← installation + data download
│   ├── PIPELINE.md                ← script-by-script description
│   └── plans/                     ← implementation design docs
├── scripts/
│   ├── utils.py                   ← shared loaders, CV helpers, benchmark logger
│   ├── 01_eda.py  …  13_meta_learner.py
├── figures/                       ← EDA + analysis plots (committed)
├── results/
│   └── benchmarks.csv             ← CV scores per model (committed)
├── submissions/                   ← Stage 1 + Stage 2 CSV files (committed)
├── features/                      ← cached parquet files (gitignored)
└── models/                        ← saved model artifacts (gitignored)
```

---

## Key Design Decisions

**Why Brier Score?** The competition uses MSE of probabilities (not log-loss). Clipping to [0.025, 0.975] prevents extreme overconfidence.

**Why LOSO-CV?** Each fold withholds one full tournament season (≈67 games). This matches the test distribution (a single tournament year) better than random k-fold.

**Why logit-space inputs to the meta-learner?** Base models output p ∈ (0,1). Logistic regression operates in log-odds space natively: sigmoid(w · logit(p) + b). Transforming inputs makes the combination linear in log-odds — the natural parameterisation for combining calibrated probability estimates.

**Why C=1.0 (not C=1e9) for the meta-learner?** With only 4 correlated inputs (all boosting models), unregularised weights diverge to compensate for multicollinearity (e.g. [5.1, −0.6, 2.1, −1.8]). C=1.0 L2 regularisation keeps weights stable and interpretable.
```

**Step 2: Verify the file exists and renders cleanly**

```bash
python -c "
with open('README.md') as f:
    lines = f.readlines()
print(f'README.md: {len(lines)} lines')
print('First heading:', [l.strip() for l in lines if l.startswith('#')][0])
"
```
Expected: `README.md: ~100 lines`, first heading matches.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add polished README with LB table, pipeline diagram, reproduction guide"
```

---

### Task 2: Write docs/SETUP.md

**Files:**
- Create: `docs/SETUP.md`

**Step 1: Write docs/SETUP.md**

```markdown
# Setup & Reproduction Guide

## Requirements

- Python 3.11 (Python 3.12+ has broken numpy compatibility with some packages)
- ~4 GB disk space for features and model artifacts

```bash
pip install -r requirements.txt
```

## Data Download

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API key in `~/.kaggle/`
3. Download competition data:

```bash
kaggle competitions download -c march-machine-learning-mania-2026
unzip march-machine-learning-mania-2026.zip -d march-machine-learning-mania-2026/
```

The `march-machine-learning-mania-2026/` directory must be at the project root (next to `scripts/`).

## Expected Directory Structure After Download

```
march_learning_mania/
├── march-machine-learning-mania-2026/
│   ├── MRegularSeasonCompactResults.csv
│   ├── MRegularSeasonDetailedResults.csv
│   ├── MMasseyOrdinals.csv
│   ├── MNCAATourneySeeds.csv
│   ├── SampleSubmissionStage1.csv
│   ├── SampleSubmissionStage2.csv
│   └── ... (30+ CSV files)
├── scripts/
├── docs/
└── requirements.txt
```

## Running the Pipeline

All scripts are run from the **project root** (not from `scripts/`):

```bash
cd march_learning_mania/

# Step 1: Build features (~2 min)
python scripts/02_feature_engineering.py

# Step 2: Train baseline
python scripts/03_train_baseline.py

# Step 3: Tune all 4 models with Optuna (~30 min total)
python scripts/09_tune_optuna.py --model lgbm   --gender M --trials 50
python scripts/09_tune_optuna.py --model lgbm   --gender W --trials 50
python scripts/09_tune_optuna.py --model xgb    --gender M --trials 50
python scripts/09_tune_optuna.py --model xgb    --gender W --trials 50
python scripts/09_tune_optuna.py --model catboost --gender M --trials 50
python scripts/09_tune_optuna.py --model catboost --gender W --trials 50
python scripts/09_tune_optuna.py --model histgb  --gender M --trials 50
python scripts/09_tune_optuna.py --model histgb  --gender W --trials 50

# Step 4: Ensemble + calibration
python scripts/10_ensemble.py
python scripts/11_calibrate.py

# Step 5: Meta-learner + submissions
python scripts/13_meta_learner.py
```

Submission files are written to `submissions/`.

## Notes

- `features/` and `models/` are gitignored (too large). Re-generate locally.
- `results/benchmarks.csv` and `BENCHMARKS.md` are committed — these track CV scores.
- Script `01_eda.py` and `07_analysis.py` are optional (generate figures only).
```

**Step 2: Commit**

```bash
git add docs/SETUP.md
git commit -m "docs: add SETUP.md with installation, data download, pipeline run instructions"
```

---

### Task 3: Write docs/PIPELINE.md

**Files:**
- Create: `docs/PIPELINE.md`

**Step 1: Write docs/PIPELINE.md**

```markdown
# Pipeline Reference

Each numbered script reads specific inputs and writes specific outputs.
Run from the project root in ascending order.

---

| Script | Reads | Writes | Purpose |
|--------|-------|--------|---------|
| `01_eda.py` | Raw CSVs | `figures/` | Exploratory analysis — histograms, distributions, seed statistics |
| `02_feature_engineering.py` | Raw CSVs | `features/team_features_{M,W}.parquet` | Build per-(Season, TeamID) features: efficiency, Massey, seeds, location splits, conference, coach |
| `03_train_baseline.py` | `features/` | `results/benchmarks.csv` | Seed-diff logistic regression baseline (Brier ≈ 0.196 Men's) |
| `04_train_lgbm.py` | `features/` | `results/benchmarks.csv` | LightGBM with curated features and LOSO-CV (pre-Optuna) |
| `05_train_xgb.py` | `features/` | `results/benchmarks.csv` | XGBoost with curated features and LOSO-CV (pre-Optuna) |
| `06_train_catboost.py` | `features/` | `results/benchmarks.csv` | CatBoost with curated features and LOSO-CV (pre-Optuna) |
| `07_analysis.py` | `features/`, model artifacts | `figures/` | Feature importance, correlation heatmap, prediction diagnostics |
| `08_train_histgb.py` | `features/` | `results/benchmarks.csv` | HistGradientBoosting with monotonic constraints (pre-Optuna) |
| `09_tune_optuna.py` | `features/` | `models/{model}_tuned_{M,W}.pkl`, `results/best_params_*.json` | Optuna TPE hyperparameter search (50 trials per model×gender); retrains final model on full data |
| `10_ensemble.py` | `models/`, `features/` | `results/oof_preds_{M,W}.npz`, `results/ensemble_config.json` | Collects OOF predictions from all 4 models; selects mean vs rank-mean ensemble strategy |
| `11_calibrate.py` | `results/oof_preds_{M,W}.npz` | `results/calibrator_{M,W}.pkl` | Platt scaling (LogReg on logit(ensemble_pred) → labels) |
| `12_predict.py` | `models/`, `results/calibrator_*.pkl`, `results/ensemble_config.json`, `features/` | `submissions/submission_stage{1,2}.csv` | Mean ensemble + Platt calibration predictions for Stage 1 and Stage 2 |
| `13_meta_learner.py` | `results/oof_preds_{M,W}.npz`, `models/` | `results/meta_learner_{M,W}.pkl`, `submissions/submission_meta_stage{1,2}.csv` | LogReg stacking meta-learner on logit OOF predictions; generates meta submissions |

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
```

**Step 2: Commit**

```bash
git add docs/PIPELINE.md
git commit -m "docs: add PIPELINE.md with script-by-script reference table"
```

---

## Acceptance Criteria

- [ ] `README.md` renders cleanly with LB table, ASCII pipeline, reproduction steps
- [ ] `docs/SETUP.md` has complete data download + run instructions
- [ ] `docs/PIPELINE.md` has one row per script with exact read/write paths
- [ ] All 3 files committed to master
