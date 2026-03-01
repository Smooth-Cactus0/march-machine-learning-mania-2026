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
12_predict.py
   │  mean ensemble + Platt calibration
   ▼
submissions/submission_stage{1,2}.csv           (LB 0.1384)

13_meta_learner.py
   │  LogisticRegression on logit(OOF predictions)
   │  C=1.0 (L2 regularisation prevents multicollinearity)
   ▼
submissions/submission_meta_stage{1,2}.csv      (LB 0.125)
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
python scripts/12_predict.py              # ensemble + Platt submissions (LB 0.1384)
python scripts/13_meta_learner.py         # meta-learner submissions  (LB 0.125) ← best
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
