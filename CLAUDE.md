# March Machine Learning Mania 2026

## Competition Overview
- **Goal**: Predict win probability for every possible matchup in the 2026 NCAA Men's and Women's Basketball Tournaments
- **Evaluation Metric**: Brier Score (MSE of probability predictions — lower is better)
- **Submission Format**: CSV with columns `ID` (format: `YYYY_TeamID1_TeamID2`, lower ID first) and `Pred` (probability 0–1)
- **Stage 1**: ~519k predictions using 2022 as test year (calibration/validation)
- **Stage 2**: ~132k predictions for 2026 tournament (actual submission)
- **Target**: Top-10 finish on the leaderboard

## Data Overview
All raw data is in `march-machine-learning-mania-2026/`. Key files:
- `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` — game-by-game results 1985–2025
- `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` — box score stats 2003–2025
- `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` — tournament results
- `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` — tournament seeds
- `MMasseyOrdinals.csv` — 196 ranking systems (POM, SAG, MOR, etc.), 2003–2026, ~5.7M rows
- `SampleSubmissionStage1.csv` / `SampleSubmissionStage2.csv` — submission templates
- Men's teams: IDs 1xxx, Women's teams: IDs 3xxx

## Architecture
```
Raw CSVs → Feature Engineering → Matchup DataFrame → Model Training → Calibration → Submission
```
- Features engineered per (Season, TeamID): efficiency metrics, Massey composite, seed info
- Matchup rows: diff(team1_features - team2_features)
- CV strategy: Leave-One-Season-Out (last 10 tourney seasons)
- Models: LightGBM, XGBoost, CatBoost (tuned with Optuna) → ensemble

## Script Inventory
| Script | Purpose |
|--------|---------|
| `scripts/utils.py` | Shared data loaders, Brier score, CV helpers, benchmark logger |
| `scripts/01_eda.py` | EDA → saves figures to `figures/` |
| `scripts/02_feature_engineering.py` | Build team-season features → `features/` parquet |
| `scripts/03_train_baseline.py` | Seed-diff logistic regression baseline |
| `scripts/04_train_lgbm.py` | LightGBM with LOSO-CV |
| `scripts/05_train_xgb.py` | XGBoost with LOSO-CV |
| `scripts/06_train_catboost.py` | CatBoost with LOSO-CV |
| `scripts/07_tune_optuna.py` | Optuna hyperparameter search for all models |
| `scripts/08_ensemble.py` | Rank/probability averaging + stacking |
| `scripts/09_calibrate.py` | Brier-optimized calibration |
| `scripts/10_predict.py` | Generate Stage 1 + Stage 2 submissions |

## Commands
```bash
# Python interpreter
PYTHON=C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe

# Run scripts from project root
$PYTHON scripts/01_eda.py
$PYTHON scripts/02_feature_engineering.py
$PYTHON scripts/03_train_baseline.py
```

## Directory Structure
```
march_learning_mania/
├── CLAUDE.md                    ← this file
├── README.md
├── BENCHMARKS.md                ← auto-updated by training scripts
├── requirements.txt
├── .gitignore
├── docs/plans/                  ← implementation plans
├── scripts/                     ← all numbered scripts + utils.py
├── figures/                     ← EDA + analysis plots (committed)
├── features/                    ← cached parquet files (gitignored)
├── models/                      ← saved model artifacts (gitignored)
├── results/
│   └── benchmarks.csv           ← CV scores per model (committed)
└── submissions/                 ← Stage 1 + Stage 2 CSV files
```

## Key Notes
- Run all scripts from project root (not from `scripts/`)
- Both Men's (M) and Women's (W) tournaments modeled separately
- `figures/`, `results/`, `submissions/` committed to git
- `features/`, `models/`, raw data gitignored (too large)
- `BENCHMARKS.md` auto-regenerated from `results/benchmarks.csv` by each training script
