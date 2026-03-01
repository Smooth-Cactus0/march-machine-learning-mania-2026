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
python scripts/09_tune_optuna.py --model lgbm     --gender M --trials 50
python scripts/09_tune_optuna.py --model lgbm     --gender W --trials 50
python scripts/09_tune_optuna.py --model xgb      --gender M --trials 50
python scripts/09_tune_optuna.py --model xgb      --gender W --trials 50
python scripts/09_tune_optuna.py --model catboost --gender M --trials 50
python scripts/09_tune_optuna.py --model catboost --gender W --trials 50
python scripts/09_tune_optuna.py --model histgb   --gender M --trials 50
python scripts/09_tune_optuna.py --model histgb   --gender W --trials 50

# Step 4: Ensemble + calibration
python scripts/10_ensemble.py
python scripts/11_calibrate.py

# Step 5: Generate submissions
python scripts/12_predict.py              # ensemble + Platt (LB 0.1384)
python scripts/13_meta_learner.py         # meta-learner stacking (LB 0.125) ← best
```

Submission files are written to `submissions/`.

## Notes

- `features/` and `models/` are gitignored (too large). Re-generate locally using the steps above.
- `results/benchmarks.csv` and `BENCHMARKS.md` are committed — these track CV scores across all experiments.
- Scripts `01_eda.py` and `07_analysis.py` are optional (generate figures only).
- Script `08_train_histgb.py` and `04–06_train_*.py` are pre-Optuna training scripts preserved for reference; `09_tune_optuna.py` supersedes them.
