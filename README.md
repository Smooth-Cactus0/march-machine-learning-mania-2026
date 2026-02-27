# March Machine Learning Mania 2026

> Kaggle competition · NCAA Basketball Tournament probability forecasting · Target: Top 10

## Competition

[Kaggle competition page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

Predict the win probability for every possible game matchup in the 2026 NCAA Men's and Women's Basketball Tournaments. Evaluated on **Brier Score** (lower = better).

## Approach

1. **Feature Engineering** — per-team season efficiency metrics (eFG%, offensive/defensive efficiency, margin), composite Massey rating from 196 ranking systems, seed features
2. **Models** — LightGBM, XGBoost, CatBoost independently trained on matchup-level feature diffs
3. **Ensemble** — rank/probability averaging + stacking
4. **Calibration** — Brier-score-optimized isotonic regression

## Benchmarks

| Model | CV Brier (Men's) | CV Brier (Women's) | Notes |
|-------|-------------------|---------------------|-------|
| Baseline (seed-diff) | — | — | floor |
| LightGBM | — | — | |
| XGBoost | — | — | |
| CatBoost | — | — | |
| **Ensemble** | **—** | **—** | final |

*See [BENCHMARKS.md](BENCHMARKS.md) for full history.*

## Structure

```
scripts/         numbered pipeline scripts
figures/         EDA and analysis plots
results/         CV benchmark CSV
submissions/     Stage 1 + Stage 2 predictions
```

## Usage

```bash
python scripts/01_eda.py                 # EDA + figures
python scripts/02_feature_engineering.py # build features
python scripts/03_train_baseline.py      # seed-diff baseline
# ... continues through scripts 04–10 (models, tuning, ensemble, calibration, predict)
# See CLAUDE.md for full pipeline
```

## Author

Alexy Louis
