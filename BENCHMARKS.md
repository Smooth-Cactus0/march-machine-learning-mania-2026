# Benchmarks

CV Brier scores for all models (lower is better). Auto-updated by training scripts.

*Source data: `results/benchmarks.csv`*

| Model | Gender | CV Brier | Notes | Timestamp |
|-------|--------|----------|-------|-----------|
| baseline_logreg | M | 0.196454 | seed-diff only | 2026-02-27 14:09 |
| calibrated_v1 | M | 0.186554 | Platt-scaled ensemble (pre=0.1866 post=0.1866) | 2026-02-28 14:11 |
| catboost_v1 | M | 0.200576 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | M | 0.201369 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| catboost_v3 | M | 0.188090 | catboost Optuna n=50 best_brier=0.1881 | 2026-02-28 13:50 |
| ensemble_v1 | M | 0.186627 | 4-model mean ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned | 2026-02-28 14:10 |
| histgb_v1 | M | 0.202765 | HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints | 2026-02-28 13:42 |
| histgb_v3 | M | 0.189263 | histgb Optuna n=50 best_brier=0.1893 | 2026-02-28 13:51 |
| lgbm_v1 | M | 0.202406 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | M | 0.203346 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v3 | M | 0.185535 | lgbm Optuna n=50 best_brier=0.1855 | 2026-02-28 13:45 |
| xgb_v1 | M | 0.203340 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | M | 0.204288 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v3 | M | 0.187904 | xgb Optuna n=50 best_brier=0.1879 | 2026-02-28 13:46 |
| baseline_logreg | W | 0.149706 | seed-diff only | 2026-02-27 14:09 |
| calibrated_v1 | W | 0.146519 | Platt-scaled ensemble (pre=0.1463 post=0.1465) | 2026-02-28 14:11 |
| catboost_v1 | W | 0.159824 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | W | 0.161340 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| catboost_v3 | W | 0.147394 | catboost Optuna n=50 best_brier=0.1474 | 2026-02-28 13:55 |
| ensemble_v1 | W | 0.146340 | 4-model mean ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned | 2026-02-28 14:10 |
| histgb_v1 | W | 0.155766 | HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints | 2026-02-28 13:42 |
| histgb_v3 | W | 0.147059 | histgb Optuna n=50 best_brier=0.1471 | 2026-02-28 13:57 |
| lgbm_v1 | W | 0.167394 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | W | 0.169039 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v3 | W | 0.146463 | lgbm Optuna n=50 best_brier=0.1465 | 2026-02-28 13:51 |
| xgb_v1 | W | 0.172000 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | W | 0.170134 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v3 | W | 0.148330 | xgb Optuna n=50 best_brier=0.1483 | 2026-02-28 13:52 |
