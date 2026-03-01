# Benchmarks

CV Brier scores for all models (lower is better). Auto-updated by training scripts.

*Source data: `results/benchmarks.csv`*

| Model | Gender | CV Brier | Notes | Timestamp |
|-------|--------|----------|-------|-----------|
| baseline_logreg | M | 0.196454 | seed-diff only | 2026-02-27 14:09 |
| calibrated_v1 | M | 0.184544 | Platt-scaled ensemble (pre=0.1846 post=0.1845) | 2026-03-01 19:22 |
| catboost_v1 | M | 0.200576 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | M | 0.201369 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| catboost_v3 | M | 0.185371 | catboost Optuna n=50 best_brier=0.1854 | 2026-03-01 19:14 |
| ensemble_v1 | M | 0.184625 | 4-model mean ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned | 2026-03-01 19:22 |
| histgb_v1 | M | 0.202765 | HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints | 2026-02-28 13:42 |
| histgb_v3 | M | 0.186309 | histgb Optuna n=50 best_brier=0.1863 | 2026-03-01 19:19 |
| lgbm_v1 | M | 0.202406 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | M | 0.203346 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v3 | M | 0.186021 | lgbm Optuna n=50 best_brier=0.1860 | 2026-03-01 19:05 |
| meta_v1 | M | 0.184361 | LogReg meta-learner on logit OOF (mean=0.1846 -> meta=0.1844) | 2026-03-01 19:22 |
| meta_v2 | M | 0.187005 | logreg meta-learner (lgbm=0.1879 logreg=0.1870) | 2026-03-01 19:25 |
| xgb_v1 | M | 0.203340 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | M | 0.204288 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v3 | M | 0.184831 | xgb Optuna n=50 best_brier=0.1848 | 2026-03-01 19:07 |
| baseline_logreg | W | 0.149706 | seed-diff only | 2026-02-27 14:09 |
| calibrated_v1 | W | 0.146278 | Platt-scaled ensemble (pre=0.1463 post=0.1463) | 2026-03-01 19:22 |
| catboost_v1 | W | 0.159824 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | W | 0.161340 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| catboost_v3 | W | 0.144319 | catboost Optuna n=50 best_brier=0.1443 | 2026-03-01 19:17 |
| ensemble_v1 | W | 0.146274 | 4-model mean ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned | 2026-03-01 19:22 |
| histgb_v1 | W | 0.155766 | HistGBClassifier iter=500 lr=0.05 md=4 monotone-constraints | 2026-02-28 13:42 |
| histgb_v3 | W | 0.148803 | histgb Optuna n=50 best_brier=0.1488 | 2026-03-01 19:21 |
| lgbm_v1 | W | 0.167394 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | W | 0.169039 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v3 | W | 0.149248 | lgbm Optuna n=50 best_brier=0.1492 | 2026-03-01 19:06 |
| meta_v1 | W | 0.143871 | LogReg meta-learner on logit OOF (mean=0.1463 -> meta=0.1439) | 2026-03-01 19:22 |
| meta_v2 | W | 0.145991 | logreg meta-learner (lgbm=0.1500 logreg=0.1460) | 2026-03-01 19:25 |
| xgb_v1 | W | 0.172000 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | W | 0.170134 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v3 | W | 0.147552 | xgb Optuna n=50 best_brier=0.1476 | 2026-03-01 19:08 |
