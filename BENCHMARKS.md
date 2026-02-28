# Benchmarks

CV Brier scores for all models (lower is better). Auto-updated by training scripts.

*Source data: `results/benchmarks.csv`*

| Model | Gender | CV Brier | Notes | Timestamp |
|-------|--------|----------|-------|-----------|
| baseline_logreg | M | 0.196454 | seed-diff only | 2026-02-27 14:09 |
| catboost_v1 | M | 0.200576 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | M | 0.201369 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v1 | M | 0.202406 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | M | 0.203346 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v1 | M | 0.203340 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | M | 0.204288 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| baseline_logreg | W | 0.149706 | seed-diff only | 2026-02-27 14:09 |
| catboost_v1 | W | 0.159824 | CatBoostClassifier iter=500 lr=0.05 depth=4 | 2026-02-27 16:06 |
| catboost_v2 | W | 0.161340 | CatBoostClassifier iter=500 lr=0.05 depth=4 curated-feats | 2026-02-28 13:23 |
| lgbm_v1 | W | 0.167394 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:14 |
| lgbm_v2 | W | 0.169039 | LGBMClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
| xgb_v1 | W | 0.172000 | XGBClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:15 |
| xgb_v2 | W | 0.170134 | XGBClassifier n=500 lr=0.05 md=4 curated-feats | 2026-02-28 13:23 |
