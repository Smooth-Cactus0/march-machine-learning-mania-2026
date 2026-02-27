# Benchmarks

CV Brier scores for all models (lower is better). Auto-updated by training scripts.

*Source data: `results/benchmarks.csv`*

| Model | Gender | CV Brier | Notes | Timestamp |
|-------|--------|----------|-------|-----------|
| baseline_logreg | M | 0.196454 | seed-diff only | 2026-02-27 14:09 |
| lgbm_v1 | M | 0.238931 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:05 |
| baseline_logreg | W | 0.149706 | seed-diff only | 2026-02-27 14:09 |
| lgbm_v1 | W | 0.176829 | LGBMClassifier n=500 lr=0.05 md=4 | 2026-02-27 16:05 |
