# Improvement Design — March Machine Learning Mania 2026

> **Date:** 2026-03-01
> **Session context:** After achieving LB=0.125 with the LogReg meta-learner on Stage 1.

---

## Current State

| Component | Status | Score |
|-----------|--------|-------|
| 18 curated features | Done | — |
| 4 Optuna-tuned models (LGBM, XGB, CatBoost, HistGB) | Done | M=0.1850, W=0.1459 CV |
| Mean ensemble | Done | M=0.1866, W=0.1463 CV |
| LogReg meta-learner (logit OOF) | Done | M=0.1850, W=0.1459 CV |
| **Kaggle LB (Stage 1, meta)** | **Done** | **0.125** |

---

## Plan Overview

Four parallel workstreams, to be executed in order:

1. **GitHub repo polish** — README, docs, pipeline documentation
2. **Kaggle notebooks** — two pedagogical notebooks (Feature Engineering + Modelling)
3. **Direction A** — Elo + Massey PCA + momentum features
4. **Direction C** — LightGBM meta-learner with LogReg fallback

Direction B (more Optuna trials) deferred to end of project.

---

## Workstream 1: GitHub Repo

**Goal:** Polish the current project for public presentation without restructuring it.

### Files to create

#### `README.md`
- Competition overview: what we're predicting, Brier score metric, Stage 1 vs Stage 2
- Architecture diagram (ASCII): Raw CSVs → Feature Engineering → 4 Models → Ensemble → Meta-learner → Submission
- Results table:

| Submission | CV Brier (M) | CV Brier (W) | LB Stage 1 |
|------------|-------------|-------------|------------|
| Seed baseline | 0.1965 | 0.1497 | — |
| Mean ensemble (Optuna) | 0.1866 | 0.1463 | 0.1384 |
| Meta-learner (LogReg) | 0.1850 | 0.1459 | 0.125 |

- Reproduction instructions: install requirements → download data → run scripts 01–13 in order
- Link to both Kaggle notebooks

#### `docs/SETUP.md`
- Python 3.11 requirement
- `pip install -r requirements.txt`
- Where to download competition data (Kaggle link)
- Expected directory structure after data download

#### `docs/PIPELINE.md`
- Table: each script, what it reads, what it writes, one-line description

### Key message
The repo tells a progression: seed baseline (0.196) → Optuna-tuned ensemble (0.1384 LB) → meta-learner stacking (0.125 LB). Every intermediate benchmark is preserved in `BENCHMARKS.md`.

---

## Workstream 2: Kaggle Notebooks

Both notebooks are **fully self-contained**: read from `/kaggle/input/march-machine-learning-mania-2026/`, compute everything from scratch, produce a submission CSV. Audience is intermediate-level Kaggle users who want to understand the approach, not just copy predictions.

### Notebook 1: Feature Engineering

Target length: ~2,500 words of commentary + all code cells.

| Section | Content |
|---------|---------|
| 1. Introduction | Tournament format, why Brier score, what we're predicting |
| 2. Data tour | `.head()` + shape of each key CSV |
| 3. Efficiency metrics | Four-factor model, Kubatko possessions formula, net efficiency |
| 4. Massey Ordinals deep dive | 196 systems overview, PCA rationale, explained-variance curve |
| 5. Elo ratings | Formula derivation, margin-of-victory K-factor, season carryover |
| 6. Late-season momentum | Last-10-games split, peak vs. decline examples |
| 7. Feature correlation heatmap | Justify curated feature set, show collinearity |
| 8. Save | Unified parquet output per gender |

### Notebook 2: Modelling Pipeline

Target length: ~2,500 words of commentary + all code cells.

| Section | Content |
|---------|---------|
| 1. CV strategy | LOSO-CV explained with diagram, 2020 exclusion |
| 2. Baseline | Seed-diff logistic regression (Brier 0.196) |
| 3. LightGBM | Feature importance, Optuna search (parallel coordinates plot) |
| 4. Ensemble | Mean vs rank-mean comparison |
| 5. Meta-learner | Stacking explanation, logit-space weights, gain over mean ensemble |
| 6. Submission | Parse IDs, gender split, write CSV |

---

## Workstream 3: Direction A — New Features

All three feature groups added to `02_feature_engineering.py` and registered in
`utils.CURATED_FEATURES` + `utils._MONOTONE_MAP`.

### 3a. Margin-of-Victory Elo

**Source:** `M/WRegularSeasonCompactResults.csv`

**Formula:**
```
expected_A = 1 / (1 + 10^((elo_B - elo_A) / 400))
K_eff      = K_base * (1 + margin / 20)^0.6      # K_base = 20
delta      = K_eff * (outcome - expected_A)
```

**Season carryover (mean reversion):**
```
elo_start_of_season = elo_end_prev_season * 0.75 + 1500 * 0.25
```
Teams new to the data initialise at 1500.

**New features per (Season, TeamID):**
- `elo_rating` — end-of-season Elo (after all regular season games)
- `elo_k_weighted_wins` — sum of K_eff for all wins (captures dominance, not just record)

**Monotone constraints:** `elo_rating_diff: +1`, `elo_k_weighted_wins_diff: +1`

### 3b. Massey PCA Composite

**Source:** `MMasseyOrdinals.csv` (Men's only; Women's → NaN)

**Steps:**
1. Filter to `RankingDayNum <= 133` (pre-tournament), last available rank per (Season, TeamID, SystemName)
2. Keep systems with ≥ 50% team coverage for the season
3. Pivot to (Season, TeamID) × SystemName matrix
4. Flip sign: `rank_score = max_rank - rank + 1` so higher = better
5. Standardise per system (z-score), fill NaN with 0 (= average)
6. PCA → keep first 2 components: `massey_pc1`, `massey_pc2`
   - PC1 ≈ consensus ranking (~80% variance)
   - PC2 ≈ orthogonal disagreement between systems

**Monotone constraints:** `massey_pc1_diff: +1`, `massey_pc2_diff: 0` (direction unclear)

### 3c. Late-Season Momentum

**Source:** `M/WRegularSeasonCompactResults.csv`

**Definition:** last 10 regular season games by `DayNum` per (Season, TeamID)

**New features:**
- `recent_win_pct` — win% in last 10 games
- `recent_net_margin` — mean point margin in last 10 games
- `streak` — current streak entering tournament: positive = win streak length, negative = loss streak length

**Monotone constraints:** `recent_win_pct_diff: +1`, `recent_net_margin_diff: +1`, `streak_diff: +1`

### Integration
After adding features to `02_feature_engineering.py`:
- Re-run `02_feature_engineering.py` to regenerate parquets
- Re-run `09_tune_optuna.py` for all 4 models (new features change the optimal hyperparameters)
- Re-run `10_ensemble.py`, `11_calibrate.py`, `13_meta_learner.py`
- Generate new submission → Kaggle benchmark

---

## Workstream 4: Direction C — LightGBM Meta-Learner

**Script:** `scripts/14_meta_lgbm.py` (alongside existing `13_meta_learner.py`)

### Inputs (per gender, N ≈ 670 rows after Direction A)
- 4 logit-transformed OOF predictions: `logit_lgbm`, `logit_xgb`, `logit_catboost`, `logit_histgb`
- 2 direct features: `SeedNum_diff`, `massey_pc1_diff` (anchor to strongest raw signals)

Total: 6 features, N ≈ 670 → requires aggressive regularisation.

### Model spec
```python
LGBMClassifier(
    objective="binary",
    metric="binary_logloss",
    num_leaves=4,       # tuned by Optuna: range 3–16
    min_child_samples=40,  # tuned: range 20–80
    n_estimators=200,   # with early stopping on held-out fold
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,     # tuned: range 0.1–10
)
```

### CV strategy for meta-learner
5-fold stratified CV (stratified by season decade), 30 Optuna trials.
LOSO-CV not used here — N≈670 per fold is ~67 samples, too small for tree models.

### Fallback rule
After tuning:
- If LightGBM CV Brier < LogReg CV Brier for **both** genders → use LightGBM
- If mixed (one gender better, one worse) → use LogReg (conservative)
- If LogReg wins both → use LogReg

The submission script (`15_predict_v2.py` or updated `12_predict.py`) auto-selects
based on saved benchmark comparison.

---

## Execution Order

```
# Workstream 1
Write README.md, docs/SETUP.md, docs/PIPELINE.md → commit

# Workstream 2
Write notebooks/01_feature_engineering.ipynb → test on Kaggle
Write notebooks/02_modelling_pipeline.ipynb   → test on Kaggle

# Workstream 3
Update scripts/02_feature_engineering.py (Elo + PCA + momentum)
Update scripts/utils.py (CURATED_FEATURES + monotone map)
Run 02 → 09 → 10 → 11 → 13 → generate submission → Kaggle

# Workstream 4
Write scripts/14_meta_lgbm.py
Run → compare vs LogReg → generate submission if better → Kaggle
```

---

## Acceptance Criteria

- [ ] `README.md` explains pipeline end-to-end with LB scores
- [ ] `docs/SETUP.md` and `docs/PIPELINE.md` exist
- [ ] Notebook 1 runs end-to-end on Kaggle in < 30 min
- [ ] Notebook 2 runs end-to-end on Kaggle in < 30 min
- [ ] New features (Elo, Massey PCA, momentum) improve CV Brier vs current baseline
- [ ] New submission generated and benchmarked on Kaggle Stage 1
- [ ] LightGBM meta-learner evaluated; LogReg kept as fallback if not better
