# Optuna Hyperparameter Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write `scripts/09_tune_optuna.py` that runs Optuna hyperparameter search for all four models (LGBM, XGB, CatBoost, HistGB), saves best parameters, retrains final tuned models, and logs v3 benchmarks.

**Architecture:** One Optuna `Study` per `(model, gender)` combination (8 total). Each trial evaluates LOSO-CV mean Brier over the last 10 seasons. After `N_TRIALS=50` trials, the best params are saved as JSON, the final model is retrained on all data, and the result is logged. The monotonic-constraint vector used by HistGB is factored into `utils.py` (shared with `08_train_histgb.py`) to keep logic DRY.

**Tech Stack:** `optuna==4.6.0`, `lightgbm`, `xgboost`, `catboost`, `sklearn.ensemble.HistGradientBoostingClassifier`, `sklearn.impute.SimpleImputer`, project `utils.py`.

---

### Task 1: Move monotone constraints into utils.py

The monotone constraint dict currently lives only in `08_train_histgb.py`. The Optuna script also needs it for HistGB trials. Factor it into `utils.py` once, import in both places.

**Files:**
- Modify: `scripts/utils.py` (after `curate_features`)
- Modify: `scripts/08_train_histgb.py` (remove local copy, import from utils)

**Step 1: Add to utils.py**

Add immediately after `curate_features()`:

```python
# ── Monotonic constraints for HistGB / any sklearn model ──────────────────────
# +1 = higher diff → higher win prob (quality metrics)
# -1 = higher diff → lower win prob  (SeedNum: lower number = better team;
#      sos_massey/massey ranks: lower number = better rank; to_pct: worse team)
#  0 = no strong prior
_MONOTONE_MAP = {
    "SeedNum_diff":               -1,
    "massey_composite_diff":      -1,
    "massey_POM_diff":            -1,
    "massey_MOR_diff":            -1,
    "sos_massey_diff":            -1,
    "net_eff_diff":               +1,
    "efg_pct_diff":               +1,
    "oreb_pct_diff":              +1,
    "dreb_pct_diff":              +1,
    "ft_rate_diff":               +1,
    "win_pct_diff":               +1,
    "avg_margin_diff":            +1,
    "neutral_win_pct_diff":       +1,
    "neutral_net_eff_diff":       +1,
    "is_power_conf_diff":         +1,
    "conf_tourney_wins_diff":     +1,
    "coach_years_at_school_diff":  0,
    "to_pct_diff":                -1,
}


def build_monotone_vec(feat_cols: list) -> list:
    """
    Return monotonic constraint list aligned to feat_cols.
    +1 = increasing, -1 = decreasing, 0 = unconstrained.
    Defaults to 0 for any column not in _MONOTONE_MAP.
    """
    return [_MONOTONE_MAP.get(c, 0) for c in feat_cols]
```

**Step 2: Update 08_train_histgb.py**

Remove the local `_MONOTONE_CONSTRAINTS` dict and `build_monotone_vec` function.
Replace with:

```python
from utils import build_monotone_vec  # already on sys.path via sys.path.insert
```

Verify the existing call `monotone = build_monotone_vec(FEATURE_COLS)` still works.

**Step 3: Smoke-test the import**

```bash
cd "march_learning_mania"
python -c "import sys; sys.path.insert(0,'scripts'); from utils import build_monotone_vec; print(build_monotone_vec(['SeedNum_diff','net_eff_diff']))"
```
Expected output: `[-1, 1]`

**Step 4: Re-run 08 to confirm nothing broke**

```bash
python scripts/08_train_histgb.py 2>&1 | grep "Mean Brier"
```
Expected: same scores as before (`M ~0.2028, W ~0.1558`).

**Step 5: Commit**

```bash
git add scripts/utils.py scripts/08_train_histgb.py
git commit -m "refactor: move monotone constraints to utils.py for reuse in Optuna script"
```

---

### Task 2: Write the Optuna CV helper functions

These are the per-model CV loops called inside each Optuna trial. They must be fast and suppress all warnings/output.

**Files:**
- Create: `scripts/09_tune_optuna.py` (initial skeleton + helpers)

**Step 1: Create the file with imports and constants**

```python
"""
09_tune_optuna.py — March Machine Learning Mania 2026

Optuna hyperparameter search for LGBM, XGB, CatBoost, HistGB.
50 trials per (model, gender); LOSO-CV Brier is the objective.

Outputs:
  results/best_params_{model}_{gender}.json   — best hyperparameters
  models/{model}_tuned_{gender}.pkl           — retrained final models
  results/benchmarks.csv / BENCHMARKS.md      — v3 benchmarks
"""

import sys
import json
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
```

**Step 2: Add shared data loader**

```python
def load_data(gender: str):
    """Return (matchups, feat_cols, cv_seasons) for one gender."""
    features   = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    tourney    = utils.load_tourney(gender)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)
    all_diff   = [c for c in matchups.columns if c.endswith("_diff")]
    feat_cols  = utils.curate_features(all_diff)
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)
    return matchups, feat_cols, cv_seasons
```

**Step 3: Add four CV runner functions**

```python
def _run_lgbm_cv(matchups, feat_cols, cv_seasons, params):
    briers = []
    for season in cv_seasons:
        tm = matchups["Season"] != season
        te = matchups["Season"] == season
        if te.sum() == 0:
            continue
        imp = SimpleImputer(strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtr = imp.fit_transform(matchups.loc[tm, feat_cols])
            Xte = imp.transform(matchups.loc[te, feat_cols])
        m = LGBMClassifier(**params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, matchups.loc[tm, "Label"].values)
        briers.append(utils.brier_score(
            matchups.loc[te, "Label"].values, m.predict_proba(Xte)[:, 1]
        ))
    return float(np.mean(briers))


def _run_xgb_cv(matchups, feat_cols, cv_seasons, params):
    briers = []
    for season in cv_seasons:
        tm = matchups["Season"] != season
        te = matchups["Season"] == season
        if te.sum() == 0:
            continue
        imp = SimpleImputer(strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtr = imp.fit_transform(matchups.loc[tm, feat_cols])
            Xte = imp.transform(matchups.loc[te, feat_cols])
        try:
            m = XGBClassifier(**params, use_label_encoder=False,
                              eval_metric="logloss", random_state=42, verbosity=0)
        except TypeError:
            m = XGBClassifier(**params, eval_metric="logloss",
                              random_state=42, verbosity=0)
        m.fit(Xtr, matchups.loc[tm, "Label"].values)
        briers.append(utils.brier_score(
            matchups.loc[te, "Label"].values, m.predict_proba(Xte)[:, 1]
        ))
    return float(np.mean(briers))


def _run_catboost_cv(matchups, feat_cols, cv_seasons, params):
    briers = []
    for season in cv_seasons:
        tm = matchups["Season"] != season
        te = matchups["Season"] == season
        if te.sum() == 0:
            continue
        m = CatBoostClassifier(**params, nan_mode="Min",
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=42, verbose=False)
        m.fit(matchups.loc[tm, feat_cols].values,
              matchups.loc[tm, "Label"].values)
        briers.append(utils.brier_score(
            matchups.loc[te, "Label"].values,
            m.predict_proba(matchups.loc[te, feat_cols].values)[:, 1]
        ))
    return float(np.mean(briers))


def _run_histgb_cv(matchups, feat_cols, cv_seasons, params, monotone):
    briers = []
    for season in cv_seasons:
        tm = matchups["Season"] != season
        te = matchups["Season"] == season
        if te.sum() == 0:
            continue
        m = HistGradientBoostingClassifier(
            **params, monotonic_cst=monotone,
            early_stopping=False, random_state=42
        )
        m.fit(matchups.loc[tm, feat_cols].values,
              matchups.loc[tm, "Label"].values)
        briers.append(utils.brier_score(
            matchups.loc[te, "Label"].values,
            m.predict_proba(matchups.loc[te, feat_cols].values)[:, 1]
        ))
    return float(np.mean(briers))
```

**Step 4: Verify file is syntactically valid**

```bash
python -c "import ast; ast.parse(open('scripts/09_tune_optuna.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 3: Write the four Optuna objective factories

Each factory closes over the data and returns an `objective(trial)` function for Optuna.

**Files:**
- Modify: `scripts/09_tune_optuna.py` (append after CV helpers)

**Step 1: LGBM objective**

```python
def make_lgbm_objective(matchups, feat_cols, cv_seasons):
    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 50, 400),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth         = trial.suggest_int("max_depth", 2, 5),
            num_leaves        = trial.suggest_int("num_leaves", 7, 31),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 30),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 3.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        return _run_lgbm_cv(matchups, feat_cols, cv_seasons, p)
    return objective
```

**Step 2: XGB objective**

```python
def make_xgb_objective(matchups, feat_cols, cv_seasons):
    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 50, 400),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth        = trial.suggest_int("max_depth", 2, 5),
            min_child_weight = trial.suggest_int("min_child_weight", 3, 20),
            gamma            = trial.suggest_float("gamma", 0.0, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 3.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        return _run_xgb_cv(matchups, feat_cols, cv_seasons, p)
    return objective
```

**Step 3: CatBoost objective**

```python
def make_catboost_objective(matchups, feat_cols, cv_seasons):
    def objective(trial):
        p = dict(
            iterations        = trial.suggest_int("iterations", 50, 400),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth             = trial.suggest_int("depth", 2, 6),
            l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            min_data_in_leaf  = trial.suggest_int("min_data_in_leaf", 5, 30),
        )
        return _run_catboost_cv(matchups, feat_cols, cv_seasons, p)
    return objective
```

**Step 4: HistGB objective**

```python
def make_histgb_objective(matchups, feat_cols, cv_seasons):
    monotone = utils.build_monotone_vec(feat_cols)
    def objective(trial):
        p = dict(
            max_iter          = trial.suggest_int("max_iter", 50, 400),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth         = trial.suggest_int("max_depth", 2, 5),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 5, 30),
            l2_regularization = trial.suggest_float("l2_regularization", 0.1, 20.0, log=True),
        )
        return _run_histgb_cv(matchups, feat_cols, cv_seasons, p, monotone)
    return objective
```

**Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/09_tune_optuna.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 4: Write the tune_model orchestrator and final retraining

`tune_model` creates the study, runs N_TRIALS, saves params JSON, retrains the final model on all data, saves it, and logs the benchmark.

**Files:**
- Modify: `scripts/09_tune_optuna.py` (append)

**Step 1: Add tune_model function**

```python
_OBJECTIVE_FACTORIES = {
    "lgbm":    make_lgbm_objective,
    "xgb":     make_xgb_objective,
    "catboost": make_catboost_objective,
    "histgb":  make_histgb_objective,
}


def tune_model(model_name: str, gender: str) -> float:
    """
    Run Optuna study, save best params JSON, retrain final model.
    Returns best CV Brier.
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Tuning {model_name.upper()} — {label}  ({N_TRIALS} trials)")
    print(f"{'='*52}")

    matchups, feat_cols, cv_seasons = load_data(gender)

    factory   = _OBJECTIVE_FACTORIES[model_name]
    objective = factory(matchups, feat_cols, cv_seasons)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_brier  = study.best_value
    best_params = study.best_params
    print(f"  Best Brier: {best_brier:.4f}  params: {best_params}")

    # Save best params
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    params_path = utils.RESULTS / f"best_params_{model_name}_{gender}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Retrain final model on ALL data with best params
    _retrain_final(model_name, gender, matchups, feat_cols, best_params)

    # Log benchmark
    notes = f"{model_name} Optuna n={N_TRIALS} best_brier={best_brier:.4f}"
    utils.log_benchmark(f"{model_name}_v3", gender, best_brier, notes)

    return best_brier
```

**Step 2: Add _retrain_final helper**

```python
def _retrain_final(model_name, gender, matchups, feat_cols, params):
    """Retrain with best params on all available data and save to models/."""
    X_all = matchups[feat_cols].values
    y_all = matchups["Label"].values

    if model_name == "lgbm":
        imp   = SimpleImputer(strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_imp = imp.fit_transform(matchups[feat_cols])
        m = LGBMClassifier(**params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_imp, y_all)
        artifact = {"model": m, "imputer": imp, "feature_cols": feat_cols}

    elif model_name == "xgb":
        imp   = SimpleImputer(strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_imp = imp.fit_transform(matchups[feat_cols])
        try:
            m = XGBClassifier(**params, use_label_encoder=False,
                              eval_metric="logloss", random_state=42, verbosity=0)
        except TypeError:
            m = XGBClassifier(**params, eval_metric="logloss",
                              random_state=42, verbosity=0)
        m.fit(X_imp, y_all)
        artifact = {"model": m, "imputer": imp, "feature_cols": feat_cols}

    elif model_name == "catboost":
        m = CatBoostClassifier(**params, nan_mode="Min",
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=42, verbose=False)
        m.fit(X_all, y_all)
        artifact = {"model": m, "feature_cols": feat_cols}

    else:  # histgb
        monotone = utils.build_monotone_vec(feat_cols)
        m = HistGradientBoostingClassifier(
            **params, monotonic_cst=monotone,
            early_stopping=False, random_state=42
        )
        m.fit(X_all, y_all)
        artifact = {"model": m, "feature_cols": feat_cols}

    utils.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, utils.MODELS / f"{model_name}_tuned_{gender}.pkl")
    print(f"  Saved models/{model_name}_tuned_{gender}.pkl")
```

**Step 3: Add __main__ block**

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna tuning for all models")
    parser.add_argument("--model",  choices=["lgbm","xgb","catboost","histgb","all"],
                        default="all")
    parser.add_argument("--gender", choices=["M","W","both"], default="both")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    N_TRIALS = args.trials
    models  = list(_OBJECTIVE_FACTORIES.keys()) if args.model == "all" else [args.model]
    genders = ["M", "W"] if args.gender == "both" else [args.gender]

    summary = {}
    for gender in genders:
        for model in models:
            best = tune_model(model, gender)
            summary[f"{model}_{gender}"] = round(best, 4)

    print("\n" + "="*52)
    print("TUNING SUMMARY (v3 benchmarks)")
    print("="*52)
    for k, v in summary.items():
        print(f"  {k:<20} {v:.4f}")
    print("Benchmarks updated in results/benchmarks.csv and BENCHMARKS.md")
```

**Step 4: Verify full syntax**

```bash
python -c "import ast; ast.parse(open('scripts/09_tune_optuna.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 5: Smoke test one (model, gender) before full run

Run a 3-trial smoke test on LGBM Men's to confirm the plumbing works end to end before committing to the full ~20 min run.

**Step 1: Quick smoke test**

```bash
cd march_learning_mania
python scripts/09_tune_optuna.py --model lgbm --gender M --trials 3 2>&1 | tail -20
```

Expected output (numbers will vary):
```
==================================================
Tuning LGBM — Men's  (3 trials)
==================================================
  Best Brier: 0.19XX  params: {'n_estimators': ..., ...}
  Saved models/lgbm_tuned_M.pkl
```
`results/best_params_lgbm_M.json` should exist.

**Step 2: Verify benchmark was logged**

```bash
python -c "import pandas as pd; print(pd.read_csv('results/benchmarks.csv')[['model','gender','cv_brier']].tail(3).to_string())"
```
Expected: last row shows `lgbm_v3 | M | 0.19XX`

**Step 3: Commit smoke-test version**

```bash
git add scripts/09_tune_optuna.py scripts/utils.py scripts/08_train_histgb.py
git commit -m "feat: Optuna tuning script (09_tune_optuna.py) with 4 model objectives"
```

---

### Task 6: Full Optuna run — all models, both genders

Run the full 50-trial search for all 8 (model, gender) combinations (~20 minutes total).

**Step 1: Run full tuning**

```bash
python scripts/09_tune_optuna.py --model all --gender both --trials 50 2>&1 | tee results/optuna_run.log
```

Expected: 8 study completions, each printing best Brier and params.

**Step 2: Inspect summary**

```bash
tail -20 results/optuna_run.log
```

Target thresholds (these would represent clear improvement over v2 defaults):
- Men's: any model ≤ 0.1960 (beats seed-diff baseline)
- Women's: HistGB already at 0.1558; target ≤ 0.1490 (beats baseline)

**Step 3: Check all 8 best_params files exist**

```bash
ls results/best_params_*.json
```
Expected: 8 files (lgbm_M, lgbm_W, xgb_M, xgb_W, catboost_M, catboost_W, histgb_M, histgb_W)

**Step 4: Check all 8 tuned models saved**

```bash
ls models/*_tuned_*.pkl
```
Expected: 8 files.

**Step 5: Print full benchmark table**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/benchmarks.csv')[['model','gender','cv_brier']]
pivot = df.pivot(index='model', columns='gender', values='cv_brier')
print(pivot.sort_values('M').to_string())
"
```

**Step 6: Commit results**

```bash
git add results/best_params_*.json results/benchmarks.csv BENCHMARKS.md results/optuna_run.log
git commit -m "feat: Optuna v3 benchmarks — best params from 50-trial TPE search"
```

---

## Acceptance Criteria

- [ ] All 8 `best_params_{model}_{gender}.json` files saved under `results/`
- [ ] All 8 `{model}_tuned_{gender}.pkl` files saved under `models/`
- [ ] At least one model beats the seed-diff baseline for Men's (≤ 0.1965)
- [ ] At least two models beat the seed-diff baseline for Women's (≤ 0.1497)
- [ ] `BENCHMARKS.md` updated with v3 rows
- [ ] `build_monotone_vec` lives in `utils.py`, imported by both `08` and `09`

---

## Notes

- **Runtime**: ~20 min for 50 trials × 4 models × 2 genders on a local machine with GTX 1050.
- **Increasing trials**: Re-run with `--trials 100` if initial results are marginal. TPE converges well by 50 trials for 9-parameter spaces.
- **Per-model tuning**: Use `--model lgbm --gender M` etc. to re-tune a single combination without re-running all 8.
- **Next steps after this plan**: `10_ensemble.py` (rank-average + stacking of tuned models), `11_calibrate.py` (Platt scaling), `12_predict.py` (Stage 1 + Stage 2 submission generation).
