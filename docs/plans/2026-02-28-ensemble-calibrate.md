# Ensemble + Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write `scripts/10_ensemble.py` (probability-average the 4 Optuna-tuned models with LOSO-CV evaluation) and `scripts/11_calibrate.py` (Platt scaling of ensemble OOF predictions), producing all artifacts needed by `12_predict.py`.

**Architecture:** `10_ensemble.py` re-runs LOSO-CV with each model's best saved hyperparams to collect OOF predictions, combines them with both simple-mean and rank-mean strategies, picks the better one, and saves the OOF arrays (`.npz`) plus an ensemble config (`.json`). `11_calibrate.py` loads those OOF arrays, fits Platt scaling (logistic regression on logit(p) → labels), evaluates pre/post Brier, and saves the calibrator object. Both scripts log v3-style benchmarks. The prediction script (`12_predict.py`) will load tuned models + calibrator to produce final submissions.

**Tech Stack:** `lightgbm`, `xgboost`, `catboost`, `sklearn`, `scipy.stats.rankdata`, project `utils.py`, `joblib`, `json`, `numpy`.

---

### Task 1: Write 10_ensemble.py — OOF collectors

**Files:**
- Create: `scripts/10_ensemble.py`

These are LOSO-CV runners that return `(y_true, y_pred)` arrays instead of a Brier scalar. One function per model type.

**Step 1: Create file with imports + constants**

```python
"""
10_ensemble.py -- March Machine Learning Mania 2026

Ensemble of 4 Optuna-tuned models (LGBM, XGB, CatBoost, HistGB).
Uses best hyperparams from results/best_params_*.json.
Evaluates simple-mean vs rank-mean combination on LOSO-CV OOF predictions.

Outputs:
  results/oof_preds_{M,W}.npz   -- OOF predictions for all 4 models + ensemble
  results/ensemble_config.json  -- chosen strategy per gender
  models/ensemble_config.json   -- copy for predict script
  results/benchmarks.csv / BENCHMARKS.md
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
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

MODELS = ["lgbm", "xgb", "catboost", "histgb"]
```

**Step 2: Add `load_best_params` helper**

```python
def load_best_params(model_name: str, gender: str) -> dict:
    """Load Optuna best params from results/best_params_{model}_{gender}.json."""
    path = utils.RESULTS / f"best_params_{model_name}_{gender}.json"
    with open(path) as f:
        return json.load(f)
```

**Step 3: Add four OOF prediction collectors**

These all have the same signature: `(matchups, feat_cols, cv_seasons, params) -> (y_true, y_pred)`.

```python
def _oof_lgbm(matchups, feat_cols, cv_seasons, params):
    all_true, all_pred = [], []
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
        all_true.append(matchups.loc[te, "Label"].values)
        all_pred.append(m.predict_proba(Xte)[:, 1])
    return np.concatenate(all_true), np.concatenate(all_pred)


def _oof_xgb(matchups, feat_cols, cv_seasons, params):
    all_true, all_pred = [], []
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
        all_true.append(matchups.loc[te, "Label"].values)
        all_pred.append(m.predict_proba(Xte)[:, 1])
    return np.concatenate(all_true), np.concatenate(all_pred)


def _oof_catboost(matchups, feat_cols, cv_seasons, params):
    all_true, all_pred = [], []
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
        all_true.append(matchups.loc[te, "Label"].values)
        all_pred.append(m.predict_proba(matchups.loc[te, feat_cols].values)[:, 1])
    return np.concatenate(all_true), np.concatenate(all_pred)


def _oof_histgb(matchups, feat_cols, cv_seasons, params):
    monotone = utils.build_monotone_vec(feat_cols)
    all_true, all_pred = [], []
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
        all_true.append(matchups.loc[te, "Label"].values)
        all_pred.append(m.predict_proba(matchups.loc[te, feat_cols].values)[:, 1])
    return np.concatenate(all_true), np.concatenate(all_pred)


_OOF_COLLECTORS = {
    "lgbm":     _oof_lgbm,
    "xgb":      _oof_xgb,
    "catboost": _oof_catboost,
    "histgb":   _oof_histgb,
}
```

**Step 4: Verify syntax**

```bash
cd march_learning_mania
python -c "import ast; ast.parse(open('scripts/10_ensemble.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 2: Write 10_ensemble.py — ensemble_gender() main function

**Files:**
- Modify: `scripts/10_ensemble.py` (append)

**Step 1: Add `ensemble_gender` function**

```python
def ensemble_gender(gender: str) -> float:
    """
    Collect OOF predictions for all 4 models, combine, pick best strategy.
    Returns best ensemble Brier.
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Ensemble -- {label}")
    print(f"{'='*52}")

    # Load data
    features   = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    tourney    = utils.load_tourney(gender)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)
    all_diff   = [c for c in matchups.columns if c.endswith("_diff")]
    feat_cols  = utils.curate_features(all_diff)
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    # Collect OOF predictions from each model
    preds = {}
    y_true = None
    for model_name in MODELS:
        params   = load_best_params(model_name, gender)
        collector = _OOF_COLLECTORS[model_name]
        print(f"  Collecting OOF: {model_name} ...", end=" ", flush=True)
        yt, yp   = collector(matchups, feat_cols, cv_seasons, params)
        preds[model_name] = yp
        if y_true is None:
            y_true = yt
        brier_indiv = utils.brier_score(yt, yp)
        print(f"Brier = {brier_indiv:.4f}")

    # Strategy 1: simple mean
    stack      = np.column_stack([preds[m] for m in MODELS])
    mean_pred  = stack.mean(axis=1)
    brier_mean = utils.brier_score(y_true, mean_pred)

    # Strategy 2: rank mean (each model's probs → percentile rank → average)
    rank_stack  = np.column_stack([rankdata(preds[m]) / len(preds[m]) for m in MODELS])
    rank_pred   = rank_stack.mean(axis=1)
    brier_rank  = utils.brier_score(y_true, rank_pred)

    print(f"\n  Combination strategies:")
    print(f"    mean      Brier = {brier_mean:.4f}")
    print(f"    rank_mean Brier = {brier_rank:.4f}")

    if brier_mean <= brier_rank:
        best_pred, strategy, best_brier = mean_pred, "mean", brier_mean
    else:
        best_pred, strategy, best_brier = rank_pred, "rank_mean", brier_rank

    print(f"  -> Best strategy: {strategy}  Brier = {best_brier:.4f}")

    # Save OOF arrays (needed by 11_calibrate.py)
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    npz_path = utils.RESULTS / f"oof_preds_{gender}.npz"
    np.savez(
        npz_path,
        y_true=y_true,
        y_lgbm=preds["lgbm"],
        y_xgb=preds["xgb"],
        y_catboost=preds["catboost"],
        y_histgb=preds["histgb"],
        y_ensemble=best_pred,
    )
    print(f"  Saved {npz_path.name}")

    return best_brier, strategy
```

**Step 2: Add `__main__` block**

```python
if __name__ == "__main__":
    ensemble_config = {}

    for gender in ["M", "W"]:
        best_brier, strategy = ensemble_gender(gender)
        ensemble_config[gender] = {
            "models": MODELS,
            "strategy": strategy,
            "cv_brier": round(best_brier, 6),
        }
        utils.log_benchmark(
            "ensemble_v1", gender, best_brier,
            f"4-model {strategy} ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned"
        )

    # Save ensemble config for predict script
    cfg_path = utils.RESULTS / "ensemble_config.json"
    with open(cfg_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    # Also mirror to models/ so predict script has one obvious place to look
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    with open(utils.MODELS / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "="*52)
    print("ENSEMBLE SUMMARY")
    print("="*52)
    for gender, cfg in ensemble_config.items():
        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label:<10} strategy={cfg['strategy']:<10} Brier={cfg['cv_brier']:.4f}")
    print("Saved results/ensemble_config.json and results/oof_preds_{M,W}.npz")
    print("Benchmarks updated.")
```

**Step 3: Verify full syntax**

```bash
python -c "import ast; ast.parse(open('scripts/10_ensemble.py').read()); print('OK')"
```
Expected: `OK`

**Step 4: Smoke-run ensemble (Men's only to check plumbing)**

```bash
cd march_learning_mania
python -c "
import sys; sys.path.insert(0,'scripts')
import pandas as pd, warnings
import utils
from scripts_10 import ensemble_gender  # can't do this — just run full script
"
# Actually just run the full script to smoke-test
python scripts/10_ensemble.py 2>&1 | grep -E "Brier|Saved|strategy|ERROR"
```
Expected: 8 individual Brier lines + 2 combination lines per gender + config files saved.

---

### Task 3: Write 11_calibrate.py

**Files:**
- Create: `scripts/11_calibrate.py`

Platt scaling: fit `LogisticRegression` on `logit(ensemble_pred) -> y_true`. This learns a monotone stretch/shift of the probability scale. With ~670 OOF samples and 2 parameters, it's unlikely to overfit.

**Note on OOF calibration bias:** We fit the calibrator on the same OOF predictions used to measure ensemble Brier. This gives a slightly optimistic post-calibration Brier estimate. However:
1. Platt scaling has only 2 degrees of freedom (slope + intercept) so overfitting risk is negligible with 670 samples.
2. The tournament we're predicting (2026) is genuinely held-out from both the tree models and the calibrator.

**Step 1: Create file**

```python
"""
11_calibrate.py -- March Machine Learning Mania 2026

Platt scaling calibration applied to ensemble OOF predictions.
Loads results/oof_preds_{M,W}.npz from 10_ensemble.py.

Outputs:
  results/calibrator_{M,W}.pkl  -- fitted LogisticRegression (slope + intercept)
  results/benchmarks.csv / BENCHMARKS.md
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression


def platt_calibrate(gender: str) -> tuple[float, float]:
    """
    Fit Platt scaling on ensemble OOF predictions.
    Returns (brier_before, brier_after).
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Calibration -- {label}")
    print(f"{'='*52}")

    # Load OOF arrays produced by 10_ensemble.py
    npz = np.load(utils.RESULTS / f"oof_preds_{gender}.npz")
    y_true    = npz["y_true"]
    ens_pred  = npz["y_ensemble"]

    brier_before = utils.brier_score(y_true, ens_pred)
    print(f"  Pre-calibration  Brier = {brier_before:.4f}")

    # Platt scaling: fit logistic regression on logit(p) -> labels
    eps      = 1e-7
    log_odds = np.log(
        np.clip(ens_pred, eps, 1 - eps) / np.clip(1 - ens_pred, eps, 1 - eps)
    )
    platt = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    platt.fit(log_odds.reshape(-1, 1), y_true)

    cal_pred     = platt.predict_proba(log_odds.reshape(-1, 1))[:, 1]
    brier_after  = utils.brier_score(y_true, cal_pred)
    print(f"  Post-calibration Brier = {brier_after:.4f}  "
          f"(delta = {brier_after - brier_before:+.4f})")
    print(f"  Platt params: slope={platt.coef_[0][0]:.4f}  "
          f"intercept={platt.intercept_[0]:.4f}")

    # Save calibrator
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    cal_path = utils.RESULTS / f"calibrator_{gender}.pkl"
    joblib.dump({"platt": platt}, cal_path)
    print(f"  Saved {cal_path.name}")

    return brier_before, brier_after


if __name__ == "__main__":
    for gender in ["M", "W"]:
        brier_before, brier_after = platt_calibrate(gender)
        utils.log_benchmark(
            "calibrated_v1", gender, brier_after,
            f"Platt-scaled ensemble (before={brier_before:.4f})"
        )

    print("\n" + "="*52)
    print("CALIBRATION SUMMARY")
    print("="*52)
    print("Saved results/calibrator_{M,W}.pkl")
    print("Benchmarks updated.")
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/11_calibrate.py').read()); print('OK')"
```
Expected: `OK`

---

### Task 4: Run both scripts end-to-end + commit

**Step 1: Run 10_ensemble.py**

```bash
cd march_learning_mania
python scripts/10_ensemble.py 2>&1 | grep -v "UserWarning\|warnings.warn\|LGBMClassifier"
```

Expected output pattern:
```
====================================================
Ensemble -- Men's
====================================================
  Collecting OOF: lgbm ... Brier = 0.18XX
  Collecting OOF: xgb ... Brier = 0.18XX
  Collecting OOF: catboost ... Brier = 0.18XX
  Collecting OOF: histgb ... Brier = 0.18XX

  Combination strategies:
    mean      Brier = 0.18XX
    rank_mean Brier = 0.18XX
  -> Best strategy: mean  Brier = 0.18XX
  Saved oof_preds_M.npz
...
```

**Step 2: Verify OOF files exist**

```bash
ls results/oof_preds_*.npz results/ensemble_config.json models/ensemble_config.json
```
Expected: 4 files.

**Step 3: Run 11_calibrate.py**

```bash
python scripts/11_calibrate.py
```

Expected output pattern:
```
====================================================
Calibration -- Men's
====================================================
  Pre-calibration  Brier = 0.18XX
  Post-calibration Brier = 0.18XX  (delta = -0.00XX)
  Platt params: slope=X.XXXX  intercept=X.XXXX
  Saved calibrator_M.pkl
```

**Step 4: Check calibrator files exist**

```bash
ls results/calibrator_*.pkl
```
Expected: `calibrator_M.pkl`, `calibrator_W.pkl`

**Step 5: Print updated benchmark table**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/benchmarks.csv')[['model','gender','cv_brier']]
pivot = df.pivot(index='model', columns='gender', values='cv_brier')
print(pivot.sort_values('M').to_string())
"
```

**Step 6: Commit**

```bash
git add scripts/10_ensemble.py scripts/11_calibrate.py \
        results/ensemble_config.json results/oof_preds_*.npz \
        results/calibrator_*.pkl results/benchmarks.csv BENCHMARKS.md
git commit -m "feat: ensemble (10) + Platt calibration (11) — complete pipeline to submission"
```

---

## Acceptance Criteria

- [ ] `results/oof_preds_M.npz` and `oof_preds_W.npz` exist with keys: `y_true`, `y_lgbm`, `y_xgb`, `y_catboost`, `y_histgb`, `y_ensemble`
- [ ] `results/ensemble_config.json` exists with `M` and `W` keys, each containing `models`, `strategy`, `cv_brier`
- [ ] `results/calibrator_M.pkl` and `calibrator_W.pkl` exist with `platt` key
- [ ] `BENCHMARKS.md` has `ensemble_v1` and `calibrated_v1` rows for both genders
- [ ] Ensemble Brier ≤ best individual model Brier for at least one gender
- [ ] Calibration Brier ≤ ensemble Brier (or within 0.001 — Platt may be near-no-op if models are already well-calibrated)

---

## Notes

- **Why rank-mean?** Rank transformation removes absolute calibration differences between models (e.g., if CatBoost systematically outputs higher probabilities). Simple mean is better when models are already similarly calibrated.
- **Why Platt scaling?** With only ~670 OOF samples, isotonic regression would overfit. Platt scaling is 2 parameters — slope and intercept — making it very robust.
- **Next step:** `12_predict.py` — load each `{model}_tuned_{gender}.pkl`, predict on Stage 1 + Stage 2 matchups, combine with `ensemble_config["strategy"]`, apply `calibrator_{gender}.pkl`, write submission CSV.
