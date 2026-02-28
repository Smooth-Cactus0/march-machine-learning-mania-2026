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


# ── Best-params loader ────────────────────────────────────────────────────────

def load_best_params(model_name: str, gender: str) -> dict:
    """Load Optuna best params from results/best_params_{model}_{gender}.json."""
    path = utils.RESULTS / f"best_params_{model_name}_{gender}.json"
    with open(path) as f:
        return json.load(f)


# ── OOF prediction collectors ─────────────────────────────────────────────────
# Each function runs LOSO-CV with a model's best params and returns
# (y_true, y_pred) arrays concatenated across all held-out folds.

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


# ── Main ensemble function ────────────────────────────────────────────────────

def ensemble_gender(gender: str):
    """
    Collect OOF predictions for all 4 models, combine, pick best strategy.
    Returns (best_brier, strategy).
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Ensemble -- {label}")
    print(f"{'='*52}")

    # Load data once (shared across all model OOF runs)
    features   = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    tourney    = utils.load_tourney(gender)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)
    all_diff   = [c for c in matchups.columns if c.endswith("_diff")]
    feat_cols  = utils.curate_features(all_diff)
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    # Collect OOF predictions from each model
    preds  = {}
    y_true = None
    for model_name in MODELS:
        params    = load_best_params(model_name, gender)
        collector = _OOF_COLLECTORS[model_name]
        print(f"  Collecting OOF: {model_name:<10}", end=" ", flush=True)
        yt, yp = collector(matchups, feat_cols, cv_seasons, params)
        preds[model_name] = yp
        if y_true is None:
            y_true = yt
        brier_indiv = utils.brier_score(yt, yp)
        print(f"Brier = {brier_indiv:.4f}")

    # Strategy 1: simple mean
    stack      = np.column_stack([preds[m] for m in MODELS])
    mean_pred  = stack.mean(axis=1)
    brier_mean = utils.brier_score(y_true, mean_pred)

    # Strategy 2: rank mean — each model's preds converted to percentile ranks
    rank_stack = np.column_stack(
        [rankdata(preds[m]) / len(preds[m]) for m in MODELS]
    )
    rank_pred  = rank_stack.mean(axis=1)
    brier_rank = utils.brier_score(y_true, rank_pred)

    print(f"\n  Combination strategies:")
    print(f"    mean       Brier = {brier_mean:.4f}")
    print(f"    rank_mean  Brier = {brier_rank:.4f}")

    if brier_mean <= brier_rank:
        best_pred, strategy, best_brier = mean_pred, "mean", brier_mean
    else:
        best_pred, strategy, best_brier = rank_pred, "rank_mean", brier_rank

    print(f"  -> Best: {strategy}  Brier = {best_brier:.4f}")

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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensemble_config = {}

    for gender in ["M", "W"]:
        best_brier, strategy = ensemble_gender(gender)
        ensemble_config[gender] = {
            "models":    MODELS,
            "strategy":  strategy,
            "cv_brier":  round(best_brier, 6),
        }
        utils.log_benchmark(
            "ensemble_v1", gender, best_brier,
            f"4-model {strategy} ensemble (lgbm+xgb+catboost+histgb) Optuna-tuned"
        )

    # Save ensemble config for predict script
    cfg_path = utils.RESULTS / "ensemble_config.json"
    with open(cfg_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    # Mirror to models/ so 12_predict.py has one obvious location
    utils.MODELS.mkdir(parents=True, exist_ok=True)
    with open(utils.MODELS / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "="*52)
    print("ENSEMBLE SUMMARY")
    print("="*52)
    for gender, cfg in ensemble_config.items():
        label = "Men's" if gender == "M" else "Women's"
        print(f"  {label:<10} strategy={cfg['strategy']:<10} Brier={cfg['cv_brier']:.4f}")
    print("Saved results/ensemble_config.json + results/oof_preds_{{M,W}}.npz")
    print("Benchmarks updated.")
