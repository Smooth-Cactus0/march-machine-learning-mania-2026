"""
09_tune_optuna.py -- March Machine Learning Mania 2026

Optuna hyperparameter search for LGBM, XGB, CatBoost, HistGB.
50 trials per (model, gender); LOSO-CV Brier is the objective.

Outputs:
  results/best_params_{model}_{gender}.json   -- best hyperparameters
  models/{model}_tuned_{gender}.pkl           -- retrained final models
  results/benchmarks.csv / BENCHMARKS.md     -- v3 benchmarks
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


# ── Shared data loader ────────────────────────────────────────────────────────

def load_data(gender: str):
    """Return (matchups, feat_cols, cv_seasons) for one gender.

    Called once per study — data is shared across all trials to avoid
    repeated disk I/O.
    """
    features   = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    tourney    = utils.load_tourney(gender)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)
    all_diff   = [c for c in matchups.columns if c.endswith("_diff")]
    feat_cols  = utils.curate_features(all_diff)
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)
    return matchups, feat_cols, cv_seasons


# ── Per-model CV runners (called inside each Optuna trial) ───────────────────

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


# ── Optuna objective factories ─────────────────────────────────────────────────

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


# ── Orchestrator ───────────────────────────────────────────────────────────────

_OBJECTIVE_FACTORIES = {
    "lgbm":     make_lgbm_objective,
    "xgb":      make_xgb_objective,
    "catboost": make_catboost_objective,
    "histgb":   make_histgb_objective,
}


def tune_model(model_name: str, gender: str) -> float:
    """
    Run Optuna study, save best params JSON, retrain final model.
    Returns best CV Brier.
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*52}")
    print(f"Tuning {model_name.upper()} -- {label}  ({N_TRIALS} trials)")
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


def _retrain_final(model_name, gender, matchups, feat_cols, params):
    """Retrain with best params on all available data and save to models/."""
    X_all = matchups[feat_cols].values
    y_all = matchups["Label"].values

    if model_name == "lgbm":
        imp = SimpleImputer(strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_imp = imp.fit_transform(matchups[feat_cols])
        m = LGBMClassifier(**params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_imp, y_all)
        artifact = {"model": m, "imputer": imp, "feature_cols": feat_cols}

    elif model_name == "xgb":
        imp = SimpleImputer(strategy="median")
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna tuning for all models")
    parser.add_argument("--model",  choices=["lgbm", "xgb", "catboost", "histgb", "all"],
                        default="all")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both")
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

    print("\n" + "=" * 52)
    print("TUNING SUMMARY (v3 benchmarks)")
    print("=" * 52)
    for k, v in summary.items():
        print(f"  {k:<20} {v:.4f}")
    print("Benchmarks updated in results/benchmarks.csv and BENCHMARKS.md")
