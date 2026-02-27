"""
07_analysis.py — March Machine Learning Mania 2026
Feature importance and prediction error analysis.

Produces 3 diagnostic figures:
  figures/06_feature_importance.png   — Top features per model (M + W, all 3 models)
  figures/07_feature_correlation.png  — Feature correlation heatmap (M + W)
  figures/08_prediction_diagnostics.png — Calibration + Brier by game difficulty
"""

import sys
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

sns.set_theme(style="whitegrid")
FIGURES = utils.FIGURES
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_matchups_feats(gender: str):
    """Load features + tourney results, build NaN-tolerant matchup df."""
    features = pd.read_parquet(utils.FEATURES / f"team_features_{gender}.parquet")
    tourney  = utils.load_tourney(gender)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchups = utils.make_matchup_df_nan_tolerant(tourney, features)
    diff_cols = [c for c in matchups.columns if c.endswith("_diff")]
    diff_cols = [c for c in diff_cols if c != "neutral_games_diff"]
    return matchups, diff_cols, tourney


def get_imp(model_name: str, model, feat_cols: list, matchups: pd.DataFrame) -> np.ndarray:
    """
    Extract normalized feature importances for any model type.
    CatBoost may have a narrower feature set (always-NaN columns were stripped
    during training), so we align its importances back to feat_cols.
    """
    n = len(feat_cols)

    if model_name == "lgbm":
        raw = model.booster_.feature_importance(importance_type="gain")[:n].astype(float)
    elif model_name == "xgb":
        raw = model.feature_importances_[:n].astype(float)
    else:  # catboost — re-align to full feat_cols
        cb_feats = [c for c in feat_cols if not matchups[c].isna().all()]
        cb_raw   = model.get_feature_importance().astype(float)
        raw = np.zeros(n)
        for i, col in enumerate(feat_cols):
            if col in cb_feats:
                j = cb_feats.index(col)
                if j < len(cb_raw):
                    raw[i] = cb_raw[j]

    s = raw.sum()
    return raw / s if s > 0 else raw


def bar_color(fname: str) -> str:
    if any(k in fname for k in ("seed", "massey", "sos")):
        return "#e74c3c"
    if any(k in fname for k in ("eff", "pct", "margin", "score")):
        return "#3498db"
    return "#95a5a6"


# ── Figure 6: Feature Importance ──────────────────────────────────────────────

def make_feature_importance():
    model_names  = ["lgbm",        "xgb",          "catboost"]
    model_titles = ["LightGBM\n(gain)", "XGBoost\n(weight)", "CatBoost\n(PredVal)"]
    genders      = ["M", "W"]
    row_labels   = ["Men's", "Women's"]

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle(
        "Feature Importance by Model — top 12 features, normalized to sum = 1",
        fontsize=14, fontweight="bold"
    )

    for row, (gender, rlabel) in enumerate(zip(genders, row_labels)):
        matchups, feat_cols, _ = get_matchups_feats(gender)

        lgbm_m = joblib.load(utils.MODELS / f"lgbm_{gender}.pkl")["model"]
        xgb_m  = joblib.load(utils.MODELS / f"xgb_{gender}.pkl")["model"]
        cb_m   = joblib.load(utils.MODELS / f"catboost_{gender}.pkl")["model"]
        models = [lgbm_m, xgb_m, cb_m]

        for col, (mname, mtitle, model) in enumerate(zip(model_names, model_titles, models)):
            ax  = axes[row, col]
            imp = get_imp(mname, model, feat_cols, matchups)

            top_idx   = np.argsort(imp)[::-1][:12]
            top_names = [feat_cols[i].replace("_diff", "") for i in top_idx]
            top_vals  = imp[top_idx]
            colors    = [bar_color(n) for n in top_names]

            # Plot reversed so most important is at the top
            ax.barh(
                range(len(top_names)),
                top_vals[::-1],
                color=colors[::-1],
                edgecolor="white",
                linewidth=0.4,
            )
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names[::-1], fontsize=8)
            ax.set_title(f"{rlabel} — {mtitle}", fontsize=10)
            ax.set_xlabel("Norm. importance", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)

    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Seed / Massey / SOS"),
        mpatches.Patch(color="#3498db", label="Efficiency / Margins / Win rates"),
        mpatches.Patch(color="#95a5a6", label="Location / Conference / Coach"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = FIGURES / "06_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 7: Feature Correlation ────────────────────────────────────────────

def make_feature_correlation():
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    for ax, gender in zip(axes, ["M", "W"]):
        label = "Men's" if gender == "M" else "Women's"
        matchups, feat_cols, _ = get_matchups_feats(gender)

        # Keep only columns that have at least some non-NaN data
        valid = [c for c in feat_cols if not matchups[c].isna().all()]
        corr  = matchups[valid].corr()  # pairwise Pearson (uses all available data per pair)
        labels = [c.replace("_diff", "") for c in valid]

        # Mask upper triangle (show lower only)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(
            corr, mask=mask, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.2, linecolor="white",
            cbar_kws={"label": "Pearson r", "shrink": 0.65},
        )
        ax.set_title(f"{label} — Feature Correlation Matrix", fontsize=12, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    out = FIGURES / "07_feature_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── OOF helper ────────────────────────────────────────────────────────────────

def collect_oof_predictions(gender: str) -> pd.DataFrame:
    """
    Re-run LGBM leave-one-season-out CV and return a DataFrame with columns:
    [Season, pred, label, SeedNum_diff].
    """
    matchups, feat_cols, tourney = get_matchups_feats(gender)
    cv_seasons = utils.get_cv_seasons(tourney, n_seasons=10)

    rows = []
    for season in cv_seasons:
        train_mask = matchups["Season"] != season
        test_mask  = matchups["Season"] == season
        if test_mask.sum() == 0:
            continue

        X_train = matchups.loc[train_mask, feat_cols]
        y_train = matchups.loc[train_mask, "Label"].values
        X_test  = matchups.loc[test_mask,  feat_cols]
        y_test  = matchups.loc[test_mask,  "Label"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp  = SimpleImputer(strategy="median")
            Xtr  = imp.fit_transform(X_train)
            Xte  = imp.transform(X_test)

        model = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, y_train)

        y_pred = model.predict_proba(Xte)[:, 1]

        test_rows = matchups.loc[test_mask].reset_index(drop=True)
        sd_vals   = (
            test_rows["SeedNum_diff"].values
            if "SeedNum_diff" in test_rows.columns
            else np.full(len(y_test), np.nan)
        )

        for pred, lab, sd in zip(y_pred, y_test, sd_vals):
            rows.append({"Season": season, "pred": pred, "label": lab, "SeedNum_diff": sd})

    return pd.DataFrame(rows)


# ── Figure 8: Prediction Diagnostics ──────────────────────────────────────────

def make_prediction_diagnostics():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "Prediction Diagnostics — LGBM OOF (10 LOSO-CV Seasons)\n"
        "vs seed-diff logistic regression baseline",
        fontsize=13, fontweight="bold"
    )

    for col_idx, gender in enumerate(["M", "W"]):
        label = "Men's" if gender == "M" else "Women's"
        print(f"  Collecting OOF predictions for {label}...")
        oof = collect_oof_predictions(gender)

        preds  = oof["pred"].values
        labels = oof["label"].values

        # Load the baseline model (seed-diff logistic regression)
        baseline = joblib.load(utils.MODELS / f"baseline_{gender}.pkl")

        # ── Calibration plot ─────────────────────────────────────────────────
        ax_cal = axes[0, col_idx]

        bin_edges = np.linspace(0, 1, 11)
        bcenters, actual_rates, sizes = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() >= 3:
                bcenters.append(preds[mask].mean())
                actual_rates.append(labels[mask].mean())
                sizes.append(mask.sum())

        bcenters     = np.array(bcenters)
        actual_rates = np.array(actual_rates)
        sizes        = np.array(sizes)

        ax_cal.plot([0, 1], [0, 1], "--", color="gray", lw=1.5,
                    label="Perfect calibration", zorder=1)
        ax_cal.scatter(bcenters, actual_rates,
                       s=sizes / sizes.max() * 260 + 20,
                       alpha=0.85, color="#3498db", zorder=3,
                       label="LGBM (bubble size ~ game count)")
        ax_cal.plot(bcenters, actual_rates, "-", color="#3498db", alpha=0.4, lw=1)

        brier = utils.brier_score(labels, preds)
        ax_cal.text(
            0.04, 0.93,
            f"OOF Brier = {brier:.4f}",
            transform=ax_cal.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        ax_cal.set_xlabel("Predicted Win Probability", fontsize=11)
        ax_cal.set_ylabel("Actual Win Rate", fontsize=11)
        ax_cal.set_title(f"{label} — Calibration (bubble size ~ game count)", fontsize=11)
        ax_cal.legend(fontsize=9)

        # ── Brier by seed-diff bucket ────────────────────────────────────────
        ax_err = axes[1, col_idx]

        oof_v = oof.dropna(subset=["SeedNum_diff"]).copy()
        oof_v["abs_sd"] = oof_v["SeedNum_diff"].abs()

        # Buckets represent "upset difficulty": how far apart are the seeds?
        bucket_edges  = [0.5, 3.5, 6.5, 9.5, 13.5, 15.5]
        bucket_labels = [
            "1–3\n(coin flip)",
            "4–6\n(slight fav)",
            "7–9\n(clear fav)",
            "10–13\n(big fav)",
            "14–15\n(huge fav)",
        ]

        model_briers, base_briers, counts = [], [], []
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            sub = oof_v[(oof_v["abs_sd"] >= lo) & (oof_v["abs_sd"] < hi)]
            if len(sub) > 0:
                mb = utils.brier_score(sub["label"].values, sub["pred"].values)
                # Baseline uses SeedNum_diff directly
                sd_input = sub["SeedNum_diff"].values.reshape(-1, 1)
                bp = baseline.predict_proba(sd_input)[:, 1]
                bb = utils.brier_score(sub["label"].values, bp)
            else:
                mb, bb = np.nan, np.nan
            model_briers.append(mb)
            base_briers.append(bb)
            counts.append(len(sub))

        x = np.arange(len(bucket_labels))
        w = 0.35

        bars_m = ax_err.bar(x - w / 2, model_briers, w, label="LightGBM",
                            color="#3498db", alpha=0.85, edgecolor="white")
        bars_b = ax_err.bar(x + w / 2, base_briers,  w, label="Seed-diff baseline",
                            color="#e74c3c", alpha=0.85, edgecolor="white")

        for bar, cnt in zip(bars_m, counts):
            if cnt > 0 and not np.isnan(bar.get_height()):
                ax_err.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"n={cnt}", ha="center", va="bottom", fontsize=7,
                )

        ax_err.set_xticks(x)
        ax_err.set_xticklabels(bucket_labels, fontsize=9)
        ax_err.set_xlabel("Seed Difference |Seed1 − Seed2|", fontsize=11)
        ax_err.set_ylabel("Brier Score (lower = better)", fontsize=11)
        ax_err.set_title(
            f"{label} — Brier Score by Game Difficulty\n"
            "(lower bar = model wins that bucket vs baseline)",
            fontsize=10,
        )
        ax_err.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES / "08_prediction_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Figure 06: Feature importance")
    print("=" * 50)
    make_feature_importance()

    print("=" * 50)
    print("Figure 07: Feature correlation")
    print("=" * 50)
    make_feature_correlation()

    print("=" * 50)
    print("Figure 08: Prediction diagnostics (OOF CV, ~2 min)")
    print("=" * 50)
    make_prediction_diagnostics()

    print("\nDone. All 3 figures saved to figures/")
