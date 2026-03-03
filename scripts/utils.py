from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).parent.parent
DATA = ROOT / "march-machine-learning-mania-2026"
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SUBMISSIONS = ROOT / "submissions"

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_compact(gender: str) -> pd.DataFrame:
    """Load regular season compact results. gender='M' or 'W'."""
    prefix = "M" if gender == "M" else "W"
    return pd.read_csv(DATA / f"{prefix}RegularSeasonCompactResults.csv")

def load_detailed(gender: str) -> pd.DataFrame:
    """Load regular season detailed (box score) results. gender='M' or 'W'."""
    prefix = "M" if gender == "M" else "W"
    return pd.read_csv(DATA / f"{prefix}RegularSeasonDetailedResults.csv")

def load_tourney(gender: str) -> pd.DataFrame:
    """Load NCAA tournament compact results. gender='M' or 'W'."""
    prefix = "M" if gender == "M" else "W"
    return pd.read_csv(DATA / f"{prefix}NCAATourneyCompactResults.csv")

def load_tourney_detailed(gender: str) -> pd.DataFrame:
    """Load NCAA tournament detailed results. gender='M' or 'W'."""
    prefix = "M" if gender == "M" else "W"
    return pd.read_csv(DATA / f"{prefix}NCAATourneyDetailedResults.csv")

def load_seeds(gender: str) -> pd.DataFrame:
    """
    Load tournament seeds. Returns DataFrame with columns:
    Season, Seed (original str e.g. 'W01a'), TeamID, SeedNum (int 1-16),
    IsFirstFour (bool — seeds ending in 'a' or 'b').
    """
    prefix = "M" if gender == "M" else "W"
    df = pd.read_csv(DATA / f"{prefix}NCAATourneySeeds.csv")
    # Extract numeric seed (e.g. 'W01' -> 1, 'W11a' -> 11)
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    df["IsFirstFour"] = df["Seed"].str[-1].isin(["a", "b"])
    return df

def load_massey() -> pd.DataFrame:
    """Load Massey ordinals (all systems, all seasons). ~5.7M rows."""
    return pd.read_csv(DATA / "MMasseyOrdinals.csv")

def load_sample_submission(stage: int = 2) -> pd.DataFrame:
    """Load sample submission file. stage=1 or 2."""
    return pd.read_csv(DATA / f"SampleSubmissionStage{stage}.csv")

# ── Evaluation ────────────────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score = MSE of probability predictions. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    return float(np.mean((y_pred - y_true) ** 2))

# ── Curated feature set ───────────────────────────────────────────────────────
# Derived from feature importance + correlation analysis (07_analysis.py).
# Rules applied:
#   - Keep massey_POM + massey_MOR (top 2 individual systems); drop SAG/DOK/MAS
#     (r > 0.9 with the top pair; composite already averages them)
#   - Keep net_eff (= off_eff - def_eff); drop off_eff + def_eff individually
#   - Keep avg_margin; drop avg_pts_scored + avg_pts_allowed (linear combo)
#   - Keep neutral_win_pct + neutral_net_eff (most tournament-relevant splits);
#     drop home_win_pct + away_win_pct (correlated; tournament has no home side)
#   - Drop IsFirstFour (sparse, <4 games per season), conf_tourney_champion
#     (sparse/low signal vs conf_tourney_wins)
#   - Keep coach_years_at_school (consistent signal across models);
#     drop is_new_coach (binary version already captured by years=1)
CURATED_FEATURES = {
    # Seed / ranking (19 → 5)
    "SeedNum", "massey_composite", "massey_POM", "massey_MOR", "sos_massey",
    # Efficiency — net only, plus four-factor components
    "net_eff", "efg_pct", "oreb_pct", "dreb_pct", "to_pct", "ft_rate",
    # Win rates & margin
    "win_pct", "avg_margin",
    # Neutral-court splits (most relevant for single-elimination tournament)
    "neutral_win_pct", "neutral_net_eff",
    # Conference / schedule quality
    "is_power_conf",
    # Momentum
    "conf_tourney_wins",
    # Coach continuity
    "coach_years_at_school",
    # Massey PCA (Men's only; Women's gets NaN)
    "massey_pc1", "massey_pc2",
    # Elo (Haupts-style, 7 features)
    "elo_last", "elo_mean", "elo_median", "elo_std", "elo_min", "elo_max", "elo_trend",
    # Recent form (last 10 regular season + conf tourney games)
    "recent_win_pct", "recent_net_margin", "streak",
    # Historical tournament performance (last 3 seasons)
    "tourney_win_pct_hist", "tourney_net_margin_hist", "tourney_rounds_advanced_avg",
    # Tournament box score efficiency (last 3 seasons)
    "tourney_off_eff_hist", "tourney_def_eff_hist",
}


def curate_features(feat_cols: list) -> list:
    """
    Filter a list of *_diff column names to the CURATED_FEATURES set.
    Preserves original ordering and silently skips unavailable features
    (e.g. Women's lacks Massey individual systems).
    """
    return [c for c in feat_cols if c.replace("_diff", "") in CURATED_FEATURES]


# ── Monotonic constraints for HistGB / sklearn histogram boosting ──────────────
# +1 = higher diff → higher win prob (quality metrics: bigger advantage = better)
# -1 = higher diff → lower win prob  (SeedNum: lower number = better team;
#      massey/sos ranks: lower rank number = better; to_pct: more turnovers = worse)
#  0 = no strong directional prior
_MONOTONE_MAP: dict = {
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
    # Elo (Haupts-style, 7 features)
    "elo_last_diff":            +1,   # final regular-season rating
    "elo_mean_diff":            +1,   # avg rating across season
    "elo_median_diff":          +1,   # robust to spikes
    "elo_std_diff":              0,   # consistency — direction ambiguous
    "elo_min_diff":             +1,   # worst point of season
    "elo_max_diff":             +1,   # peak rating
    "elo_trend_diff":           +1,   # improving trend = better
    # Massey PCA
    "massey_pc1_diff":          +1,   # higher consensus rank = better
    "massey_pc2_diff":           0,   # direction of disagreement is ambiguous
    # Momentum
    "recent_win_pct_diff":      +1,
    "recent_net_margin_diff":   +1,
    "streak_diff":              +1,   # win streak advantage = better
    # Historical tournament performance
    "tourney_win_pct_hist_diff":          +1,   # better tourney win rate = stronger team
    "tourney_net_margin_hist_diff":       +1,   # higher margin = dominant in tourney
    "tourney_rounds_advanced_avg_diff":   +1,   # deeper runs = better team
    # Tournament box score efficiency
    "tourney_off_eff_hist_diff":          +1,   # more pts/poss in tourney = better offence
    "tourney_def_eff_hist_diff":          -1,   # fewer opp pts/poss = better defence
}


def build_monotone_vec(feat_cols: list) -> list:
    """
    Return monotonic constraint list aligned to feat_cols.
    +1 = increasing, -1 = decreasing, 0 = unconstrained.
    Defaults to 0 for any column not in _MONOTONE_MAP.
    """
    return [_MONOTONE_MAP.get(c, 0) for c in feat_cols]


# ── Cross-validation ──────────────────────────────────────────────────────────

def get_cv_seasons(tourney_df: pd.DataFrame, n_seasons: int = 10) -> list:
    """
    Return the last n_seasons that have tournament data, sorted ascending.
    These are used as validation folds in leave-one-season-out CV.
    Excludes 2020 (tournament cancelled due to COVID).
    """
    seasons = sorted(tourney_df["Season"].unique())
    seasons = [s for s in seasons if s != 2020]
    return seasons[-n_seasons:]

def _build_diff_result(tourney_df: pd.DataFrame, features_df: pd.DataFrame):
    """Shared inner logic: build Team1/Team2 diffs without any row-dropping."""
    feat_cols = [c for c in features_df.columns if c not in ("Season", "TeamID")]

    df = tourney_df.copy()
    df["Team1ID"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["Team2ID"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["Label"]   = (df["WTeamID"] == df["Team1ID"]).astype(int)

    feat = features_df.set_index(["Season", "TeamID"])
    t1 = (df[["Season", "Team1ID"]]
          .rename(columns={"Team1ID": "TeamID"})
          .join(feat, on=["Season", "TeamID"])
          .drop(columns="TeamID"))
    t2 = (df[["Season", "Team2ID"]]
          .rename(columns={"Team2ID": "TeamID"})
          .join(feat, on=["Season", "TeamID"])
          .drop(columns="TeamID"))

    result = df[["Season", "Team1ID", "Team2ID", "Label"]].copy()
    for c in feat_cols:
        result[f"{c}_diff"] = t1[c].values - t2[c].values

    return result, feat_cols


def make_matchup_df(tourney_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matchup-level DataFrame for model training.
    Team1 = lower TeamID, Team2 = higher TeamID.
    Feature columns are diffs: team1_feat - team2_feat.
    Label = 1 if Team1 (lower ID) won, 0 if Team2 won.

    Rows where ANY feature diff is NaN are dropped (strict version).
    Use make_matchup_df_nan_tolerant for models that handle NaN natively
    or apply imputation after matchup construction.
    """
    result, feat_cols = _build_diff_result(tourney_df, features_df)

    n_before = len(result)
    result = result.dropna(subset=[f"{c}_diff" for c in feat_cols])
    skipped = n_before - len(result)
    if skipped:
        import warnings
        warnings.warn(
            f"make_matchup_df: dropped {skipped} games due to missing features",
            stacklevel=2,
        )

    return result.reset_index(drop=True)


def make_matchup_df_nan_tolerant(
    tourney_df: pd.DataFrame, features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    NaN-tolerant variant of make_matchup_df.

    Rows are only dropped when ALL feature diffs are NaN (both teams entirely
    absent from the feature set — no information whatsoever).  Rows with
    *partial* NaN (e.g. a ranking system absent for recent seasons) are kept
    so that a downstream SimpleImputer can fill them without losing games.

    Use this for sklearn-style models (LGBM, XGBoost) where imputation happens
    after matchup construction.
    """
    result, feat_cols = _build_diff_result(tourney_df, features_df)

    diff_cols = [f"{c}_diff" for c in feat_cols]
    all_nan_mask = result[diff_cols].isna().all(axis=1)
    n_dropped = int(all_nan_mask.sum())
    if n_dropped:
        import warnings
        warnings.warn(
            f"make_matchup_df_nan_tolerant: dropped {n_dropped} games where all "
            "feature diffs were NaN (both teams absent from feature set)",
            stacklevel=2,
        )

    return result[~all_nan_mask].reset_index(drop=True)

# ── Benchmark logging ─────────────────────────────────────────────────────────

def log_benchmark(model: str, gender: str, cv_brier: float, notes: str = "") -> None:
    """
    Append a row to results/benchmarks.csv and regenerate BENCHMARKS.md.
    Thread-safe only for sequential calls (no file locking).
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    benchmarks_csv = RESULTS / "benchmarks.csv"
    benchmarks_md = ROOT / "BENCHMARKS.md"

    # Append to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = pd.DataFrame([{
        "model": model,
        "gender": gender,
        "cv_brier": round(cv_brier, 6),
        "notes": notes,
        "timestamp": timestamp,
    }])

    if benchmarks_csv.exists():
        existing = pd.read_csv(benchmarks_csv)
    else:
        existing = pd.DataFrame(columns=["model", "gender", "cv_brier", "notes", "timestamp"])
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated = updated.drop_duplicates(subset=["model", "gender"], keep="last")
    updated = updated.sort_values(["gender", "model"]).reset_index(drop=True)
    updated.to_csv(benchmarks_csv, index=False)

    # Regenerate BENCHMARKS.md
    lines = [
        "# Benchmarks\n",
        "\nCV Brier scores for all models (lower is better). Auto-updated by training scripts.\n",
        "\n*Source data: `results/benchmarks.csv`*\n\n",
        "| Model | Gender | CV Brier | Notes | Timestamp |\n",
        "|-------|--------|----------|-------|-----------|\n",
    ]
    for _, row in updated.iterrows():
        lines.append(
            f"| {row['model']} | {row['gender']} | {row['cv_brier']:.6f} | {row['notes']} | {row['timestamp']} |\n"
        )

    with open(benchmarks_md, "w") as f:
        f.writelines(lines)
