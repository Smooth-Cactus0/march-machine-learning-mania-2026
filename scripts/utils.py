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

def make_matchup_df(tourney_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matchup-level DataFrame for model training.

    For each tournament game, creates one row with:
    - Team1 = lower TeamID, Team2 = higher TeamID
    - All feature columns as diffs: feat_diff = team1_feat - team2_feat
    - Label = 1 if Team1 (lower ID) won, 0 if Team2 won
    - Season column retained

    Args:
        tourney_df: output of load_tourney()
        features_df: output of feature engineering, indexed by (Season, TeamID)

    Returns:
        DataFrame with columns: Season, Team1ID, Team2ID, *feat_diff_cols, Label
    """
    feat_cols = [c for c in features_df.columns if c not in ("Season", "TeamID")]
    feat_indexed = features_df.set_index(["Season", "TeamID"])

    rows = []
    skipped = 0
    for _, game in tourney_df.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]
        t1_id = min(w_id, l_id)
        t2_id = max(w_id, l_id)
        label = 1 if w_id == t1_id else 0

        if (season, t1_id) not in feat_indexed.index or (season, t2_id) not in feat_indexed.index:
            skipped += 1
            continue

        t1_feats = feat_indexed.loc[(season, t1_id), feat_cols]
        t2_feats = feat_indexed.loc[(season, t2_id), feat_cols]
        diffs = (t1_feats - t2_feats).values
        rows.append([season, t1_id, t2_id] + list(diffs) + [label])

    if skipped:
        import warnings
        warnings.warn(
            f"make_matchup_df: skipped {skipped} games due to missing features",
            stacklevel=2,
        )
    diff_cols = [f"{c}_diff" for c in feat_cols]
    return pd.DataFrame(rows, columns=["Season", "Team1ID", "Team2ID"] + diff_cols + ["Label"])

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
