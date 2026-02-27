"""
02_feature_engineering.py — March Machine Learning Mania 2026
Computes per-(Season, TeamID) features from regular season data only.

Outputs:
  features/team_features_M.parquet
  features/team_features_W.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import utils

import numpy as np
import pandas as pd


# ── Section 1: Efficiency features from detailed regular season results ────────

def build_efficiency_features(gender: str) -> pd.DataFrame:
    """
    Reshape detailed results to team-perspective rows, then aggregate
    per (Season, TeamID) into efficiency/shooting/rebounding stats.
    """
    det = utils.load_detailed(gender)

    # ── Step 1: Reshape to two perspective rows per game ──────────────────────

    # Winner perspective
    win_rows = pd.DataFrame({
        "Season":  det["Season"],
        "TeamID":  det["WTeamID"],
        "own_pts": det["WScore"],
        "opp_pts": det["LScore"],
        "FGM":     det["WFGM"],
        "FGA":     det["WFGA"],
        "FGM3":    det["WFGM3"],
        "FGA3":    det["WFGA3"],
        "FTM":     det["WFTM"],
        "FTA":     det["WFTA"],
        "OR":      det["WOR"],
        "DR":      det["WDR"],
        "Ast":     det["WAst"],
        "TO":      det["WTO"],
        "Stl":     det["WStl"],
        "Blk":     det["WBlk"],
        "PF":      det["WPF"],
        "oFGM":    det["LFGM"],
        "oFGA":    det["LFGA"],
        "oFGM3":   det["LFGM3"],
        "oFGA3":   det["LFGA3"],
        "oFTM":    det["LFTM"],
        "oFTA":    det["LFTA"],
        "oOR":     det["LOR"],
        "oDR":     det["LDR"],
        "oAst":    det["LAst"],
        "oTO":     det["LTO"],
        "oStl":    det["LStl"],
        "oBlk":    det["LBlk"],
        "oPF":     det["LPF"],
        "won":     1,
    })

    # Loser perspective
    loss_rows = pd.DataFrame({
        "Season":  det["Season"],
        "TeamID":  det["LTeamID"],
        "own_pts": det["LScore"],
        "opp_pts": det["WScore"],
        "FGM":     det["LFGM"],
        "FGA":     det["LFGA"],
        "FGM3":    det["LFGM3"],
        "FGA3":    det["LFGA3"],
        "FTM":     det["LFTM"],
        "FTA":     det["LFTA"],
        "OR":      det["LOR"],
        "DR":      det["LDR"],
        "Ast":     det["LAst"],
        "TO":      det["LTO"],
        "Stl":     det["LStl"],
        "Blk":     det["LBlk"],
        "PF":      det["LPF"],
        "oFGM":    det["WFGM"],
        "oFGA":    det["WFGA"],
        "oFGM3":   det["WFGM3"],
        "oFGA3":   det["WFGA3"],
        "oFTM":    det["WFTM"],
        "oFTA":    det["WFTA"],
        "oOR":     det["WOR"],
        "oDR":     det["WDR"],
        "oAst":    det["WAst"],
        "oTO":     det["WTO"],
        "oStl":    det["WStl"],
        "oBlk":    det["WBlk"],
        "oPF":     det["WPF"],
        "won":     0,
    })

    games = pd.concat([win_rows, loss_rows], ignore_index=True)

    # ── Step 2: Aggregate per (Season, TeamID) ────────────────────────────────

    games["margin"] = games["own_pts"] - games["opp_pts"]

    grp = games.groupby(["Season", "TeamID"])

    # Sum columns for possession-based metrics
    sum_cols = [
        "own_pts", "opp_pts",
        "FGA", "OR", "TO", "FTA", "FGM", "FGM3",
        "oFGA", "oOR", "oTO", "oFTA",
        "DR", "oDR",
    ]
    sums = grp[sum_cols].sum().reset_index()

    # Simple per-game averages
    simple = grp.agg(
        games_played=("won", "count"),
        win_pct=("won", "mean"),
        avg_margin=("margin", "mean"),
        avg_pts_scored=("own_pts", "mean"),
        avg_pts_allowed=("opp_pts", "mean"),
    ).reset_index()

    # Merge sums into simple
    agg = simple.merge(sums, on=["Season", "TeamID"], how="left")

    # Possessions (Kubatko formula) — summed per season, not per game
    own_poss = agg["FGA"]  - agg["OR"]  + agg["TO"]  + 0.44 * agg["FTA"]
    opp_poss = agg["oFGA"] - agg["oOR"] + agg["oTO"] + 0.44 * agg["oFTA"]

    agg["off_eff"]  = agg["own_pts"] / own_poss * 100
    agg["def_eff"]  = agg["opp_pts"] / opp_poss * 100
    agg["net_eff"]  = agg["off_eff"] - agg["def_eff"]

    agg["efg_pct"]  = (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"]
    agg["oreb_pct"] = agg["OR"]  / (agg["OR"]  + agg["oDR"])
    agg["dreb_pct"] = agg["DR"]  / (agg["DR"]  + agg["oOR"])
    agg["to_pct"]   = agg["TO"]  / own_poss
    agg["ft_rate"]  = agg["FTA"] / agg["FGA"]

    # Keep only the final feature columns (drop raw sums)
    keep_cols = [
        "Season", "TeamID",
        "games_played", "win_pct", "avg_margin",
        "off_eff", "def_eff", "net_eff",
        "efg_pct", "oreb_pct", "dreb_pct", "to_pct", "ft_rate",
        "avg_pts_scored", "avg_pts_allowed",
    ]
    return agg[keep_cols].copy()


# ── Section 2: Massey composite rating ────────────────────────────────────────

TOP_SYSTEMS = ["MOR", "POM", "DOK", "SAG", "MAS", "WLK", "WIL", "PGH", "DOL", "COL"]
TOP5_SYSTEMS = ["MOR", "POM", "DOK", "SAG", "MAS"]   # individual rank columns


def build_massey_features() -> pd.DataFrame:
    """
    For each (Season, TeamID), compute:
      - massey_composite: mean ordinal rank across top-10 systems
        (last available ranking per system, RankingDayNum <= 133)
      - massey_MOR, massey_POM, massey_DOK, massey_SAG, massey_MAS
    Men's only (2003+).  Women's gets NaN via the left-join in Section 4.
    """
    massey = utils.load_massey()

    # Keep only pre-tournament rankings and the top 10 systems
    massey = massey[
        (massey["RankingDayNum"] <= 133) &
        (massey["SystemName"].isin(TOP_SYSTEMS))
    ].copy()

    # For each (Season, TeamID, SystemName) keep the last available day
    massey = massey.sort_values("RankingDayNum")
    last_rank = massey.groupby(["Season", "TeamID", "SystemName"])["OrdinalRank"].last()
    last_rank = last_rank.reset_index()

    # Pivot: one column per system
    pivot = last_rank.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Composite = mean across all available top-10 columns
    system_cols = [c for c in TOP_SYSTEMS if c in pivot.columns]
    pivot["massey_composite"] = pivot[system_cols].mean(axis=1)

    # Individual top-5 rank columns (rename with prefix)
    result = pivot[["Season", "TeamID", "massey_composite"]].copy()
    for sys in TOP5_SYSTEMS:
        col = f"massey_{sys}"
        if sys in pivot.columns:
            result[col] = pivot[sys].values
        else:
            result[col] = np.nan

    return result


# ── Section 3: Seed features ──────────────────────────────────────────────────

def build_seed_features(gender: str) -> pd.DataFrame:
    """
    Return DataFrame with Season, TeamID, SeedNum, IsFirstFour.
    Non-tournament teams are simply absent (will become NaN after left-join).
    """
    seeds = utils.load_seeds(gender)
    return seeds[["Season", "TeamID", "SeedNum", "IsFirstFour"]].copy()


# ── Section 4: Merge and save ─────────────────────────────────────────────────

def build_and_save(gender: str) -> pd.DataFrame:
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*60}")
    print(f"Building {label} features...")

    # 1. Efficiency features (base — all detailed seasons)
    eff = build_efficiency_features(gender)
    print(f"  Efficiency features: {eff.shape}")

    # 2. Massey (Men's only)
    if gender == "M":
        massey_feats = build_massey_features()
        print(f"  Massey features:     {massey_feats.shape}")
        df = eff.merge(massey_feats, on=["Season", "TeamID"], how="left")
    else:
        # Add NaN columns so both parquet files share the same schema
        df = eff.copy()
        df["massey_composite"] = np.nan
        for sys in TOP5_SYSTEMS:
            df[f"massey_{sys}"] = np.nan

    # 3. Seed features
    seed_feats = build_seed_features(gender)
    df = df.merge(seed_feats, on=["Season", "TeamID"], how="left")

    # 4. Verify uniqueness
    n_dupes = df.duplicated(subset=["Season", "TeamID"]).sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate (Season, TeamID) rows — dropping extras")
        df = df.drop_duplicates(subset=["Season", "TeamID"])

    df = df.sort_values(["Season", "TeamID"]).reset_index(drop=True)

    # 5. Save
    utils.FEATURES.mkdir(parents=True, exist_ok=True)
    out_path = utils.FEATURES / f"team_features_{gender}.parquet"
    df.to_parquet(out_path, index=False)

    # 6. Summary
    season_min = df["Season"].min()
    season_max = df["Season"].max()
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].to_dict()

    print(f"\n{label} features: {df.shape} -> features/team_features_{gender}.parquet")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Seasons: {season_min}-{season_max}, NaN counts: {nan_counts}")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_m = build_and_save("M")
    df_w = build_and_save("W")
    print("\nDone.")
