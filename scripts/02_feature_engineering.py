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

    own_poss = own_poss.replace(0, np.nan)
    opp_poss = opp_poss.replace(0, np.nan)
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


# ── Section 5: Location features (home/neutral/away splits) ───────────────────

def build_location_features(gender: str) -> pd.DataFrame:
    """
    Compute per-(Season, TeamID) location-split features:
      - neutral_off_eff, neutral_def_eff, neutral_net_eff: efficiency at neutral sites
      - neutral_games: count of neutral site games (context only, not a model feature)
      - home_win_pct: win% in home games (NaN if 0 home games)
      - away_win_pct: win% in away games (NaN if 0 away games)
      - neutral_win_pct: win% in neutral games (NaN if 0 neutral games)
    """
    det = utils.load_detailed(gender)
    comp = utils.load_compact(gender)

    # ── Part A: Neutral-site efficiency from detailed results ─────────────────

    det_neutral = det[det["WLoc"] == "N"].copy()

    # Winner perspective at neutral sites
    win_neutral = pd.DataFrame({
        "Season":  det_neutral["Season"],
        "TeamID":  det_neutral["WTeamID"],
        "own_pts": det_neutral["WScore"],
        "opp_pts": det_neutral["LScore"],
        "FGA":     det_neutral["WFGA"],
        "OR":      det_neutral["WOR"],
        "TO":      det_neutral["WTO"],
        "FTA":     det_neutral["WFTA"],
        "oFGA":    det_neutral["LFGA"],
        "oOR":     det_neutral["LOR"],
        "oTO":     det_neutral["LTO"],
        "oFTA":    det_neutral["LFTA"],
    })

    # Loser perspective at neutral sites
    loss_neutral = pd.DataFrame({
        "Season":  det_neutral["Season"],
        "TeamID":  det_neutral["LTeamID"],
        "own_pts": det_neutral["LScore"],
        "opp_pts": det_neutral["WScore"],
        "FGA":     det_neutral["LFGA"],
        "OR":      det_neutral["LOR"],
        "TO":      det_neutral["LTO"],
        "FTA":     det_neutral["LFTA"],
        "oFGA":    det_neutral["WFGA"],
        "oOR":     det_neutral["WOR"],
        "oTO":     det_neutral["WTO"],
        "oFTA":    det_neutral["WFTA"],
    })

    neutral_games_det = pd.concat([win_neutral, loss_neutral], ignore_index=True)

    sum_cols = ["own_pts", "opp_pts", "FGA", "OR", "TO", "FTA", "oFGA", "oOR", "oTO", "oFTA"]
    neutral_agg = neutral_games_det.groupby(["Season", "TeamID"])[sum_cols].sum().reset_index()

    own_poss = neutral_agg["FGA"]  - neutral_agg["OR"]  + neutral_agg["TO"]  + 0.44 * neutral_agg["FTA"]
    opp_poss = neutral_agg["oFGA"] - neutral_agg["oOR"] + neutral_agg["oTO"] + 0.44 * neutral_agg["oFTA"]
    own_poss = own_poss.replace(0, np.nan)
    opp_poss = opp_poss.replace(0, np.nan)

    neutral_agg["neutral_off_eff"] = neutral_agg["own_pts"] / own_poss * 100
    neutral_agg["neutral_def_eff"] = neutral_agg["opp_pts"] / opp_poss * 100
    neutral_agg["neutral_net_eff"] = neutral_agg["neutral_off_eff"] - neutral_agg["neutral_def_eff"]

    # Count neutral games per team (each game appears twice in concat, so count games here)
    neutral_game_counts = (
        neutral_games_det.groupby(["Season", "TeamID"])
        .size()
        .reset_index(name="neutral_games")
    )

    neutral_eff = neutral_agg[["Season", "TeamID", "neutral_off_eff", "neutral_def_eff", "neutral_net_eff"]].merge(
        neutral_game_counts, on=["Season", "TeamID"], how="left"
    )

    # ── Part B: Win% by location from compact results ─────────────────────────

    # Winner perspective: determine this team's location
    # WLoc = 'H' means WTeamID was at home, 'A' means WTeamID was away, 'N' neutral
    win_comp = pd.DataFrame({
        "Season": comp["Season"],
        "TeamID": comp["WTeamID"],
        "won":    1,
        # From winner's perspective: WLoc directly describes their location
        "loc":    comp["WLoc"],
    })

    # Loser perspective: invert H/A for the losing team; N stays N
    loc_map = {"H": "A", "A": "H", "N": "N"}
    loss_comp = pd.DataFrame({
        "Season": comp["Season"],
        "TeamID": comp["LTeamID"],
        "won":    0,
        "loc":    comp["WLoc"].map(loc_map),
    })

    all_comp = pd.concat([win_comp, loss_comp], ignore_index=True)

    # Aggregate per (Season, TeamID, loc): games and wins
    loc_grp = all_comp.groupby(["Season", "TeamID", "loc"]).agg(
        games=("won", "count"),
        wins=("won", "sum"),
    ).reset_index()

    # Pivot so we have separate columns for H, A, N
    loc_pivot = loc_grp.pivot_table(
        index=["Season", "TeamID"],
        columns="loc",
        values=["games", "wins"],
        aggfunc="first",
    ).reset_index()
    loc_pivot.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in loc_pivot.columns
    ]

    # Compute win percentages — NaN if no games at that location
    for loc_code, col_suffix in [("H", "home"), ("A", "away"), ("N", "neutral")]:
        games_col = f"games_{loc_code}"
        wins_col  = f"wins_{loc_code}"
        pct_col   = f"{col_suffix}_win_pct"
        if games_col in loc_pivot.columns and wins_col in loc_pivot.columns:
            loc_pivot[pct_col] = np.where(
                loc_pivot[games_col] > 0,
                loc_pivot[wins_col] / loc_pivot[games_col],
                np.nan,
            )
        else:
            loc_pivot[pct_col] = np.nan

    win_pct_cols = ["Season", "TeamID", "home_win_pct", "away_win_pct", "neutral_win_pct"]
    win_pcts = loc_pivot[[c for c in win_pct_cols if c in loc_pivot.columns]].copy()

    # ── Part C: Merge neutral efficiency + win pcts ───────────────────────────

    # Start with all unique (Season, TeamID) from compact results (broader coverage)
    all_teams = all_comp[["Season", "TeamID"]].drop_duplicates()
    result = all_teams.merge(neutral_eff, on=["Season", "TeamID"], how="left")
    result = result.merge(win_pcts, on=["Season", "TeamID"], how="left")

    return result[["Season", "TeamID",
                   "neutral_off_eff", "neutral_def_eff", "neutral_net_eff",
                   "neutral_games",
                   "home_win_pct", "away_win_pct", "neutral_win_pct"]].copy()


# ── Section 6: Conference quality and strength of schedule ────────────────────

POWER_CONFS = {
    'sec', 'acc', 'big_ten', 'big_east', 'big_twelve',
    'pac_ten', 'pac_twelve',
}


def build_conference_features(gender: str, massey_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute per-(Season, TeamID):
      - is_power_conf: 1 if team's conference is a power conf, 0 otherwise
      - sos_massey: mean massey_composite of all regular season opponents
        (Men's 2003+ only; Women's and pre-2003 Men's get NaN)
    """
    prefix = "M" if gender == "M" else "W"
    tc = pd.read_csv(utils.DATA / f"{prefix}TeamConferences.csv")

    # is_power_conf
    tc["is_power_conf"] = tc["ConfAbbrev"].isin(POWER_CONFS).astype(int)
    conf_feat = tc[["Season", "TeamID", "is_power_conf"]].copy()

    # sos_massey: for each team-season, find opponents and average their massey_composite
    if massey_df is not None and gender == "M":
        comp = utils.load_compact(gender)

        # Build opponent lookup: for each game, each team's opponent
        win_side = comp[["Season", "WTeamID", "LTeamID"]].rename(
            columns={"WTeamID": "TeamID", "LTeamID": "OppID"}
        )
        loss_side = comp[["Season", "LTeamID", "WTeamID"]].rename(
            columns={"LTeamID": "TeamID", "WTeamID": "OppID"}
        )
        matchups = pd.concat([win_side, loss_side], ignore_index=True)

        # Join massey_composite for each opponent
        massey_lookup = massey_df[["Season", "TeamID", "massey_composite"]].rename(
            columns={"TeamID": "OppID", "massey_composite": "opp_massey"}
        )
        matchups = matchups.merge(massey_lookup, on=["Season", "OppID"], how="left")

        # Average opponent massey per (Season, TeamID)
        sos = (
            matchups.groupby(["Season", "TeamID"])["opp_massey"]
            .mean()
            .reset_index(name="sos_massey")
        )
        conf_feat = conf_feat.merge(sos, on=["Season", "TeamID"], how="left")
    else:
        conf_feat["sos_massey"] = np.nan

    return conf_feat


# ── Section 7: Conference tournament performance ──────────────────────────────

def build_conf_tourney_features(gender: str) -> pd.DataFrame:
    """
    Compute per-(Season, TeamID):
      - conf_tourney_wins: number of conference tournament wins that season
      - conf_tourney_champion: 1 if team won the conference tournament, else 0
    Teams that didn't participate will be absent (becomes 0 after fillna in merge).
    """
    prefix = "M" if gender == "M" else "W"
    ct_path = utils.DATA / f"{prefix}ConferenceTourneyGames.csv"

    if not ct_path.exists():
        # Return empty structure if file missing
        return pd.DataFrame(columns=["Season", "TeamID", "conf_tourney_wins", "conf_tourney_champion"])

    ct = pd.read_csv(ct_path)

    # conf_tourney_wins: count wins per (Season, TeamID)
    wins = (
        ct.groupby(["Season", "WTeamID"])
        .size()
        .reset_index(name="conf_tourney_wins")
        .rename(columns={"WTeamID": "TeamID"})
    )

    # conf_tourney_champion: winner of the last game (highest DayNum) per (Season, ConfAbbrev)
    last_game_idx = ct.groupby(["Season", "ConfAbbrev"])["DayNum"].idxmax()
    finals = ct.loc[last_game_idx, ["Season", "WTeamID"]].copy()
    finals = finals.rename(columns={"WTeamID": "TeamID"})
    finals["conf_tourney_champion"] = 1
    # Deduplicate in case of ties on DayNum (rare edge case)
    finals = finals.drop_duplicates(subset=["Season", "TeamID"])

    # Merge wins and champion flag
    result = wins.merge(finals, on=["Season", "TeamID"], how="left")
    result["conf_tourney_wins"] = result["conf_tourney_wins"].fillna(0).astype(int)
    result["conf_tourney_champion"] = result["conf_tourney_champion"].fillna(0).astype(int)

    return result[["Season", "TeamID", "conf_tourney_wins", "conf_tourney_champion"]].copy()


# ── Section 9: Elo ratings with margin-of-victory + season carryover ──────────

def build_elo_features(gender: str) -> pd.DataFrame:
    """
    Compute end-of-season Elo ratings per (Season, TeamID).

    Formula:
      K_eff = 20 * (1 + margin/20)^0.6    (margin-of-victory scaling)
      expected_A = 1 / (1 + 10^((elo_B - elo_A) / 400))
      delta = K_eff * (outcome - expected_A)

    Season carryover (mean reversion):
      elo_start = prev_elo * 0.75 + 1500 * 0.25
    New teams start at 1500.

    Returns DataFrame with: Season, TeamID, elo_rating, elo_k_weighted_wins
    """
    comp    = utils.load_compact(gender)
    seasons = sorted(comp["Season"].unique())

    elo: dict  = {}   # {TeamID: current_elo}  — persists across seasons
    records    = []

    for season in seasons:
        season_games = comp[comp["Season"] == season].sort_values("DayNum")
        teams        = set(season_games["WTeamID"]) | set(season_games["LTeamID"])

        # Carryover / initialise
        for tid in teams:
            elo[tid] = elo[tid] * 0.75 + 1500 * 0.25 if tid in elo else 1500.0

        k_wins: dict = {tid: 0.0 for tid in teams}

        for row in season_games.itertuples(index=False):
            w      = int(row.WTeamID)
            l      = int(row.LTeamID)
            margin = max(float(row.WScore - row.LScore), 1.0)
            ew, el = elo.get(w, 1500.0), elo.get(l, 1500.0)
            exp_w  = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
            k_eff  = 20.0 * (1.0 + margin / 20.0) ** 0.6
            delta  = k_eff * (1.0 - exp_w)
            elo[w] += delta
            elo[l] -= delta
            k_wins[w] += k_eff

        for tid in teams:
            records.append({
                "Season":              season,
                "TeamID":              tid,
                "elo_rating":          elo[tid],
                "elo_k_weighted_wins": k_wins[tid],
            })

    result = pd.DataFrame(records)
    result["elo_rating"]          = result["elo_rating"].astype(float)
    result["elo_k_weighted_wins"] = result["elo_k_weighted_wins"].astype(float)
    return result


# ── Section 10: Massey PCA — all systems with >= 50% coverage ─────────────────

def build_massey_pca_features() -> pd.DataFrame:
    """
    PCA of all Massey ordinal systems with >= 50% team coverage.

    Steps per season:
      1. Keep last pre-tournament rank per (Season, TeamID, SystemName)
      2. Pivot to (TeamID × SystemName) matrix
      3. Keep systems with >= 50% team coverage
      4. Flip sign: lower rank = better → flip so higher score = better
      5. StandardScaler, fill NaN with 0 (= mean in z-score space)
      6. PCA → massey_pc1 (consensus), massey_pc2 (disagreement)

    Men's only — Women's has no Massey data and will receive NaN via left-join.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    massey = utils.load_massey()
    massey = massey[massey["RankingDayNum"] <= 133].copy()

    # Last available rank per (Season, TeamID, SystemName)
    massey = massey.sort_values("RankingDayNum")
    last   = (massey
              .groupby(["Season", "TeamID", "SystemName"])["OrdinalRank"]
              .last()
              .reset_index())

    records = []

    for season in sorted(last["Season"].unique()):
        sdf   = last[last["Season"] == season]
        pivot = sdf.pivot_table(
            index="TeamID", columns="SystemName",
            values="OrdinalRank", aggfunc="first",
        )

        # Keep systems with >= 50% team coverage
        n_teams  = len(pivot)
        coverage = pivot.notna().sum() / n_teams
        keep     = coverage[coverage >= 0.5].index.tolist()

        if len(keep) < 2:
            # Not enough systems for PCA — return NaN for this season
            for tid in pivot.index:
                records.append({"Season": season, "TeamID": int(tid),
                                 "massey_pc1": np.nan, "massey_pc2": np.nan})
            continue

        pivot = pivot[keep].copy()

        # Flip: lower rank number = better → higher value = better
        max_rank = float(pivot.max().max())
        pivot    = max_rank - pivot + 1.0

        # Standardise — fill NaN with column mean before scaling
        pivot_filled = pivot.fillna(pivot.mean())
        mat = StandardScaler().fit_transform(pivot_filled)

        n_comp = min(2, mat.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        pcs    = pca.fit_transform(mat)

        # Align PC1 sign: better teams (lower original rank = higher flipped score)
        # should have higher PC1.  Check: if the median rank of the top-quartile
        # teams (by flipped score) has *negative* PC1, flip the sign.
        top_q_idx = np.where(pivot.mean(axis=1).values > np.percentile(pivot.mean(axis=1), 75))[0]
        if len(top_q_idx) > 0 and pcs[top_q_idx, 0].mean() < 0:
            pcs[:, 0] = -pcs[:, 0]

        var_explained = pca.explained_variance_ratio_
        if season == sorted(last["Season"].unique())[-1]:
            print(f"  Season {season}: {len(keep)} systems, "
                  f"PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")

        for i, tid in enumerate(pivot.index):
            records.append({
                "Season":     season,
                "TeamID":     int(tid),
                "massey_pc1": float(pcs[i, 0]),
                "massey_pc2": float(pcs[i, 1]) if n_comp >= 2 else np.nan,
            })

    return pd.DataFrame(records)


# ── Section 11: Late-season momentum (last N games) ───────────────────────────

def build_momentum_features(gender: str, n_games: int = 10) -> pd.DataFrame:
    """
    Compute momentum features from the last n_games regular season games.

    Features per (Season, TeamID):
      - recent_win_pct:     win% in last n_games
      - recent_net_margin:  mean point margin in last n_games
      - streak:             current streak length (positive=wins, negative=losses)
                            e.g. +5 = 5-game win streak, -2 = 2-game losing streak
    """
    comp = utils.load_compact(gender)

    # Build long format: one row per (Season, TeamID, game)
    win_rows = comp[["Season", "DayNum", "WTeamID", "WScore", "LScore"]].copy()
    win_rows["TeamID"] = win_rows["WTeamID"]
    win_rows["won"]    = 1
    win_rows["margin"] = win_rows["WScore"] - win_rows["LScore"]

    loss_rows = comp[["Season", "DayNum", "LTeamID", "WScore", "LScore"]].copy()
    loss_rows["TeamID"] = loss_rows["LTeamID"]
    loss_rows["won"]    = 0
    loss_rows["margin"] = loss_rows["LScore"] - loss_rows["WScore"]

    games = pd.concat([
        win_rows[["Season", "DayNum", "TeamID", "won", "margin"]],
        loss_rows[["Season", "DayNum", "TeamID", "won", "margin"]],
    ], ignore_index=True).sort_values(["Season", "TeamID", "DayNum"])

    # Keep only the last n_games per (Season, TeamID)
    last_n = games.groupby(["Season", "TeamID"]).tail(n_games)

    # Aggregate recent_win_pct and recent_net_margin
    agg = last_n.groupby(["Season", "TeamID"]).agg(
        recent_win_pct    =("won",    "mean"),
        recent_net_margin =("margin", "mean"),
    ).reset_index()

    # Streak: count consecutive same outcomes from the last game backwards
    def compute_streak(grp: pd.DataFrame) -> int:
        outcomes = grp.sort_values("DayNum")["won"].values
        if len(outcomes) == 0:
            return 0
        last_outcome = int(outcomes[-1])
        count = 0
        for o in reversed(outcomes):
            if int(o) == last_outcome:
                count += 1
            else:
                break
        return count if last_outcome == 1 else -count

    streak_series = (
        games
        .groupby(["Season", "TeamID"])
        .apply(compute_streak)
        .reset_index(name="streak")
    )

    result = agg.merge(streak_series, on=["Season", "TeamID"], how="left")
    result["streak"] = result["streak"].fillna(0).astype(int)
    return result


# ── Section 8: Coach continuity ───────────────────────────────────────────────

def build_coach_features() -> pd.DataFrame:
    """
    Compute per-(Season, TeamID) coach continuity features (Men's only):
      - coach_years_at_school: seasons the primary coach has been at this school
        (1 = first year)
      - is_new_coach: 1 if coach_years_at_school == 1, else 0

    Primary coach = the coach whose tenure spans the most of the season
    (highest LastDayNum - FirstDayNum).
    """
    coaches = pd.read_csv(utils.DATA / "MTeamCoaches.csv")

    # For each (Season, TeamID), pick the coach with the longest tenure span
    coaches["tenure_span"] = coaches["LastDayNum"] - coaches["FirstDayNum"]
    primary = (
        coaches
        .sort_values("tenure_span", ascending=False)
        .drop_duplicates(subset=["Season", "TeamID"], keep="first")
        [["Season", "TeamID", "CoachName"]]
        .copy()
    )

    # For each (CoachName, TeamID), compute cumulative seasons at that school.
    # Sort by Season and assign rank within each (CoachName, TeamID) group.
    primary = primary.sort_values(["CoachName", "TeamID", "Season"])
    primary["coach_years_at_school"] = (
        primary.groupby(["CoachName", "TeamID"]).cumcount() + 1
    )
    primary["is_new_coach"] = (primary["coach_years_at_school"] == 1).astype(int)

    return primary[["Season", "TeamID", "coach_years_at_school", "is_new_coach"]].copy()


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

    # Fix IsFirstFour dtype: True→1, False→0, NaN→<NA> (nullable Int8 for parquet)
    df["IsFirstFour"] = df["IsFirstFour"].astype("boolean").astype("Int8")

    # 4. Location features (home/neutral/away splits)
    location_feats = build_location_features(gender)
    print(f"  Location features:   {location_feats.shape}")
    df = df.merge(location_feats, on=["Season", "TeamID"], how="left")

    # 5. Conference quality + SOS
    conf_feats = build_conference_features(
        gender,
        massey_df=massey_feats if gender == "M" else None
    )
    print(f"  Conference features: {conf_feats.shape}")
    df = df.merge(conf_feats, on=["Season", "TeamID"], how="left")

    # Women's: add NaN sos_massey column if not present
    if "sos_massey" not in df.columns:
        df["sos_massey"] = np.nan

    # 6. Conference tournament features
    conf_tourney_feats = build_conf_tourney_features(gender)
    print(f"  Conf tourney features: {conf_tourney_feats.shape}")
    df = df.merge(conf_tourney_feats, on=["Season", "TeamID"], how="left")
    # Teams that didn't play in conf tourney get 0 wins and 0 champion
    df["conf_tourney_wins"]     = df["conf_tourney_wins"].fillna(0).astype(int)
    df["conf_tourney_champion"] = df["conf_tourney_champion"].fillna(0).astype(int)

    # 7. Coach features (Men's only)
    if gender == "M":
        coach_feats = build_coach_features()
        print(f"  Coach features:      {coach_feats.shape}")
        df = df.merge(coach_feats, on=["Season", "TeamID"], how="left")
    else:
        df["coach_years_at_school"] = np.nan
        df["is_new_coach"]          = np.nan

    # 8. Elo features
    elo_feats = build_elo_features(gender)
    print(f"  Elo features:        {elo_feats.shape}")
    df = df.merge(elo_feats, on=["Season", "TeamID"], how="left")

    # 9. Massey PCA features (Men's only; Women's gets NaN via left-join)
    if gender == "M":
        massey_pca_feats = build_massey_pca_features()
        print(f"  Massey PCA features: {massey_pca_feats.shape}")
        df = df.merge(massey_pca_feats, on=["Season", "TeamID"], how="left")
    else:
        df["massey_pc1"] = np.nan
        df["massey_pc2"] = np.nan

    # 10. Momentum features (last 10 regular season games)
    momentum_feats = build_momentum_features(gender)
    print(f"  Momentum features:   {momentum_feats.shape}")
    df = df.merge(momentum_feats, on=["Season", "TeamID"], how="left")

    # 11. Verify uniqueness
    n_dupes = df.duplicated(subset=["Season", "TeamID"]).sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate (Season, TeamID) rows — dropping extras")
        df = df.drop_duplicates(subset=["Season", "TeamID"])

    df = df.sort_values(["Season", "TeamID"]).reset_index(drop=True)

    # 11. Save
    utils.FEATURES.mkdir(parents=True, exist_ok=True)
    out_path = utils.FEATURES / f"team_features_{gender}.parquet"
    df.to_parquet(out_path, index=False)

    # 12. Summary
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
