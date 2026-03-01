# Kaggle Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write two self-contained, pedagogical Jupyter notebooks that run end-to-end on Kaggle using competition data at `/kaggle/input/march-machine-learning-mania-2026/`, targeting intermediate Kaggle users who want to understand the full approach.

**Architecture:** Two `.ipynb` files in `notebooks/`. Each notebook is fully standalone — no imports from `scripts/`, no external artifacts. All feature engineering and model training computed from scratch inside the notebook. Kaggle runtime ~15-25 min each. All paths use `/kaggle/input/march-machine-learning-mania-2026/`.

**Tech Stack:** `pandas`, `numpy`, `sklearn`, `lightgbm`, `optuna`, `matplotlib`, `seaborn`, Jupyter notebooks.

---

### Notebook structure conventions

- Every section starts with a **Markdown cell** explaining the concept (2-4 paragraphs)
- Followed by one or more **Code cells** implementing it
- Output cells should show representative `.head()` or plots — make them informative
- Keep code cells short (≤ 30 lines each) for readability
- Use `print()` liberally to show progress and results

---

### Task 1: Set up notebooks directory

**Files:**
- Create: `notebooks/` directory

**Step 1: Create directory**

```bash
mkdir -p notebooks
```

**Step 2: Verify**
```bash
ls notebooks/
```
Expected: empty directory exists.

---

### Task 2: Write Notebook 1 — Feature Engineering

**Files:**
- Create: `notebooks/01_feature_engineering.ipynb`

This notebook has 8 sections. Each section is described below with exact cell content.

---

#### Cell 1 — Markdown: Title + Introduction

```markdown
# NCAA Basketball Tournament — Feature Engineering

## March Machine Learning Mania 2026

In this notebook we build the features that will power our tournament prediction model.

**What we're predicting:** For every possible matchup in the 2026 NCAA tournament, predict the probability that the lower-seeded team (by TeamID, not seed number) wins.

**Why Brier Score?** The competition uses Brier Score = mean((p - y)²), not accuracy. This rewards *calibrated* probabilities — saying 0.7 when the true probability is 0.7 is better than saying 1.0. Extreme overconfidence is penalised heavily.

**Our approach:** Engineer per-team, per-season features → represent each matchup as the *difference* between two teams' features → train a classifier to predict win probability.
```

---

#### Cell 2 — Code: Imports + data paths

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────
DATA = Path("/kaggle/input/march-machine-learning-mania-2026")

# Verify data is present
csv_files = list(DATA.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files")
print("Key files:", [f.name for f in csv_files if "Compact" in f.name or "Massey" in f.name][:6])
```

---

#### Cell 3 — Markdown: Data Tour

```markdown
## 1. Data Tour

The competition provides 30+ CSV files. We use four main sources:

| File | Content | Rows |
|------|---------|------|
| `MRegularSeasonCompactResults` | Men's game-by-game results 1985–2025 | ~180k |
| `MRegularSeasonDetailedResults` | Men's box scores 2003–2025 | ~120k |
| `MMasseyOrdinals` | 196 external rating systems (Pomeroy, Sagarin, etc.) | ~5.7M |
| `MNCAATourneySeeds` | Tournament seeds 1985–2025 | ~8k |

Each game row has `WTeamID` (winner), `LTeamID` (loser), `WScore`, `LScore`, `DayNum` (day of season), and `WLoc` (H/A/N).
```

---

#### Cell 4 — Code: Data tour

```python
# Quick look at each key file
compact  = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
detailed = pd.read_csv(DATA / "MRegularSeasonDetailedResults.csv")
massey   = pd.read_csv(DATA / "MMasseyOrdinals.csv")
seeds    = pd.read_csv(DATA / "MNCAATourneySeeds.csv")

print("=== Regular Season Compact ===")
print(compact.shape, "\n", compact.head(3))

print("\n=== Regular Season Detailed ===")
print(detailed.shape, "\n", detailed.columns.tolist())

print("\n=== Massey Ordinals ===")
print(massey.shape)
print("Systems:", massey["SystemName"].nunique(), "unique rating systems")
print(massey.head(3))

print("\n=== Tournament Seeds ===")
print(seeds.shape, "\n", seeds.head(3))
```

---

#### Cell 5 — Markdown: Efficiency Metrics

```markdown
## 2. Efficiency Metrics (Four-Factor Model)

Raw points and win% don't tell the full story. A team scoring 80 points in 60 possessions is very different from one scoring 80 in 75 possessions.

We compute **adjusted efficiency** using the Kubatko possession formula:

> Possessions ≈ FGA − ORB + TO + 0.44 × FTA

Then:
- **Offensive efficiency (off_eff)** = 100 × points scored / possessions
- **Defensive efficiency (def_eff)** = 100 × points allowed / possessions
- **Net efficiency (net_eff)** = off_eff − def_eff ← our primary quality metric

We also compute the **Four Factors** (Dean Oliver):
- **eFG%** = (FGM + 0.5 × FGM3) / FGA — shooting quality
- **OREB%** = OR / (OR + opponent DR) — offensive rebounding
- **TOV%** = TO / possessions — turnover rate
- **FT Rate** = FTA / FGA — free throw generation

These 4 factors explain ~95% of scoring variance at the team level.
```

---

#### Cell 6 — Code: Build efficiency features

```python
def build_efficiency_features(compact_df, detailed_df):
    """Aggregate box-score stats into per-(Season, TeamID) efficiency metrics."""
    det = detailed_df.copy()

    # Reshape: winner perspective + loser perspective
    win_rows = pd.DataFrame({
        "Season": det.Season, "TeamID": det.WTeamID,
        "own_pts": det.WScore, "opp_pts": det.LScore,
        "FGM": det.WFGM, "FGA": det.WFGA, "FGM3": det.WFGM3,
        "FTA": det.WFTA, "OR": det.WOR, "DR": det.WDR, "TO": det.WTO,
        "oFGA": det.LFGA, "oOR": det.LOR, "oTO": det.LTO, "oFTA": det.LFTA, "oDR": det.LDR,
        "won": 1,
    })
    loss_rows = pd.DataFrame({
        "Season": det.Season, "TeamID": det.LTeamID,
        "own_pts": det.LScore, "opp_pts": det.WScore,
        "FGM": det.LFGM, "FGA": det.LFGA, "FGM3": det.LFGM3,
        "FTA": det.LFTA, "OR": det.LOR, "DR": det.LDR, "TO": det.LTO,
        "oFGA": det.WFGA, "oOR": det.WOR, "oTO": det.WTO, "oFTA": det.WFTA, "oDR": det.WDR,
        "won": 0,
    })
    games = pd.concat([win_rows, loss_rows], ignore_index=True)
    games["margin"] = games["own_pts"] - games["opp_pts"]

    grp  = games.groupby(["Season", "TeamID"])
    sums = grp[["own_pts","opp_pts","FGA","FGM","FGM3","FTA","OR","DR","TO",
                "oFGA","oOR","oTO","oFTA","oDR"]].sum().reset_index()
    base = grp.agg(win_pct=("won","mean"), avg_margin=("margin","mean")).reset_index()
    agg  = base.merge(sums, on=["Season","TeamID"])

    own_poss = (agg.FGA - agg.OR  + agg.TO  + 0.44*agg.FTA).replace(0, np.nan)
    opp_poss = (agg.oFGA - agg.oOR + agg.oTO + 0.44*agg.oFTA).replace(0, np.nan)

    agg["off_eff"]  = agg.own_pts / own_poss * 100
    agg["def_eff"]  = agg.opp_pts / opp_poss * 100
    agg["net_eff"]  = agg.off_eff - agg.def_eff
    agg["efg_pct"]  = (agg.FGM + 0.5*agg.FGM3) / agg.FGA
    agg["oreb_pct"] = agg.OR  / (agg.OR  + agg.oDR)
    agg["dreb_pct"] = agg.DR  / (agg.DR  + agg.oOR)
    agg["to_pct"]   = agg.TO  / own_poss
    agg["ft_rate"]  = agg.FTA / agg.FGA

    return agg[["Season","TeamID","win_pct","avg_margin","net_eff","off_eff","def_eff",
                "efg_pct","oreb_pct","dreb_pct","to_pct","ft_rate"]].copy()

eff_m = build_efficiency_features(compact, detailed)
print(f"Efficiency features shape: {eff_m.shape}")
print(eff_m.describe().round(2))
```

---

#### Cell 7 — Markdown: Elo Ratings

```markdown
## 3. Elo Ratings with Margin-of-Victory

Efficiency metrics summarise the *average* season. But Elo captures *trajectory* — it updates after every game, so it knows whether a team is peaking or fading.

**Standard Elo update:**
```
expected_A = 1 / (1 + 10^((elo_B - elo_A) / 400))
delta = K × (outcome − expected_A)
```

**Margin-of-victory scaling (key improvement):**
A 30-point win should count more than a 1-point win. We scale the K-factor:
```
K_eff = 20 × (1 + margin/20)^0.6
```
This is the formula used by 4th-place winners in this competition.

**Season carryover (75/25 mean reversion):**
Program quality persists across seasons — Duke is reliably strong year after year. We carry forward 75% of each team's prior Elo:
```
elo_start = prev_season_elo × 0.75 + 1500 × 0.25
```
New teams initialise at 1500 (the global mean).
```

---

#### Cell 8 — Code: Build Elo features

```python
def build_elo_features(compact_df):
    """Margin-of-victory Elo with season carryover (75/25 mean reversion)."""
    seasons = sorted(compact_df["Season"].unique())
    elo    = {}   # {TeamID: float}
    records = []

    for season in seasons:
        season_games = compact_df[compact_df["Season"] == season].sort_values("DayNum")
        teams = set(season_games["WTeamID"]) | set(season_games["LTeamID"])

        # Carryover / initialise
        for tid in teams:
            elo[tid] = elo[tid] * 0.75 + 1500 * 0.25 if tid in elo else 1500.0

        k_wins = {tid: 0.0 for tid in teams}

        for _, row in season_games.iterrows():
            w, l   = int(row["WTeamID"]), int(row["LTeamID"])
            margin = row["WScore"] - row["LScore"]
            ew     = elo.get(w, 1500.0)
            el     = elo.get(l, 1500.0)
            exp_w  = 1 / (1 + 10 ** ((el - ew) / 400))
            k_eff  = 20 * (1 + margin / 20) ** 0.6
            delta  = k_eff * (1 - exp_w)
            elo[w] += delta
            elo[l] -= delta
            k_wins[w] = k_wins.get(w, 0.0) + k_eff

        for tid in teams:
            records.append({
                "Season": season, "TeamID": tid,
                "elo_rating": elo[tid],
                "elo_k_weighted_wins": k_wins[tid],
            })

    return pd.DataFrame(records)

elo_m = build_elo_features(compact)
print(f"Elo features shape: {elo_m.shape}")

# Plot top/bottom 10 teams by final Elo (most recent season)
latest = elo_m[elo_m["Season"] == elo_m["Season"].max()].sort_values("elo_rating", ascending=False)
print("\nTop 10 teams by Elo (most recent season):")
print(latest.head(10)[["TeamID","elo_rating"]].to_string(index=False))
```

---

#### Cell 9 — Markdown: Massey Ordinals + PCA

```markdown
## 4. Massey Ordinals — 196 Rating Systems → 2 PCA Dimensions

The competition provides `MMasseyOrdinals.csv` — rankings from 196 external systems (Ken Pomeroy, Sagarin, Moore, RPI, BPI, and 191 others). Each system has different methodology and blind spots.

**Problem:** Picking 3 systems and averaging them leaves signal on the table. But using all 196 as separate features causes severe multicollinearity — many systems rank teams nearly identically.

**Solution: PCA (Principal Component Analysis)**

PCA finds orthogonal axes that explain maximum variance across all systems:
- **PC1** (~80% variance): the "consensus ranking" — roughly, the average of all systems
- **PC2** (~10% variance): where systems *disagree* — this captures teams that are ranked very differently by, say, pace-adjusted vs. raw-points systems

We keep only systems with ≥50% team coverage to avoid noise from niche systems with sparse data.
```

---

#### Cell 10 — Code: Build Massey PCA features

```python
def build_massey_pca_features(massey_df):
    """PCA across all Massey systems with >= 50% coverage. Returns massey_pc1, massey_pc2."""
    df = massey_df[massey_df["RankingDayNum"] <= 133].copy()

    # Last available rank per (Season, TeamID, System)
    df = df.sort_values("RankingDayNum")
    last = df.groupby(["Season","TeamID","SystemName"])["OrdinalRank"].last().reset_index()

    records = []
    for season in sorted(last["Season"].unique()):
        sdf   = last[last["Season"] == season]
        pivot = sdf.pivot_table(index="TeamID", columns="SystemName",
                                values="OrdinalRank", aggfunc="first")

        # Keep systems covering >= 50% of teams
        coverage = pivot.notna().sum() / len(pivot)
        keep     = coverage[coverage >= 0.5].index.tolist()
        if len(keep) < 2:
            for tid in pivot.index:
                records.append({"Season": season, "TeamID": int(tid),
                                "massey_pc1": np.nan, "massey_pc2": np.nan})
            continue

        pivot = pivot[keep]
        # Flip sign: lower rank number = better → higher score = better
        pivot = pivot.max().max() - pivot + 1

        # Standardise, fill missing with 0 (= mean in z-score space)
        mat = StandardScaler().fit_transform(pivot)
        mat = np.nan_to_num(mat, nan=0.0)

        n_comp = min(2, mat.shape[1])
        pcs    = PCA(n_components=n_comp).fit_transform(mat)

        # Align PC1 so that better teams have higher values
        # Heuristic: if top-ranked team has negative PC1, flip
        if pcs[0, 0] < 0 and pivot.iloc[0].mean() > pivot.mean().mean():
            pcs[:, 0] = -pcs[:, 0]

        for i, tid in enumerate(pivot.index):
            records.append({
                "Season": season, "TeamID": int(tid),
                "massey_pc1": pcs[i, 0],
                "massey_pc2": pcs[i, 1] if n_comp >= 2 else np.nan,
            })

    result = pd.DataFrame(records)
    print(f"Massey PCA features: {result.shape}")
    print(f"Systems per season (example): {len(keep)} (from {massey_df['SystemName'].nunique()} total)")
    return result

massey_pca_m = build_massey_pca_features(massey)
print(massey_pca_m.describe().round(3))
```

---

#### Cell 11 — Markdown: Late-Season Momentum

```markdown
## 5. Late-Season Momentum

A team's full-season average masks whether they are *peaking* or *fading* entering the tournament.

Consider two teams with identical 20-10 records:
- Team A: went 8-2 in their last 10 games (hot)
- Team B: went 2-8 in their last 10 games (cold)

These teams have the same seed and same win%, but very different tournament prospects.

We compute three momentum features from the **last 10 regular season games** (by `DayNum`):
- `recent_win_pct` — win rate over last 10 games
- `recent_net_margin` — average point margin over last 10 games
- `streak` — current streak entering the tournament (positive = wins, negative = losses)
```

---

#### Cell 12 — Code: Build momentum features

```python
def build_momentum_features(compact_df, n_games=10):
    """Last-N-games momentum features per (Season, TeamID)."""
    win_rows  = compact_df[["Season","DayNum","WTeamID","WScore","LScore"]].copy()
    win_rows["TeamID"] = win_rows["WTeamID"]
    win_rows["won"]    = 1
    win_rows["margin"] = win_rows["WScore"] - win_rows["LScore"]

    loss_rows  = compact_df[["Season","DayNum","LTeamID","WScore","LScore"]].copy()
    loss_rows["TeamID"] = loss_rows["LTeamID"]
    loss_rows["won"]    = 0
    loss_rows["margin"] = loss_rows["LScore"] - loss_rows["WScore"]

    games = pd.concat([
        win_rows[["Season","DayNum","TeamID","won","margin"]],
        loss_rows[["Season","DayNum","TeamID","won","margin"]],
    ], ignore_index=True).sort_values(["Season","TeamID","DayNum"])

    last_n = games.groupby(["Season","TeamID"]).tail(n_games)

    agg = last_n.groupby(["Season","TeamID"]).agg(
        recent_win_pct=("won",    "mean"),
        recent_net_margin=("margin", "mean"),
    ).reset_index()

    # Streak: consecutive outcomes from the last game backwards
    def compute_streak(grp):
        outcomes = grp.sort_values("DayNum")["won"].values
        if len(outcomes) == 0:
            return 0
        val, streak = outcomes[-1], 0
        for o in reversed(outcomes):
            if o == val:
                streak += 1
            else:
                break
        return streak if val == 1 else -streak

    streak = (games.groupby(["Season","TeamID"])
              .apply(compute_streak)
              .reset_index(name="streak"))

    return agg.merge(streak, on=["Season","TeamID"], how="left")

momentum_m = build_momentum_features(compact)
print(f"Momentum features shape: {momentum_m.shape}")
print(momentum_m.describe().round(3))
```

---

#### Cell 13 — Markdown: Merge + Correlation

```markdown
## 6. Merging All Features + Correlation Analysis

We merge all feature groups on `(Season, TeamID)` and inspect the correlation matrix. This guides feature selection — highly correlated features (r > 0.85) add redundancy without signal.

Key relationships to verify:
- `elo_rating` should correlate strongly with `net_eff` (both measure team quality) — but not perfectly, since Elo is sequential and efficiency is aggregate
- `massey_pc1` should correlate strongly with both (consensus of 196 external systems)
- Momentum features should show lower correlation with the above (they capture recent form, not season average)
```

---

#### Cell 14 — Code: Merge and correlation heatmap

```python
# Merge all features
seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
seeds_df["SeedNum"] = seeds_df["Seed"].str[1:3].astype(int)

team_features = (eff_m
    .merge(elo_m,         on=["Season","TeamID"], how="left")
    .merge(massey_pca_m,  on=["Season","TeamID"], how="left")
    .merge(momentum_m,    on=["Season","TeamID"], how="left")
    .merge(seeds_df[["Season","TeamID","SeedNum"]], on=["Season","TeamID"], how="left")
)

print(f"Final feature matrix: {team_features.shape}")
print(f"NaN counts:\n{team_features.isnull().sum()[team_features.isnull().sum() > 0]}")

# Correlation heatmap — numeric features only, tourney teams only (have SeedNum)
tourney_teams = team_features.dropna(subset=["SeedNum"])
feature_cols  = ["SeedNum","net_eff","elo_rating","massey_pc1","massey_pc2",
                 "efg_pct","oreb_pct","to_pct","win_pct","avg_margin",
                 "recent_win_pct","recent_net_margin","streak"]
corr = tourney_teams[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix (tournament teams only)", pad=12)
plt.tight_layout()
plt.show()
```

---

#### Cell 15 — Markdown: Save

```markdown
## 7. Saving Features

We save the merged feature table as a Parquet file. In the modelling notebook we load this directly and build matchup rows by computing `team1_feature - team2_feature` for each possible pair.

One important note: **we keep features as season-level aggregates** (not matchup-level). The matchup diff is computed at training/prediction time. This means we only need to engineer features once per team per season, not once per matchup (which would be hundreds of thousands of rows).
```

---

#### Cell 16 — Code: Save and summary

```python
out_path = Path("/kaggle/working/team_features_M.parquet")
team_features.to_parquet(out_path, index=False)
print(f"Saved to {out_path}")
print(f"Shape: {team_features.shape}")
print(f"Seasons: {team_features['Season'].min()} – {team_features['Season'].max()}")
print(f"Unique teams: {team_features['TeamID'].nunique()}")
print("\nFeature columns:")
for col in team_features.columns:
    nn = team_features[col].notna().sum()
    print(f"  {col:<30} {nn:>6} non-null ({100*nn/len(team_features):.0f}%)")
```

---

**Step 2: Verify notebook runs end-to-end (locally)**

```bash
cd march_learning_mania
jupyter nbconvert --to notebook --execute notebooks/01_feature_engineering.ipynb \
    --output notebooks/01_feature_engineering_executed.ipynb 2>&1 | tail -20
```

Expected: `[NbConvertApp] Notebook written to ...` with no errors.

**Step 3: Commit**

```bash
git add notebooks/01_feature_engineering.ipynb
git commit -m "feat: notebook 1 — feature engineering (Elo, Massey PCA, momentum)"
```

---

### Task 3: Write Notebook 2 — Modelling Pipeline

**Files:**
- Create: `notebooks/02_modelling_pipeline.ipynb`

---

#### Cell 1 — Markdown: Title + Introduction

```markdown
# NCAA Basketball Tournament — Modelling Pipeline

## March Machine Learning Mania 2026

This notebook trains a prediction model for NCAA tournament outcomes using the features built in [Notebook 1](../01_feature_engineering.ipynb).

**Pipeline overview:**
1. Build matchup rows (feature diffs between pairs of teams)
2. Establish a seed-difference baseline
3. Train LightGBM with Optuna hyperparameter search
4. Combine models into an ensemble
5. Stack with a meta-learner
6. Generate a submission CSV

**Key design choice — LOSO-CV:** We use Leave-One-Season-Out cross-validation: each fold withholds one full tournament year (~67 games) and trains on all prior years. This matches the test distribution (predict a single future tournament) much better than random k-fold.
```

---

#### Cell 2 — Code: Imports + data loading

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = Path("/kaggle/input/march-machine-learning-mania-2026")

# ── Helper: Brier Score ────────────────────────────────────────────────────
def brier_score(y_true, y_pred):
    """MSE of probability predictions. Lower is better."""
    return float(np.mean((np.clip(y_pred, 0, 1) - y_true) ** 2))

# ── Load raw data ──────────────────────────────────────────────────────────
compact  = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
detailed = pd.read_csv(DATA / "MRegularSeasonDetailedResults.csv")
massey   = pd.read_csv(DATA / "MMasseyOrdinals.csv")
tourney  = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
seeds_df["SeedNum"] = seeds_df["Seed"].str[1:3].astype(int)

print("Data loaded:")
print(f"  Regular season games: {len(compact):,}")
print(f"  Tournament games:     {len(tourney):,}")
print(f"  Massey rankings:      {len(massey):,}")
```

---

#### Cell 3 — Code: Re-build features inline (paste functions from Notebook 1)

```python
# [Include all feature-building functions from Notebook 1 here]
# build_efficiency_features(), build_elo_features(),
# build_massey_pca_features(), build_momentum_features()
# Then merge + save to team_features

# ... (functions copied from Notebook 1) ...

team_features = (
    build_efficiency_features(compact, detailed)
    .merge(build_elo_features(compact),          on=["Season","TeamID"], how="left")
    .merge(build_massey_pca_features(massey),    on=["Season","TeamID"], how="left")
    .merge(build_momentum_features(compact),     on=["Season","TeamID"], how="left")
    .merge(seeds_df[["Season","TeamID","SeedNum"]], on=["Season","TeamID"], how="left")
)
print(f"Team features: {team_features.shape}")
```

---

#### Cell 4 — Markdown: Matchup rows

```markdown
## 2. Building Matchup Rows

Each training example is one tournament game. We represent it as:

```
feature_diff = team1_feature − team2_feature
```

Where Team1 = lower TeamID, Team2 = higher TeamID. Label = 1 if Team1 won.

This antisymmetric encoding means the model sees the same information regardless of which team is "first" — flipping the sign of all diffs gives the prediction for the reverse matchup.

We use a **NaN-tolerant** matchup builder: rows are only dropped if *all* feature diffs are NaN (both teams entirely absent from the feature set). Partial NaN (e.g. Massey unavailable for recent seasons) is kept and filled by a SimpleImputer downstream.
```

---

#### Cell 5 — Code: Build matchup DataFrame + CV seasons

```python
FEAT_COLS_RAW = [c for c in team_features.columns
                 if c not in ("Season", "TeamID")]
CURATED = {
    "SeedNum", "net_eff", "efg_pct", "oreb_pct", "dreb_pct", "to_pct", "ft_rate",
    "win_pct", "avg_margin", "elo_rating", "elo_k_weighted_wins",
    "massey_pc1", "massey_pc2",
    "recent_win_pct", "recent_net_margin", "streak",
}

def build_matchup_df(tourney_df, features_df):
    feat_cols = [c for c in features_df.columns if c not in ("Season","TeamID")]
    df = tourney_df.copy()
    df["Team1ID"] = df[["WTeamID","LTeamID"]].min(axis=1)
    df["Team2ID"] = df[["WTeamID","LTeamID"]].max(axis=1)
    df["Label"]   = (df["WTeamID"] == df["Team1ID"]).astype(int)
    feat = features_df.set_index(["Season","TeamID"])
    t1 = df[["Season","Team1ID"]].rename(columns={"Team1ID":"TeamID"}).join(feat, on=["Season","TeamID"]).drop(columns="TeamID")
    t2 = df[["Season","Team2ID"]].rename(columns={"Team2ID":"TeamID"}).join(feat, on=["Season","TeamID"]).drop(columns="TeamID")
    result = df[["Season","Team1ID","Team2ID","Label"]].copy()
    for c in feat_cols:
        result[f"{c}_diff"] = t1[c].values - t2[c].values
    return result.reset_index(drop=True)

all_matchups = build_matchup_df(tourney, team_features)
FEAT_COLS = [c for c in all_matchups.columns
             if c.endswith("_diff") and c.replace("_diff","") in CURATED]
print(f"Matchup rows: {len(all_matchups):,}  |  Features used: {len(FEAT_COLS)}")

# CV seasons: last 10 tournament seasons excluding 2020
all_seasons = sorted(tourney["Season"].unique())
cv_seasons  = [s for s in all_seasons if s != 2020][-10:]
print(f"CV seasons: {cv_seasons}")
```

---

#### Cell 6 — Markdown: Seed Baseline

```markdown
## 3. Seed-Difference Baseline

Before any ML, let's establish what "free information" gives us.

A simple logistic regression trained only on `SeedNum_diff` (Team1 seed − Team2 seed) achieves Brier ≈ 0.196. This is our floor — any model that can't beat this isn't learning anything useful beyond seeds.

Lower seeds (1, 2) beat higher seeds (15, 16) about 95% of the time historically. But seeds are noisy for mid-seed matchups (8 vs 9 is essentially a coin flip). Net efficiency and Elo should help here.
```

---

#### Cell 7 — Code: Seed baseline LOSO-CV

```python
from sklearn.impute import SimpleImputer

def run_loso_cv(model_fn, feat_cols, all_matchups, cv_seasons):
    """Run leave-one-season-out CV. model_fn(X_train, y_train) → fitted model."""
    oof_preds, oof_labels = [], []
    for val_season in cv_seasons:
        train = all_matchups[all_matchups["Season"] != val_season]
        val   = all_matchups[all_matchups["Season"] == val_season]
        X_tr, y_tr = train[feat_cols].values, train["Label"].values
        X_va, y_va = val[feat_cols].values,   val["Label"].values

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)

        model = model_fn(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        oof_preds.extend(preds)
        oof_labels.extend(y_va)

    return brier_score(np.array(oof_labels), np.array(oof_preds))

seed_cols = [c for c in FEAT_COLS if "SeedNum" in c]
brier_seed = run_loso_cv(
    lambda X, y: LogisticRegression(C=1e9).fit(X, y),
    seed_cols, all_matchups, cv_seasons
)
print(f"Seed baseline LOSO-CV Brier: {brier_seed:.4f}")
```

---

#### Cell 8 — Markdown: LightGBM + Optuna

```markdown
## 4. LightGBM with Optuna Hyperparameter Search

LightGBM (gradient boosted trees) consistently outperforms logistic regression on tabular data with non-linear feature interactions.

**Key hyperparameters we tune:**
- `num_leaves` — tree complexity (default 31; we search 10–150)
- `learning_rate` — step size (we search 0.01–0.3)
- `min_child_samples` — regularisation via minimum leaf size (search 10–100)
- `reg_lambda`, `reg_alpha` — L2/L1 regularisation

**Optuna uses Tree-structured Parzen Estimation (TPE):** Unlike grid search or random search, TPE builds a probabilistic model of which hyperparameter regions are promising and samples from them. It typically finds good solutions in 30–50 trials vs hundreds for grid search.
```

---

#### Cell 9 — Code: LightGBM Optuna tuning

```python
import lightgbm as lgb

def lgbm_objective(trial, feat_cols, all_matchups, cv_seasons):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves":        trial.suggest_int("num_leaves", 10, 150),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "binary", "verbose": -1, "n_jobs": -1,
    }
    oof_preds, oof_labels = [], []
    for val_season in cv_seasons:
        train = all_matchups[all_matchups["Season"] != val_season]
        val   = all_matchups[all_matchups["Season"] == val_season]
        X_tr, y_tr = train[feat_cols].values, train["Label"].values
        X_va, y_va = val[feat_cols].values,   val["Label"].values
        imp  = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr); X_va = imp.transform(X_va)
        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        oof_preds.extend(model.predict_proba(X_va)[:, 1])
        oof_labels.extend(y_va)
    return brier_score(np.array(oof_labels), np.array(oof_preds))

print("Running Optuna (30 trials — increase for better results)...")
study = optuna.create_study(direction="minimize")
study.optimize(
    lambda t: lgbm_objective(t, FEAT_COLS, all_matchups, cv_seasons),
    n_trials=30, show_progress_bar=True
)
print(f"\nBest LGBM Brier: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

#### Cell 10 — Markdown: Ensemble + Meta-Learner

```markdown
## 5. Ensemble + Meta-Learner

A single model's predictions can be overconfident in certain matchup types. Averaging predictions from multiple models with different inductive biases reduces this variance.

**Simple mean ensemble:** Average the probability predictions from each model. This works well when models are already well-calibrated (which Optuna-tuned models tend to be).

**Meta-learner (stacking):** Instead of a fixed-weight average, train a second-level model to learn the optimal combination. The trick is to train it on **out-of-fold predictions** — predictions generated by each base model on data it never saw during training. This prevents the meta-learner from just memorising the most confident base model.

We transform base predictions to **logit space** (log-odds) before feeding them to the meta-learner, because logistic regression is linear in log-odds — the natural scale for combining probability estimates.
```

---

#### Cell 11 — Code: OOF collection + meta-learner

```python
# Collect OOF predictions from tuned LGBM
# (In a full pipeline you'd do this for all 4 models)

def collect_oof(best_params, feat_cols, all_matchups, cv_seasons):
    """Collect out-of-fold predictions using best hyperparameters."""
    oof_df = all_matchups[all_matchups["Season"].isin(cv_seasons)][
        ["Season","Label"]].copy().reset_index(drop=True)
    oof_df["pred"] = np.nan

    for val_season in cv_seasons:
        train = all_matchups[all_matchups["Season"] != val_season]
        val   = all_matchups[all_matchups["Season"] == val_season]
        X_tr, y_tr = train[feat_cols].values, train["Label"].values
        X_va       = val[feat_cols].values

        imp  = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr); X_va = imp.transform(X_va)

        model = lgb.LGBMClassifier(**best_params, objective="binary", verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        mask = oof_df["Season"] == val_season
        oof_df.loc[mask, "pred"] = model.predict_proba(X_va)[:, 1]

    return oof_df

oof = collect_oof(study.best_params, FEAT_COLS, all_matchups, cv_seasons)

# Meta-learner: LogReg on logit(OOF)
EPS = 1e-7
def to_logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p / (1 - p))

y_true = oof["Label"].values
L      = to_logit(oof["pred"].values).reshape(-1, 1)

meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
meta.fit(L, y_true)

meta_pred  = meta.predict_proba(L)[:, 1]
brier_oof  = brier_score(y_true, oof["pred"].values)
brier_meta = brier_score(y_true, meta_pred)
print(f"OOF LGBM Brier:      {brier_oof:.4f}")
print(f"Meta-learner Brier:  {brier_meta:.4f}  (delta={brier_meta-brier_oof:+.4f})")
```

---

#### Cell 12 — Markdown: Submission

```markdown
## 6. Generating the Submission

The sample submission has IDs in format `YYYY_Team1ID_Team2ID` — all possible matchups for the given year. We:
1. Parse the ID into season, Team1ID, Team2ID
2. Determine gender (Men's TeamIDs < 3000)
3. Build feature diffs for each row
4. Predict with our pipeline
5. Clip to [0.025, 0.975] — avoids extreme overconfidence

This notebook covers Men's only for brevity. A production pipeline processes both genders.
```

---

#### Cell 13 — Code: Generate submission

```python
sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
parsed = sub["ID"].str.split("_", expand=True).astype(int)
sub["Season"]  = parsed[0]
sub["Team1ID"] = parsed[1]
sub["Team2ID"] = parsed[2]
sub["gender"]  = np.where(sub["Team1ID"] < 3000, "M", "W")

# Men's only for this notebook
sub_m = sub[sub["gender"] == "M"].copy()
feat  = team_features.set_index(["Season","TeamID"])

t1 = (sub_m[["Season","Team1ID"]].rename(columns={"Team1ID":"TeamID"})
      .join(feat, on=["Season","TeamID"]).drop(columns="TeamID"))
t2 = (sub_m[["Season","Team2ID"]].rename(columns={"Team2ID":"TeamID"})
      .join(feat, on=["Season","TeamID"]).drop(columns="TeamID"))

raw_cols = [c.replace("_diff","") for c in FEAT_COLS]
X_sub_df = pd.DataFrame({fc: t1[rc].values - t2[rc].values
                          for fc, rc in zip(FEAT_COLS, raw_cols)})

# Train final model on all available data
from sklearn.impute import SimpleImputer
X_all = all_matchups[FEAT_COLS].values
y_all = all_matchups["Label"].values
imp_final = SimpleImputer(strategy="median")
X_all_imp = imp_final.fit_transform(X_all)
final_model = lgb.LGBMClassifier(**study.best_params, objective="binary", verbose=-1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    final_model.fit(X_all_imp, y_all)

X_sub_imp = imp_final.transform(X_sub_df.values)
base_pred  = final_model.predict_proba(X_sub_imp)[:, 1]
meta_pred  = meta.predict_proba(to_logit(base_pred).reshape(-1,1))[:, 1]
sub_m["Pred"] = np.clip(meta_pred, 0.025, 0.975)

out = sub_m[["ID","Pred"]].copy()
sub.loc[sub["gender"]=="M", "Pred"] = sub_m["Pred"].values
sub.loc[sub["gender"]=="W", "Pred"] = 0.5  # placeholder for Women's

sub[["ID","Pred"]].to_csv("/kaggle/working/submission.csv", index=False)
print(f"Submission saved: {len(sub)} rows")
print(f"Men's Pred range: [{out.Pred.min():.3f}, {out.Pred.max():.3f}]  mean={out.Pred.mean():.3f}")
print(sub[["ID","Pred"]].head())
```

---

**Step 2: Verify notebook runs end-to-end (locally with adapted path)**

```bash
jupyter nbconvert --to notebook --execute notebooks/02_modelling_pipeline.ipynb \
    --output notebooks/02_modelling_pipeline_executed.ipynb 2>&1 | tail -20
```

Expected: completes without errors, prints Brier scores and submission row count.

**Step 3: Commit**

```bash
git add notebooks/02_modelling_pipeline.ipynb
git commit -m "feat: notebook 2 — modelling pipeline (LOSO-CV, Optuna, ensemble, meta-learner)"
```

---

## Acceptance Criteria

- [ ] `notebooks/01_feature_engineering.ipynb` exists, runs end-to-end on Kaggle
- [ ] `notebooks/02_modelling_pipeline.ipynb` exists, runs end-to-end on Kaggle
- [ ] Each notebook has explanatory markdown before every major code section
- [ ] Both produce output (feature stats, Brier scores, submission CSV)
- [ ] Both committed to git
