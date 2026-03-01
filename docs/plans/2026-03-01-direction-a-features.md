# Direction A — New Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new feature groups (Elo with season carryover, Massey PCA, late-season momentum) to `scripts/02_feature_engineering.py`, register them in `scripts/utils.py`, regenerate all parquets, retune all models, and benchmark a new Kaggle submission.

**Architecture:** All three feature builders are added as new functions in `02_feature_engineering.py`. They are integrated into the existing `build_and_save()` function. `utils.CURATED_FEATURES` and `utils._MONOTONE_MAP` are updated to include the new `*_diff` column names. The rest of the pipeline (scripts 09–13) picks up the new features automatically on the next run — no other scripts need modification.

**Tech Stack:** `pandas`, `numpy`, `sklearn` (PCA + StandardScaler), project `utils.py`. All existing test: `python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"`.

---

### Prerequisite: understand existing build_and_save() structure

Read `scripts/02_feature_engineering.py` sections 1–8 and `scripts/utils.py` before starting.
The merge order in `build_and_save()` is: efficiency → massey → seeds → location → conference → conf_tourney → coach.
New features will be appended in the same pattern at the end of the merge chain.

---

### Task 1: Add Elo features to 02_feature_engineering.py

**Files:**
- Modify: `scripts/02_feature_engineering.py`

**Step 1: Add `build_elo_features(gender)` function**

Insert the following function after `build_conf_tourney_features()` (before Section 4 / `build_and_save()`):

```python
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

        for _, row in season_games.iterrows():
            w      = int(row["WTeamID"])
            l      = int(row["LTeamID"])
            margin = float(row["WScore"] - row["LScore"])
            ew, el = elo.get(w, 1500.0), elo.get(l, 1500.0)
            exp_w  = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
            k_eff  = 20.0 * (1.0 + margin / 20.0) ** 0.6
            delta  = k_eff * (1.0 - exp_w)
            elo[w] += delta
            elo[l] -= delta
            k_wins[w] = k_wins.get(w, 0.0) + k_eff

        for tid in teams:
            records.append({
                "Season":             season,
                "TeamID":             tid,
                "elo_rating":         elo[tid],
                "elo_k_weighted_wins": k_wins[tid],
            })

    result = pd.DataFrame(records)
    result["elo_rating"]          = result["elo_rating"].astype(float)
    result["elo_k_weighted_wins"] = result["elo_k_weighted_wins"].astype(float)
    return result
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```
Expected: `OK`

**Step 3: Smoke test — build Elo features for Men's**

```bash
python -c "
import sys; sys.path.insert(0,'scripts')
from scripts.feature_engineering_test import build_elo_features  # won't work
# Instead:
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('fe', 'scripts/02_feature_engineering.py')
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
df = fe.build_elo_features('M')
print(df.shape)
print(df.describe().round(1))
# Check: all seasons present, elo_rating roughly 1400-1650 range
assert df['elo_rating'].between(1200, 1900).all(), 'Elo out of expected range'
assert df['elo_k_weighted_wins'].ge(0).all(), 'k_wins must be non-negative'
print('Smoke test PASSED')
"
```
Expected: shape (~30k rows × 4 cols), elo_rating mean ~1500, smoke test PASSED.

**Step 4: Integrate into build_and_save()**

In `build_and_save()`, after the coach features block (step 7), add:

```python
    # 8. Elo features
    elo_feats = build_elo_features(gender)
    print(f"  Elo features:        {elo_feats.shape}")
    df = df.merge(elo_feats, on=["Season", "TeamID"], how="left")
```

**Step 5: Verify syntax again**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```
Expected: `OK`

**Step 6: Commit**

```bash
git add scripts/02_feature_engineering.py
git commit -m "feat: add Elo features (margin-of-victory + season carryover) to feature engineering"
```

---

### Task 2: Add Massey PCA features to 02_feature_engineering.py

**Files:**
- Modify: `scripts/02_feature_engineering.py`

**Step 1: Add `build_massey_pca_features()` function**

Insert after `build_elo_features()`:

```python
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

        # Standardise + fill NaN with 0 (= average in standardised space)
        mat = StandardScaler().fit_transform(pivot)
        mat = np.nan_to_num(mat, nan=0.0)

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
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```

**Step 3: Smoke test — build Massey PCA features**

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('fe', 'scripts/02_feature_engineering.py')
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
df = fe.build_massey_pca_features()
print(df.shape)
print(df.describe().round(3))
# Massey only covers Men's 2003+, so expect NaN for pre-2003 seasons
print('NaN pc1:', df['massey_pc1'].isna().sum(), 'NaN pc2:', df['massey_pc2'].isna().sum())
print('Smoke test PASSED')
"
```
Expected: rows ≈ number of Men's teams × Men's seasons with Massey data (2003+), pc1 is standardised (mean ~0).

**Step 4: Integrate into build_and_save()**

After the Elo merge (step 8), add:

```python
    # 9. Massey PCA features (Men's only; Women's gets NaN via left-join)
    if gender == "M":
        massey_pca_feats = build_massey_pca_features()
        print(f"  Massey PCA features: {massey_pca_feats.shape}")
        df = df.merge(massey_pca_feats, on=["Season", "TeamID"], how="left")
    else:
        df["massey_pc1"] = np.nan
        df["massey_pc2"] = np.nan
```

**Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```

**Step 6: Commit**

```bash
git add scripts/02_feature_engineering.py
git commit -m "feat: add Massey PCA features (PC1 consensus, PC2 disagreement) to feature engineering"
```

---

### Task 3: Add momentum features to 02_feature_engineering.py

**Files:**
- Modify: `scripts/02_feature_engineering.py`

**Step 1: Add `build_momentum_features(gender, n_games=10)` function**

Insert after `build_massey_pca_features()`:

```python
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
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```

**Step 3: Smoke test — build momentum features**

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('fe', 'scripts/02_feature_engineering.py')
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
df = fe.build_momentum_features('M')
print(df.shape)
print(df.describe().round(3))
# recent_win_pct should be in [0,1], streak should be non-zero for most teams
assert df['recent_win_pct'].between(0, 1).all(), 'recent_win_pct out of range'
assert df['streak'].dtype in ['int32','int64'], 'streak should be int'
print('Smoke test PASSED')
"
```

**Step 4: Integrate into build_and_save()**

After the Massey PCA merge (step 9), add:

```python
    # 10. Momentum features (last 10 regular season games)
    momentum_feats = build_momentum_features(gender)
    print(f"  Momentum features:   {momentum_feats.shape}")
    df = df.merge(momentum_feats, on=["Season", "TeamID"], how="left")
```

**Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/02_feature_engineering.py').read()); print('OK')"
```

**Step 6: Commit**

```bash
git add scripts/02_feature_engineering.py
git commit -m "feat: add momentum features (recent_win_pct, recent_net_margin, streak) to feature engineering"
```

---

### Task 4: Update utils.py — CURATED_FEATURES + _MONOTONE_MAP

**Files:**
- Modify: `scripts/utils.py`

**Step 1: Add new features to CURATED_FEATURES**

In `scripts/utils.py`, update the `CURATED_FEATURES` set to include the 5 new features:

```python
CURATED_FEATURES = {
    # Seed / ranking
    "SeedNum", "massey_composite", "massey_POM", "massey_MOR", "sos_massey",
    # Massey PCA (replaces/supplements individual systems)
    "massey_pc1", "massey_pc2",
    # Efficiency
    "net_eff", "efg_pct", "oreb_pct", "dreb_pct", "to_pct", "ft_rate",
    # Win rates & margin
    "win_pct", "avg_margin",
    # Neutral-court splits
    "neutral_win_pct", "neutral_net_eff",
    # Conference / schedule quality
    "is_power_conf",
    # Momentum
    "conf_tourney_wins",
    # Coach continuity
    "coach_years_at_school",
    # Elo
    "elo_rating", "elo_k_weighted_wins",
    # Momentum (last 10 games)
    "recent_win_pct", "recent_net_margin", "streak",
}
```

**Step 2: Add new entries to _MONOTONE_MAP**

In `scripts/utils.py`, update `_MONOTONE_MAP` to add:

```python
    # Elo
    "elo_rating_diff":          +1,   # higher Elo = better team
    "elo_k_weighted_wins_diff": +1,   # more dominant wins = better
    # Massey PCA
    "massey_pc1_diff":          +1,   # higher consensus rank = better
    "massey_pc2_diff":           0,   # direction of disagreement is ambiguous
    # Momentum
    "recent_win_pct_diff":      +1,
    "recent_net_margin_diff":   +1,
    "streak_diff":              +1,   # win streak advantage = better
```

**Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/utils.py').read()); print('OK')"
```

**Step 4: Smoke test utils**

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
import utils
# New features should be in CURATED_FEATURES
for f in ['elo_rating','massey_pc1','recent_win_pct','streak']:
    assert f in utils.CURATED_FEATURES, f'{f} missing from CURATED_FEATURES'
# New diffs should be in monotone map
for f in ['elo_rating_diff','massey_pc1_diff','recent_win_pct_diff']:
    assert f in utils._MONOTONE_MAP, f'{f} missing from _MONOTONE_MAP'
print('utils smoke test PASSED')
print(f'CURATED_FEATURES count: {len(utils.CURATED_FEATURES)}')
"
```
Expected: all assertions pass, CURATED_FEATURES count = 24 (was 18, +6 new).

**Step 5: Commit**

```bash
git add scripts/utils.py
git commit -m "feat: register Elo, Massey PCA, momentum features in CURATED_FEATURES + _MONOTONE_MAP"
```

---

### Task 5: Regenerate feature parquets

**Files:** No code changes — run existing script with new features.

**Step 1: Run feature engineering for both genders**

```bash
python scripts/02_feature_engineering.py
```

Expected output (check for):
```
Building Men's features...
  Efficiency features: (XXXX, 15)
  Massey features:     (XXXX, 8)
  Location features:   (XXXX, 8)
  Conference features: (XXXX, 3)
  Conf tourney features: (XXXX, 3)
  Coach features:      (XXXX, 3)
  Elo features:        (XXXX, 4)
  Massey PCA features: (XXXX, 4)
  Momentum features:   (XXXX, 5)
  Season XXXX: YY systems, PC1=XX.X%, PC2=X.X%

Men's features: (XXXX, ~32) -> features/team_features_M.parquet
```

**Step 2: Validate parquet files**

```bash
python -c "
import pandas as pd
for g in ['M', 'W']:
    df = pd.read_parquet(f'features/team_features_{g}.parquet')
    print(f'{g}: shape={df.shape}')
    new_cols = ['elo_rating','elo_k_weighted_wins','massey_pc1','massey_pc2',
                'recent_win_pct','recent_net_margin','streak']
    for col in new_cols:
        assert col in df.columns, f'Missing column: {col}'
        nn = df[col].notna().sum()
        print(f'  {col}: {nn} non-null ({100*nn/len(df):.0f}%)')
print('Validation PASSED')
"
```

Expected:
- Both parquets have new columns
- `elo_rating`, `recent_win_pct`, `recent_net_margin`, `streak` have high coverage (>95%) for both genders
- `massey_pc1`, `massey_pc2` have coverage only for Men's 2003+ (NaN for Women's and pre-2003 Men's)

**Step 3: Commit parquet note (parquets are gitignored, no need to commit them)**

```bash
git add scripts/02_feature_engineering.py  # in case of any last-minute tweaks
git commit -m "chore: regenerated team feature parquets with Elo, Massey PCA, momentum (files gitignored)"
```

---

### Task 6: Retune all 8 models with Optuna

**Step 1: Retune all models (new features change optimal hyperparameters)**

```bash
python scripts/09_tune_optuna.py --model lgbm     --gender M --trials 50
python scripts/09_tune_optuna.py --model lgbm     --gender W --trials 50
python scripts/09_tune_optuna.py --model xgb      --gender M --trials 50
python scripts/09_tune_optuna.py --model xgb      --gender W --trials 50
python scripts/09_tune_optuna.py --model catboost --gender M --trials 50
python scripts/09_tune_optuna.py --model catboost --gender W --trials 50
python scripts/09_tune_optuna.py --model histgb   --gender M --trials 50
python scripts/09_tune_optuna.py --model histgb   --gender W --trials 50
```

Expected: all 8 Brier scores should be ≤ previous best (0.1855 M, 0.1465 W for LGBM).
If any model is significantly worse (>0.005 worse), investigate — likely a feature with too many NaNs.

**Step 2: Rebuild ensemble + calibration + meta-learner**

```bash
python scripts/10_ensemble.py
python scripts/11_calibrate.py
python scripts/13_meta_learner.py
```

Note new OOF Brier scores. Compare to previous benchmark:
- Previous meta_v1: M=0.1850, W=0.1459
- New target: M < 0.1850, W < 0.1459

**Step 3: Check BENCHMARKS.md**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/benchmarks.csv')
# Show best per model per gender
best = df.sort_values('cv_brier').groupby(['gender']).head(5)
print(best[['model','gender','cv_brier','notes']].to_string())
"
```

**Step 4: Commit benchmarks**

```bash
git add results/benchmarks.csv BENCHMARKS.md
git commit -m "chore: update benchmarks after retuning with Direction A features"
```

---

### Task 7: Generate new submission and benchmark on Kaggle

**Step 1: Generate Stage 1 + Stage 2 submissions**

```bash
python scripts/13_meta_learner.py
```

This regenerates `submissions/submission_meta_stage1.csv` and `submissions/submission_meta_stage2.csv`.

**Step 2: Validate submission**

```bash
python -c "
import pandas as pd
for stage in [1, 2]:
    df = pd.read_csv(f'submissions/submission_meta_stage{stage}.csv')
    print(f'Stage {stage}: {len(df):,} rows | '
          f'Pred range: [{df.Pred.min():.4f}, {df.Pred.max():.4f}] | '
          f'mean={df.Pred.mean():.4f} | NaN={df.Pred.isna().sum()}')
    assert df.Pred.between(0.025, 0.975).all()
    assert df.Pred.isna().sum() == 0
print('Validation PASSED')
"
```

Expected: same row counts as before (519,144 / 132,133), all Pred in range, NaN=0.

**Step 3: Submit to Kaggle (manual)**

Upload `submissions/submission_meta_stage1.csv` to the Kaggle competition Stage 1 page.
Record the leaderboard score.

**Step 4: Commit submission + log result**

```bash
git add submissions/submission_meta_stage1.csv submissions/submission_meta_stage2.csv
git commit -m "feat: Direction A submission — Elo + Massey PCA + momentum features (LB: XX.XXXX)"
```

Replace XX.XXXX with the actual leaderboard score.

---

## Acceptance Criteria

- [ ] `build_elo_features(gender)` added to `02_feature_engineering.py`, smoke test passes
- [ ] `build_massey_pca_features()` added, smoke test passes
- [ ] `build_momentum_features(gender)` added, smoke test passes
- [ ] All 3 integrated into `build_and_save()`
- [ ] `utils.CURATED_FEATURES` has 24 features (was 18)
- [ ] `utils._MONOTONE_MAP` has entries for all 7 new `*_diff` columns
- [ ] Both parquets regenerated with new columns, coverage validated
- [ ] All 8 models retuned, new OOF Brier ≤ previous best
- [ ] `submissions/submission_meta_stage1.csv` regenerated and submitted to Kaggle
- [ ] Kaggle LB score recorded in commit message
