# Feature Enhancements Design — March Mania 2026

**Date:** 2026-03-03
**Status:** Approved
**Current best LB (Stage 1):** 0.13361

---

## Overview

Four progressive feature enhancements to `02_feature_engineering.py`, each benchmarked individually via LOSO-CV Brier before being kept. A full Optuna retune runs only after all four pass their benchmarks — to avoid tuning against a moving target.

**Implementation order (lowest to highest risk):**

| # | Enhancement | Change type | Expected signal |
|---|---|---|---|
| 1 | Haupts Elo (7 features) | Replace `build_elo_features()` | Medium-High |
| 2 | Conf tourney in momentum | Remove DayNum filter | High |
| 3 | Historical tournament performance | New function | High |
| 4 | Tournament box score efficiency | New function | Medium |

**Leakage rule:** Features for Season S may only use regular season S data and tournament/game data from seasons 1…S-1. Current season tournament games are what we're predicting — they must never appear in features.

**Benchmark rule:** Run LOSO-CV after each feature addition. Keep only if Brier improves or is neutral. Log results in `results/benchmarks.csv`.

---

## Enhancement 1 — Haupts Elo (replaces current Elo)

**Function:** `build_elo_features(gender, compact_df, tourney_df)`
**Replaces:** current `build_elo_features(gender)` which produced `elo_rating`, `elo_k_weighted_wins`

### Design

**Continuous accumulation across seasons:** No per-season reset (no 75/25 mean reversion). The `team_dict` persists across all years — sustained excellence compounds naturally.

**Tournament games in the update chain:** Regular season and tournament games concatenated, sorted by `(Season, DayNum)`, processed together.
- Men's: regular weight=1.0, tourney weight=0.75
- Women's: regular weight=0.95, tourney weight=1.0
- Rationale: Women's tournament games are weighted higher due to more quality variation in regular season

**MoV multiplier retained:** `k_eff = k * weight * (1 + margin/20)^0.6` — blowouts move ratings more than 1-point wins. Haupts used no MoV multiplier; we keep ours as it was already validated.

**Parameters (Haupts-tuned):**
- `initial_rating = width = 1200`
- Men's: `k=125`, Women's: `k=190`

**Leakage safeguard:** After processing all games, filter `Tourney == 0` before computing per-season summary statistics. Season S's features reflect only regular season S games. Tournament games only affect the *starting Elo for season S+1*.

### 7 Output Features

| Feature | Description | Monotone |
|---|---|---|
| `elo_last` | Final regular season rating (≈ old `elo_rating`) | +1 |
| `elo_mean` | Average rating during regular season | +1 |
| `elo_median` | Median rating (robust to spikes) | +1 |
| `elo_std` | Rating variance — consistency signal | 0 |
| `elo_min` | Worst point of season | +1 |
| `elo_max` | Best point of season | +1 |
| `elo_trend` | Linear regression slope over season | +1 |

### utils.py changes
- **Remove** from `CURATED_FEATURES` + `_MONOTONE_MAP`: `elo_rating`, `elo_k_weighted_wins`
- **Add**: 7 new entries above (net: +5 features, 25 → 30)

---

## Enhancement 2 — Conference Tournament Games in Momentum

**Function:** `build_momentum_features(compact_df, n_games=10)`
**Change:** Remove the `DayNum < 132` filter

### Design

The data files already enforce the leakage boundary:
- `MRegularSeasonCompactResults.csv` contains ALL non-NCAA-tournament games, including conference tournament games (DayNum ~132–154)
- `MNCAATourneyCompactResults.csv` contains only NCAA tournament games

Since `build_momentum_features()` only reads from `compact_df` (regular season file), NCAA tournament games from the current season are **physically absent** from the input — no DayNum filter is needed for leakage protection.

**Impact on features:**
- `recent_win_pct`, `recent_net_margin`: now include conference tournament games in the last-10 window
- `streak`: most impacted — a team winning their conference tournament enters March Madness on a meaningful winning streak (previously invisible to the model)

**Code change:** Remove `comp = comp[comp["DayNum"] < 132].copy()` — one line deletion.

---

## Enhancement 3 — Historical Tournament Performance

**Function:** `build_tourney_history_features(tourney_df, n_seasons=3)`
**Source data:** `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv`

### Design

For each `(Season S, TeamID)`, aggregate tournament game data from seasons S-1, S-2, S-3.

**3 Output Features:**

| Feature | Description | Monotone |
|---|---|---|
| `tourney_win_pct_hist` | Win rate across all tourney games in last 3 seasons | +1 |
| `tourney_net_margin_hist` | Average net scoring margin in tourney games | +1 |
| `tourney_rounds_advanced_avg` | Average wins per appearance (0=R1 exit, 6=champion) | +1 |

**Missing data:** Teams with no prior tournament appearances → NaN for win rate and margin, 0 for rounds advanced. Downstream imputer handles NaN as population median.

**Leakage safeguard:** `past_seasons = seasons where season >= S - n_seasons AND season < S`. Never includes season S.

**Why these three:** `tourney_rounds_advanced_avg` encodes most of what a seed-overperformance feature would capture while being simpler. Adding seed-relative features risks redundancy with the existing `SeedNum_diff`.

---

## Enhancement 4 — Tournament Box Score Efficiency

**Function:** `build_tourney_efficiency_features(tourney_detailed_df, n_seasons=3)`
**Source data:** `MNCAATourneyDetailedResults.csv` / `WNCAATourneyDetailedResults.csv`

### Design

Possession-adjusted efficiency computed from prior tournament box scores. Same rolling window as Enhancement 3: seasons S-1 through S-N.

**2 Output Features:**

| Feature | Description | Monotone |
|---|---|---|
| `tourney_off_eff_hist` | Off. efficiency in prior tourney games (pts/100 poss) | +1 |
| `tourney_def_eff_hist` | Def. efficiency in prior tourney games (opp pts/100 poss) | -1 |

**Why separate off/def over net:** Gives the model granularity — elite defense + mediocre offense is a qualitatively different tournament proposition than the reverse.

**Why possession-adjusted over raw margin:** Tournament games still vary in pace; possession adjustment adds precision over `tourney_net_margin_hist` (Enhancement 3). These are complementary, not redundant.

**Leakage safeguard:** `Season < S` strictly, same as Enhancement 3.

---

## Final State After All Enhancements

**Feature count:** 25 (current) - 2 (old Elo) + 7 (Haupts Elo) + 3 (tourney history) + 2 (tourney efficiency) = **35 features**

**Files modified:**
- `scripts/02_feature_engineering.py` — 4 function updates
- `scripts/utils.py` — `CURATED_FEATURES` + `_MONOTONE_MAP` updated
- `notebooks/02_modelling_pipeline.ipynb` — mirror feature functions updated

**Post-implementation:** Full Optuna retune for all 8 models (4 model types × 2 genders) after all benchmarks pass.
