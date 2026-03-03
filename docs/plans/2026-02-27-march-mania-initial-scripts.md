# March Machine Learning Mania 2026 — Initial Scripts Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build project skeleton, data loaders, EDA, feature engineering, and seed-diff baseline — producing saved figures and a first Brier score benchmark.

**Architecture:** Load raw CSVs → engineer per-team per-season features → form matchup rows (team1_feat − team2_feat) → train on historical tourney results → evaluate with leave-one-season-out CV → log to benchmarks.csv.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm 4.6, xgboost 3.1, catboost 1.2, optuna 4.6

**Python interpreter:** `C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe`

**Data directory:** `march-machine-learning-mania-2026/` (relative to project root)

**Project root:** `c:/Users/alexy/Documents/Claude_projects/Kaggle competition/march_learning_mania/`

---

## Task 1: Project scaffold — CLAUDE.md, README.md, requirements.txt, .gitignore

**Files:**
- Create: `CLAUDE.md`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `BENCHMARKS.md`
- Create: `results/benchmarks.csv`

**Step 1: Create CLAUDE.md**

Competition bible — objectives, data overview, evaluation metric, script inventory, commands.

**Step 2: Create README.md**

Professional repo intro with competition context, approach summary, and benchmark table placeholder.

**Step 3: Create requirements.txt**

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.6
lightgbm==4.6.0
xgboost==3.1.2
catboost==1.2.8
optuna==4.6.0
pyarrow>=14.0
```

**Step 4: Create .gitignore**

```
data/
features/*.parquet
models/
march-machine-learning-mania-2026/
march-machine-learning-mania-2026.zip
__pycache__/
*.pyc
.DS_Store
```

**Step 5: Create results/benchmarks.csv**

```csv
model,gender,cv_brier,notes,timestamp
```

**Step 6: Create BENCHMARKS.md**

Header + note that it is updated by training scripts.

**Step 7: Commit**

```bash
git add CLAUDE.md README.md requirements.txt .gitignore BENCHMARKS.md results/benchmarks.csv
git commit -m "chore: project scaffold — CLAUDE.md, README, requirements, benchmarks stub"
```

---

## Task 2: scripts/utils.py — shared data loaders and CV helpers

**Files:**
- Create: `scripts/utils.py`

**Step 1: Write the file**

Key functions to implement:

```python
ROOT = Path(__file__).parent.parent
DATA = ROOT / "march-machine-learning-mania-2026"

def load_compact(gender: str) -> pd.DataFrame:
    """Load MRegularSeasonCompactResults or WRegularSeasonCompactResults."""

def load_detailed(gender: str) -> pd.DataFrame:
    """Load MRegularSeasonDetailedResults or WRegularSeasonDetailedResults."""

def load_tourney(gender: str) -> pd.DataFrame:
    """Load MNCAATourneyCompactResults or WNCAATourneyCompactResults."""

def load_seeds(gender: str) -> pd.DataFrame:
    """Load MNCAATourneySeeds or WNCAATourneySeeds. Returns Season, Seed, TeamID, SeedNum (int)."""

def load_massey() -> pd.DataFrame:
    """Load MMasseyOrdinals."""

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error of probability predictions (lower = better)."""
    return float(np.mean((y_pred - y_true) ** 2))

def get_cv_seasons(tourney_df: pd.DataFrame, n_seasons: int = 10) -> list[int]:
    """Return the last n_seasons with tourney data for leave-one-season-out CV."""

def log_benchmark(model: str, gender: str, cv_brier: float, notes: str = "") -> None:
    """Append a row to results/benchmarks.csv and regenerate BENCHMARKS.md."""
```

**Step 2: Verify it imports cleanly**

```bash
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe -c "import sys; sys.path.insert(0,'scripts'); import utils; print('utils OK')"
```

Expected: `utils OK`

**Step 3: Commit**

```bash
git add scripts/utils.py
git commit -m "feat: add shared data loaders and CV helpers (utils.py)"
```

---

## Task 3: scripts/01_eda.py — exploratory analysis + figures

**Files:**
- Create: `scripts/01_eda.py`
- Output: `figures/01_seed_win_rates_M.png`, `figures/01_seed_win_rates_W.png`
- Output: `figures/02_point_margin_distribution.png`
- Output: `figures/03_upset_rate_by_round.png`
- Output: `figures/04_massey_system_correlation.png`
- Output: `figures/05_season_trend_win_pct.png`

**Step 1: Write the script**

Sections:
1. **Seed win rates**: for each seed 1–16, compute win% in all tourney games. Bar chart M vs W.
2. **Point margin distribution**: histogram of (WScore - LScore) by gender.
3. **Upset rate by round**: % of games where lower seed (higher number) won, grouped by DayNum bracket rounds.
4. **Massey coverage heatmap**: top-20 systems × seasons, show how many teams ranked. Use seaborn heatmap.
5. **Season trend**: avg point margin in tourney over seasons (trend line).

Each figure saved as PNG at 150 dpi. Print confirmation for each save.

**Step 2: Run and verify all 5 figures are saved**

```bash
cd "c:/Users/alexy/Documents/Claude_projects/Kaggle competition/march_learning_mania" && C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe scripts/01_eda.py
```

Expected: 5 `Saved figures/…` lines, no errors.

**Step 3: Commit**

```bash
git add scripts/01_eda.py figures/
git commit -m "eda: exploratory analysis — seed win rates, margins, upsets, Massey coverage"
```

---

## Task 4: scripts/02_feature_engineering.py — team-season features

**Files:**
- Create: `scripts/02_feature_engineering.py`
- Output: `features/team_features_M.parquet`
- Output: `features/team_features_W.parquet`

**Step 1: Write the script**

For each (Season, TeamID) compute from **detailed results** (games where team won or lost):

```
win_pct           = wins / games_played
avg_score_margin  = mean(own_score - opp_score)
off_eff           = points_scored / (FGA - OR + TO + 0.44*FTA)   # ~possessions
def_eff           = points_allowed / opp_possessions
net_eff           = off_eff - def_eff
efg_pct           = (FGM + 0.5*FGM3) / FGA
oreb_pct          = OR / (OR + opp_DR)
dreb_pct          = DR / (DR + opp_OR)
to_pct            = TO / possessions
ft_rate           = FTA / FGA
```

Plus from **Massey ordinals** — for each team-season, average ordinal rank across top-10 coverage systems (MOR, POM, DOK, SAG, MAS, WLK, WIL, PGH, DOL, COL) at the last available RankingDayNum ≤ 133 (day before tourney).

Plus from **seeds** — SeedNum (1–16), is_first_four (bool).

Save each gender's features as parquet.

**Step 2: Run and verify**

```bash
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe scripts/02_feature_engineering.py
```

Expected output:
```
Men's features: (XXXX, YY) → features/team_features_M.parquet
Women's features: (XXXX, YY) → features/team_features_W.parquet
```

**Step 3: Quick sanity check**

```bash
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe -c "
import pandas as pd
m = pd.read_parquet('features/team_features_M.parquet')
print(m.columns.tolist()); print(m.shape); print(m.isnull().sum().sum(), 'nulls')
"
```

Expected: all column names printed, no unexpected nulls.

**Step 4: Commit**

```bash
git add scripts/02_feature_engineering.py
git commit -m "feat: feature engineering — efficiency metrics, Massey composite, seeds (02)"
```

---

## Task 5: scripts/03_train_baseline.py — seed-diff logistic regression

**Files:**
- Create: `scripts/03_train_baseline.py`

**Step 1: Write the script**

- Load `features/team_features_{gender}.parquet` for both M and W
- Build matchup dataframe from tourney results: for each game, row = [seed1 - seed2], label = 1 if lower TeamID won
- Leave-one-season-out CV over last 10 tourney seasons
- Model: `LogisticRegression(C=1.0)` on `[seed_diff]` only
- Compute Brier score per fold, report mean ± std
- Call `log_benchmark("baseline_logreg", gender, mean_brier, "seed-diff only")`
- Save model artifact to `models/baseline_{gender}.pkl`

**Step 2: Run and verify**

```bash
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe scripts/03_train_baseline.py
```

Expected output:
```
[M] Baseline (seed-diff) CV Brier: 0.XXXX ± 0.XXXX
[W] Baseline (seed-diff) CV Brier: 0.XXXX ± 0.XXXX
Saved models/baseline_M.pkl, models/baseline_W.pkl
Benchmarks updated.
```

Typical Brier score for seed-diff baseline: ~0.21–0.23 (lower = better).

**Step 3: Commit**

```bash
git add scripts/03_train_baseline.py results/benchmarks.csv BENCHMARKS.md
git commit -m "feat: seed-diff logistic regression baseline with LOSO-CV Brier score"
```

---

## Notes

- All scripts must be runnable from project root (not from `scripts/`)
- `figures/` dir committed; `features/`, `models/` gitignored
- `results/benchmarks.csv` and `BENCHMARKS.md` committed after every training run
- Python interpreter: `C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe`
