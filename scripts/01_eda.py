"""
01_eda.py — March Machine Learning Mania 2026 EDA
Produces 5 figures saved to figures/ at 150 dpi.
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import utils

sns.set_theme(style="whitegrid")

FIGURES = utils.FIGURES
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Figure 1: Seed Win Rates ──────────────────────────────────────────────────

def make_seed_win_rates():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("NCAA Tournament Win Rates by Seed (1985–2025)", fontsize=15, fontweight="bold")

    genders = [("M", "Men's"), ("W", "Women's")]

    for ax, (gender, label) in zip(axes, genders):
        tourney = utils.load_tourney(gender)
        seeds = utils.load_seeds(gender)
        seed_map = seeds.set_index(["Season", "TeamID"])["SeedNum"]

        wins = {}
        losses = {}

        for _, row in tourney.iterrows():
            season = row["Season"]
            w_id = row["WTeamID"]
            l_id = row["LTeamID"]

            w_seed = seed_map.get((season, w_id), None)
            l_seed = seed_map.get((season, l_id), None)

            if w_seed is not None:
                wins[w_seed] = wins.get(w_seed, 0) + 1
            if l_seed is not None:
                losses[l_seed] = losses.get(l_seed, 0) + 1

        seeds_range = range(1, 17)
        win_rates = []
        for s in seeds_range:
            w = wins.get(s, 0)
            l = losses.get(s, 0)
            total = w + l
            win_rates.append(w / total if total > 0 else 0.0)

        colors = ["#4caf50" if r >= 0.5 else "#fa8072" for r in win_rates]
        bars = ax.bar(list(seeds_range), win_rates, color=colors, edgecolor="white", linewidth=0.5)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="0.5 baseline")
        ax.set_xlabel("Seed", fontsize=12)
        ax.set_ylabel("Win Rate", fontsize=12)
        ax.set_title(f"{label} Tournament — Seed Win Rates", fontsize=13)
        ax.set_xticks(list(seeds_range))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)

    plt.tight_layout()
    out = FIGURES / "01_seed_win_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Point Margin Distribution ──────────────────────────────────────

def make_margin_distribution():
    tm = utils.load_tourney("M")
    tw = utils.load_tourney("W")

    margins_m = tm["WScore"] - tm["LScore"]
    margins_w = tw["WScore"] - tw["LScore"]

    mean_m = margins_m.mean()
    mean_w = margins_w.mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(margins_m, bins=40, alpha=0.6, color="steelblue",
            label=f"Men's (mean={mean_m:.1f})", edgecolor="white", linewidth=0.4)
    ax.hist(margins_w, bins=40, alpha=0.6, color="coral",
            label=f"Women's (mean={mean_w:.1f})", edgecolor="white", linewidth=0.4)

    ax.axvline(mean_m, color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(mean_w, color="coral", linestyle="--", linewidth=2)

    ax.set_title("Tournament Winning Margin Distribution (1985–2025)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Winning Margin (points)", fontsize=12)
    ax.set_ylabel("Games", fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = FIGURES / "02_point_margin_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Upset Rate by Round ─────────────────────────────────────────────

def make_upset_rate_by_round():
    tourney = utils.load_tourney("M")
    seeds = utils.load_seeds("M")
    seed_map = seeds.set_index(["Season", "TeamID"])["SeedNum"]

    # DayNum → Round name mapping (based on NCAA bracket structure)
    def day_to_round(day):
        if day in (134, 135):
            return "First Four"
        elif day in (136, 137):
            return "Round of 64"
        elif day in (138, 139):
            return "Round of 32"
        elif day in (143, 144, 145, 146):
            return "Sweet 16 / Elite 8"
        elif day == 152:
            return "Final Four"
        elif day == 154:
            return "Championship"
        else:
            return None

    round_order = [
        "Round of 64",
        "Round of 32",
        "Sweet 16 / Elite 8",
        "Final Four",
        "Championship",
    ]

    round_upsets = {r: {"upsets": 0, "total": 0} for r in round_order}

    for _, row in tourney.iterrows():
        season = row["Season"]
        day = row["DayNum"]
        round_name = day_to_round(day)
        if round_name is None or round_name == "First Four":
            continue

        w_seed = seed_map.get((season, row["WTeamID"]), None)
        l_seed = seed_map.get((season, row["LTeamID"]), None)
        if w_seed is None or l_seed is None:
            continue

        round_upsets[round_name]["total"] += 1
        # Upset: winner has a higher seed number than loser
        if w_seed > l_seed:
            round_upsets[round_name]["upsets"] += 1

    upset_rates = []
    for r in round_order:
        total = round_upsets[r]["total"]
        upsets = round_upsets[r]["upsets"]
        upset_rates.append(100.0 * upsets / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#e57373", "#ef9a9a", "#ffcc80", "#a5d6a7", "#80cbc4"]
    bars = ax.bar(round_order, upset_rates, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, upset_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Upset Rate by Tournament Round (Men's, 1985–2025)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Upset Rate (%)", fontsize=12)
    ax.set_ylim(0, max(upset_rates) * 1.18)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    out = FIGURES / "03_upset_rate_by_round.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Massey Coverage Heatmap ────────────────────────────────────────

def make_massey_coverage_heatmap():
    massey = utils.load_massey()

    # Filter to seasons 2010–2026
    massey = massey[(massey["Season"] >= 2010) & (massey["Season"] <= 2026)]

    # Count unique teams per system per season
    coverage = (
        massey.groupby(["SystemName", "Season"])["TeamID"]
        .nunique()
        .reset_index(name="TeamCount")
    )

    # Select top 15 systems by total team-season coverage
    top_systems = (
        coverage.groupby("SystemName")["TeamCount"]
        .sum()
        .nlargest(15)
        .index
    )
    coverage = coverage[coverage["SystemName"].isin(top_systems)]

    pivot = coverage.pivot(index="SystemName", columns="Season", values="TeamCount").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Teams Ranked"},
    )
    ax.set_title(
        "Massey Ordinals Coverage: Teams Ranked per System per Season",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Ranking System", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    out = FIGURES / "04_massey_coverage_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: Average Margin Trend ───────────────────────────────────────────

def make_avg_margin_trend():
    tm = utils.load_tourney("M")
    tw = utils.load_tourney("W")

    tm["margin"] = tm["WScore"] - tm["LScore"]
    tw["margin"] = tw["WScore"] - tw["LScore"]

    avg_m = tm.groupby("Season")["margin"].mean().reset_index()
    avg_w = tw.groupby("Season")["margin"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(avg_m["Season"], avg_m["margin"], color="steelblue", marker="o",
            markersize=4, linewidth=1.5, label="Men's", alpha=0.85)
    ax.plot(avg_w["Season"], avg_w["margin"], color="coral", marker="s",
            markersize=4, linewidth=1.5, label="Women's", alpha=0.85)

    # Linear trend lines
    for df, color in [(avg_m, "steelblue"), (avg_w, "coral")]:
        x = df["Season"].values
        y = df["margin"].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color=color, linestyle="--", linewidth=1.5, alpha=0.5)

    ax.set_title("Average Tournament Winning Margin by Season", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Avg Winning Margin (points)", fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = FIGURES / "05_avg_margin_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running EDA...")
    make_seed_win_rates()
    make_margin_distribution()
    make_upset_rate_by_round()
    make_massey_coverage_heatmap()
    make_avg_margin_trend()
    print("Done.")
