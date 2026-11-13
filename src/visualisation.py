"""Visualisation: matplotlib/seaborn charts, optionally saved to ``outputs/plots``.

Every ``plot_*`` function returns the matplotlib Figure and, if ``save_as`` is
given, also writes it to ``outputs/plots/<save_as>``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
COMPOUND_COLORS = {
    "SOFT": "#e10600", "MEDIUM": "#f5c400", "HARD": "#bdbdbd", "INTERMEDIATE": "#2ca02c", "WET": "#1f77b4",
}
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
NEUTRAL_COLORS = {"Safety car": "#ff9f1c", "VSC": "#ffd23f", "Red flag": "#d62828"}
FALLBACK_COLOR = "#888888"

sns.set_theme(style="whitegrid", context="notebook")


def team_colors(drivers):
    """Map each driver code to its team colour using the drivers table."""
    return dict(zip(drivers["code"], drivers["color"]))


def _save(fig, save_as):
    """Write the figure to ``outputs/plots/<save_as>`` when a name is given."""
    if save_as:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig


def _shade_neutralised(ax, ranges):
    """Shade the lap ranges run under safety car, VSC or red flag."""
    for kind, start, end in ranges:
        ax.axvspan(start - 0.5, end + 0.5, color=NEUTRAL_COLORS[kind], alpha=0.2, linewidth=0, zorder=0)


def _neutral_legend(ranges):
    """Legend patches for the neutralisation kinds that are present."""
    return [Patch(color=NEUTRAL_COLORS[k], alpha=0.35, label=k) for k in dict.fromkeys(r[0] for r in ranges)]


# ----------------------------------------------------------------------------
# Investigation 1: race pace
# ----------------------------------------------------------------------------
def plot_pace_ranking(summary, colors, save_as=None):
    """Horizontal bars: gap of each driver to the best median race pace."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(summary["driver"], summary["gap_to_best_s"], color=[colors.get(d, FALLBACK_COLOR) for d in summary["driver"]])
    ax.invert_yaxis()
    for y, gap in enumerate(summary["gap_to_best_s"]):
        ax.text(gap + 0.05, y, f"+{gap:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Gap to the best median pace (s per lap, fuel-corrected clean laps)")
    ax.set_title("Race pace ranking - 2026 British Grand Prix")
    return _save(fig, save_as)


def plot_pace_vs_consistency(summary, colors, save_as=None):
    """Scatter of pace (gap to best) against consistency (std/mean in %)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["gap_to_best_s"], summary["consistency_pct"], s=70,
               c=[colors.get(d, FALLBACK_COLOR) for d in summary["driver"]], edgecolor="black", zorder=3)
    for row in summary.itertuples():
        ax.annotate(row.driver, (row.gap_to_best_s, row.consistency_pct), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axvline(summary["gap_to_best_s"].median(), color="grey", linestyle="--", linewidth=1)
    ax.axhline(summary["consistency_pct"].median(), color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Gap to the best median pace (s)")
    ax.set_ylabel("Variability: std / mean of lap time (%)  -  lower = more consistent")
    ax.set_title("Pace versus consistency (dashed lines: field medians)")
    return _save(fig, save_as)


# ----------------------------------------------------------------------------
# Investigation 2: tyres and strategy
# ----------------------------------------------------------------------------
def plot_strategy(stints, order, ranges, save_as=None):
    """Stint chart: one bar per stint coloured by compound, neutralised laps shaded."""
    fig, ax = plt.subplots(figsize=(11, 7))
    _shade_neutralised(ax, ranges)
    for i, driver in enumerate(order):
        for row in stints[stints["driver"] == driver].itertuples():
            ax.broken_barh([(row.start_lap - 0.5, row.n_laps)], (i - 0.4, 0.8),
                           facecolors=COMPOUND_COLORS.get(row.compound, FALLBACK_COLOR), edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0.5, stints["end_lap"].max() + 0.5)
    ax.set_xlabel("Lap")
    ax.set_title("Tyre strategies (drivers in finishing order)")
    compounds = [c for c in COMPOUND_ORDER if c in set(stints["compound"])]
    handles = [Patch(facecolor=COMPOUND_COLORS[c], edgecolor="black", label=c.title()) for c in compounds]
    ax.legend(handles=handles + _neutral_legend(ranges), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, save_as)


def plot_degradation(deg, save_as=None):
    """Distribution of the per-stint degradation slopes for each compound."""
    order = [c for c in COMPOUND_ORDER if c in set(deg["compound"])]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.boxplot(data=deg, x="compound", y="slope_s_per_lap", order=order, hue="compound", palette=COMPOUND_COLORS,
                legend=False, width=0.5, fliersize=0, ax=ax)
    sns.stripplot(data=deg, x="compound", y="slope_s_per_lap", order=order, color="black", size=4, alpha=0.7, ax=ax)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    for i, compound in enumerate(order):
        ax.text(i, ax.get_ylim()[1], f"n={int((deg['compound'] == compound).sum())}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Degradation: lap time lost per lap of tyre age (s)")
    ax.set_title("Tyre degradation per stint (fuel-corrected)", pad=22)
    return _save(fig, save_as)


def plot_position_trace(laps, drivers, colors, ranges, save_as=None):
    """Race position of selected drivers lap by lap; triangles mark pit entries."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _shade_neutralised(ax, ranges)
    team_seen = {}
    for driver in drivers:
        sub = laps[laps["driver"] == driver].sort_values("lap")
        team = sub["team"].iloc[0]
        style = ["-", "--", ":"][team_seen.get(team, 0) % 3]
        team_seen[team] = team_seen.get(team, 0) + 1
        color = colors.get(driver, FALLBACK_COLOR)
        ax.plot(sub["lap"], sub["position"], color=color, linestyle=style, linewidth=2.2, label=driver)
        stops = sub[sub["is_in_lap"] & sub["position"].notna()]
        ax.scatter(stops["lap"], stops["position"], marker="v", s=60, color=color, edgecolor="black", zorder=4)
    lowest = int(laps.loc[laps["driver"].isin(drivers), "position"].max())
    ax.set_ylim(lowest + 0.5, 0.5)
    ax.set_yticks(range(1, lowest + 1))
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    ax.set_title("Race positions (triangles = pit entry)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + _neutral_legend(ranges), loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, save_as)


# ----------------------------------------------------------------------------
# Investigation 3: qualifying versus race
# ----------------------------------------------------------------------------
def plot_quali_vs_race(table, correlation, colors, save_as=None):
    """Scatter of qualifying position against race-pace rank, with the diagonal."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    n = len(table)
    ax.plot([0, n + 1], [0, n + 1], color="grey", linestyle="--", linewidth=1)
    ax.scatter(table["qualifying_position"], table["pace_rank"], s=80,
               c=[colors.get(d, FALLBACK_COLOR) for d in table["driver"]], edgecolor="black", zorder=3)
    for row in table.itertuples():
        ax.annotate(row.driver, (row.qualifying_position, row.pace_rank), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel("Qualifying position")
    ax.set_ylabel("Race-pace rank")
    ax.set_title(f"Qualifying vs race pace (Spearman rho = {correlation['spearman_rho']:.2f})")
    ax.text(n * 0.52, n * 0.12, "above the line: race pace worse than qualifying\nbelow the line: race pace better", fontsize=8, color="dimgrey")
    return _save(fig, save_as)
