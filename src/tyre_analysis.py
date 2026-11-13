"""Tyre and strategy analysis (Investigation 2): stints, degradation and pit stops.

Uses pandas, NumPy and SciPy on the race lap table from :mod:`src.data_cleaning`.
"""

import numpy as np
import pandas as pd
from scipy import stats

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def stints_table(laps):
    """One row per stint: compound, first/last lap, length and tyre age at start."""
    return (
        laps.groupby(["driver", "team", "stint"])
        .agg(
            compound=("compound", "first"),
            start_lap=("lap", "min"),
            end_lap=("lap", "max"),
            n_laps=("lap", "count"),
            start_age=("tyre_age", "min"),
        )
        .reset_index()
    )


def degradation(laps, min_laps=6):
    """Estimate tyre degradation for every stint.

    A straight line ``fuel_corrected_time = a + b * tyre_age`` is fitted to the
    clean laps of each stint with :func:`scipy.stats.linregress`. The slope ``b``
    is the lap time lost (seconds) for every extra lap of tyre age. Positive means
    the tyre is wearing out. The estimate is noisy: track evolution (more grip as
    rubber is laid down), traffic and the approximate fuel correction all affect it.

    Parameters
    ----------
    laps : pandas.DataFrame
        Prepared race laps.
    min_laps : int, optional
        Minimum number of clean laps needed to fit a stint.

    Returns
    -------
    pandas.DataFrame
        One row per fitted stint: driver, team, stint, compound, n_laps,
        slope_s_per_lap, intercept_s and r_squared.
    """
    columns = ["driver", "team", "stint", "compound", "n_laps", "slope_s_per_lap", "intercept_s", "r_squared"]
    rows = []
    clean = laps[laps["is_clean"]]
    for (driver, team, stint, compound), group in clean.groupby(["driver", "team", "stint", "compound"]):
        if len(group) < min_laps or group["tyre_age"].nunique() < 2:
            continue
        fit = stats.linregress(group["tyre_age"], group["fuel_corrected_s"])
        rows.append((driver, team, stint, compound, len(group), fit.slope, fit.intercept, fit.rvalue**2))
    return pd.DataFrame(rows, columns=columns)


def degradation_by_compound(deg):
    """Summarise the stint slopes per compound.

    Besides the count, median, mean and spread of the slopes, a Wilcoxon
    signed-rank test checks whether the slopes are systematically above zero
    (``p_value_vs_zero``); it needs at least 6 stints, otherwise it is left empty.
    """
    rows = []
    for compound, group in deg.groupby("compound"):
        slopes = group["slope_s_per_lap"]
        try:
            p_value = stats.wilcoxon(slopes).pvalue if len(slopes) >= 6 else np.nan
        except ValueError:  # all slopes identical
            p_value = np.nan
        rows.append((compound, len(slopes), slopes.median(), slopes.mean(), slopes.std(), p_value))
    summary = pd.DataFrame(
        rows, columns=["compound", "n_stints", "median_slope", "mean_slope", "std_slope", "p_value_vs_zero"]
    )
    summary["_order"] = summary["compound"].map({c: i for i, c in enumerate(COMPOUND_ORDER)})
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def compare_compounds(deg, first="MEDIUM", second="HARD"):
    """Test whether two compounds have different degradation (Mann-Whitney U test).

    The test is non-parametric, so it does not assume normally distributed slopes.

    Returns
    -------
    dict
        Number of stints and median slope of each compound, the U statistic and
        the two-sided ``p_value``. A p-value above 0.05 means the difference could
        easily be due to chance.
    """
    a = deg.loc[deg["compound"] == first, "slope_s_per_lap"]
    b = deg.loc[deg["compound"] == second, "slope_s_per_lap"]
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        f"n_{first.lower()}": len(a), f"n_{second.lower()}": len(b),
        f"median_{first.lower()}": float(a.median()), f"median_{second.lower()}": float(b.median()),
        "u_statistic": float(result.statistic), "p_value": float(result.pvalue),
    }


def pit_lane_times(laps):
    """Time spent in the pit lane for every stop, from the pit-in/pit-out timestamps.

    The pit-in time is on the in-lap row and the pit-out time on the next lap's
    row, so the duration is ``pit_out(next lap) - pit_in(this lap)``.

    Returns
    -------
    pandas.DataFrame
        driver, team, lap (in-lap), duration_s and ``under_neutralised`` (True if
        the stop happened under safety car or VSC). Stops that never left the pits
        (retirements) are left out.
    """
    df = laps.sort_values(["driver", "lap"]).copy()
    grouped = df.groupby("driver")
    df["next_pit_out_s"] = grouped["pit_out_s"].shift(-1)
    df["next_neutralised"] = grouped["neutralised"].shift(-1).eq(True)
    stops = df[df["is_in_lap"]].copy()
    stops["duration_s"] = stops["next_pit_out_s"] - stops["pit_in_s"]
    stops["under_neutralised"] = stops["neutralised"] | stops["next_neutralised"]
    stops = stops.dropna(subset=["duration_s"])
    return stops[["driver", "team", "lap", "duration_s", "under_neutralised"]].reset_index(drop=True)


def pit_lane_summary(pit_times):
    """Pit-lane time statistics for stops under green flag and under SC/VSC."""
    labelled = pit_times.assign(condition=np.where(pit_times["under_neutralised"], "Under SC/VSC", "Green flag"))
    return (
        labelled.groupby("condition")["duration_s"]
        .agg(n_stops="count", median_s="median", mean_s="mean", fastest_s="min")
        .reset_index()
    )


def strategy_summary(laps):
    """Strategy of each driver: number of stops, tyre sequence and stops under SC/VSC.

    Returns
    -------
    pandas.DataFrame
        driver, team, stops, strategy (for example ``"M-H-S"``), sc_stops,
        laps_completed and last_position; sorted by laps completed and position.
    """
    stints = stints_table(laps).sort_values(["driver", "stint"])
    strategy = stints.groupby("driver")["compound"].agg(lambda s: "-".join(COMPOUND_LETTER.get(c, "?") for c in s))
    sc_stops = laps[laps["is_in_lap"] & laps["neutralised"]].groupby("driver").size()
    last_position = laps.dropna(subset=["position"]).sort_values("lap").groupby("driver")["position"].last()
    table = laps.groupby(["driver", "team"]).agg(stops=("stint", lambda s: s.max() - 1), laps_completed=("lap", "max"))
    table = table.reset_index()
    table["strategy"] = table["driver"].map(strategy)
    table["sc_stops"] = table["driver"].map(sc_stops).fillna(0).astype(int)
    table["last_position"] = table["driver"].map(last_position)
    cols = ["driver", "team", "stops", "strategy", "sc_stops", "laps_completed", "last_position"]
    return table[cols].sort_values(["laps_completed", "last_position"], ascending=[False, True]).reset_index(drop=True)
