"""Driver analysis: race pace (Investigation 1) and qualifying vs race (Investigation 3).

Uses pandas, NumPy and SciPy on the tables produced by :mod:`src.data_cleaning`.
"""

import numpy as np
import pandas as pd
from scipy import stats

# FIA rule: a driver is classified if he completed at least 90% of the race distance
CLASSIFICATION_FRACTION = 0.9
SESSION_ORDER = {"Q3": 0, "Q2": 1, "Q1": 2}  # Q3 drivers start ahead of Q2, then Q1


def race_classification(laps):
    """Final classification table of the race.

    Parameters
    ----------
    laps : pandas.DataFrame
        Prepared race laps.

    Returns
    -------
    pandas.DataFrame
        Per driver: ``laps_completed``, ``stops``, ``last_position`` (last known
        race position, missing for cars that stopped on track), ``fastest_lap_s``
        (deleted laps excluded) and ``classified`` (90% distance rule). Sorted
        by laps completed, then position.
    """
    race_laps = laps["lap"].max()
    table = (
        laps.groupby(["driver", "team"])
        .agg(laps_completed=("lap", "max"), stops=("stint", lambda s: s.max() - 1))
        .reset_index()
    )
    last_position = laps.dropna(subset=["position"]).sort_values("lap").groupby("driver")["position"].last()
    valid = laps[~laps["deleted"] & ~laps["generated"]]
    fastest = valid.groupby("driver")["lap_time_s"].min()
    table["last_position"] = table["driver"].map(last_position)
    table["fastest_lap_s"] = table["driver"].map(fastest)
    table["classified"] = table["laps_completed"] >= CLASSIFICATION_FRACTION * race_laps
    return table.sort_values(["laps_completed", "last_position"], ascending=[False, True]).reset_index(drop=True)


def race_pace_summary(laps):
    """Race pace and consistency of every driver, on clean laps only.

    Statistics are given on the raw lap time and on the fuel-corrected time
    (suffix ``_fc_s``). The fuel-corrected values are the fair ones to compare
    drivers and to measure consistency, because the raw spread also contains
    the car getting lighter.

    Returns
    -------
    pandas.DataFrame
        Columns: clean_laps, mean/median/std of raw time, mean/median/std of
        fuel-corrected time, ``fastest_lap_s``, ``consistency_pct`` (std divided
        by mean, in percent; lower is more consistent), ``gap_to_best_s`` (median
        fuel-corrected gap to the best driver), ``pace_rank`` and
        ``consistency_rank`` (1 = best). Sorted by pace.
    """
    clean = laps[laps["is_clean"]]
    summary = (
        clean.groupby(["driver", "team"])
        .agg(
            clean_laps=("lap_time_s", "count"),
            mean_lap_s=("lap_time_s", "mean"),
            median_lap_s=("lap_time_s", "median"),
            std_lap_s=("lap_time_s", "std"),
            mean_fc_s=("fuel_corrected_s", "mean"),
            median_fc_s=("fuel_corrected_s", "median"),
            std_fc_s=("fuel_corrected_s", "std"),
        )
        .reset_index()
    )
    valid = laps[~laps["deleted"] & ~laps["generated"]]
    summary["fastest_lap_s"] = summary["driver"].map(valid.groupby("driver")["lap_time_s"].min())
    summary["consistency_pct"] = 100 * summary["std_fc_s"] / summary["mean_fc_s"]
    summary["gap_to_best_s"] = summary["median_fc_s"] - summary["median_fc_s"].min()
    summary["pace_rank"] = summary["median_fc_s"].rank(method="first").astype(int)
    summary["consistency_rank"] = summary["consistency_pct"].rank(method="first").astype(int)
    return summary.sort_values("pace_rank").reset_index(drop=True)


def qualifying_classification(qlaps):
    """Qualifying order, following the Q1/Q2/Q3 knockout rules.

    Drivers who reached Q3 are ranked by their best Q3 lap, those knocked out in
    Q2 by their Q2 lap, and those knocked out in Q1 by their Q1 lap. Deleted laps
    are ignored. This is the qualifying order, *not* the starting grid (grid
    penalties are not in the data).

    Parameters
    ----------
    qlaps : pandas.DataFrame
        Prepared qualifying laps.

    Returns
    -------
    pandas.DataFrame
        driver, team, reached_session, best_in_session_s, best_lap_s (best of
        all sessions), qualifying_position and gap_to_pole_pct.
    """
    valid = qlaps[qlaps["is_valid_lap"]]
    reached = qlaps.groupby("driver")["quali_session"].max()
    best_by_session = valid.groupby(["driver", "quali_session"])["lap_time_s"].min()
    table = qlaps[["driver", "team"]].drop_duplicates().reset_index(drop=True)
    table["reached_session"] = table["driver"].map(reached)
    table["best_in_session_s"] = [
        best_by_session.get((d, s), np.nan) for d, s in zip(table["driver"], table["reached_session"])
    ]
    table["best_lap_s"] = table["driver"].map(valid.groupby("driver")["lap_time_s"].min())
    table["_order"] = table["reached_session"].map(SESSION_ORDER)
    table = table.sort_values(["_order", "best_in_session_s"], na_position="last").drop(columns="_order")
    table = table.reset_index(drop=True)
    table["qualifying_position"] = np.arange(1, len(table) + 1)
    pole = table["best_lap_s"].min()
    table["gap_to_pole_pct"] = 100 * (table["best_lap_s"] - pole) / pole
    return table


def qualifying_vs_race(quali, pace, classification):
    """Combine qualifying and race performance in one table.

    Parameters
    ----------
    quali : pandas.DataFrame
        Output of :func:`qualifying_classification`.
    pace : pandas.DataFrame
        Output of :func:`race_pace_summary`.
    classification : pandas.DataFrame
        Output of :func:`race_classification`.

    Returns
    -------
    pandas.DataFrame
        Per driver: qualifying position and gap to pole (%), race-pace rank and
        gap to the best race pace (%), ``rank_difference`` (qualifying position
        minus pace rank: positive means the race pace was better than qualifying
        suggested), final position and ``positions_gained`` (qualifying position
        minus final position, only for classified drivers with a known position).
    """
    table = quali[["driver", "team", "qualifying_position", "gap_to_pole_pct"]].merge(
        pace[["driver", "pace_rank", "median_fc_s"]], on="driver"
    )
    best = table["median_fc_s"].min()
    table["race_gap_pct"] = 100 * (table["median_fc_s"] - best) / best
    table["rank_difference"] = table["qualifying_position"] - table["pace_rank"]
    table = table.merge(classification[["driver", "last_position", "classified"]], on="driver", how="left")
    table["positions_gained"] = (table["qualifying_position"] - table["last_position"]).where(table["classified"])
    return table.drop(columns="median_fc_s").sort_values("qualifying_position").reset_index(drop=True)


def quali_race_correlation(table):
    """Correlation between qualifying and race performance.

    Returns
    -------
    dict
        ``spearman_rho``/``spearman_p`` between qualifying position and race-pace
        rank (rank correlation), and ``pearson_r``/``pearson_p`` between the two
        percentage gaps.
    """
    rho, rho_p = stats.spearmanr(table["qualifying_position"], table["pace_rank"])
    r, r_p = stats.pearsonr(table["gap_to_pole_pct"], table["race_gap_pct"])
    return {"spearman_rho": float(rho), "spearman_p": float(rho_p), "pearson_r": float(r), "pearson_p": float(r_p)}
