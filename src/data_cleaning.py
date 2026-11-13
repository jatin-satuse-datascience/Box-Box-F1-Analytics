"""Data manipulation: clean the raw lap tables and add derived columns (pandas, NumPy).

Main entry points are :func:`prepare_race_laps` and :func:`prepare_qualifying_laps`.
"""

import numpy as np
import pandas as pd

# raw field -> readable column name
LAP_COLUMNS = {
    "drv": "driver", "team": "team", "lap": "lap", "time": "lap_time_s", "pos": "position",
    "compound": "compound", "stint": "stint", "life": "tyre_age", "status": "status",
    "pin": "pit_in_s", "pout": "pit_out_s", "del": "deleted", "ff1G": "generated",
}
NUMERIC_COLUMNS = ["lap_time_s", "position", "pit_in_s", "pit_out_s"]
BOOLEAN_COLUMNS = ["deleted", "generated"]

# Fuel-correction assumptions (approximation: real fuel loads are not public)
START_FUEL_KG = 100.0
FUEL_EFFECT_S_PER_KG = 0.03


def clean_laps(raw, session="race"):
    """Select, rename and type the columns of a raw lap table.

    In the source files a missing value is the *text* ``"None"``, which pandas
    does not treat as missing, so it is converted to ``NaN`` first.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`src.data_loader.load_raw_laps`.
    session : {"race", "qualifying"}
        Qualifying keeps the extra ``quali_session`` column (Q1/Q2/Q3).

    Returns
    -------
    pandas.DataFrame
        Clean table with numeric, boolean and text columns of the right type.
    """
    columns = dict(LAP_COLUMNS)
    if session == "qualifying":
        columns["qs"] = "quali_session"
    df = raw[list(columns)].rename(columns=columns)
    df = df.mask(df.eq("None"))
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].eq(True)
    df["status"] = df["status"].fillna("").astype(str)
    return df


def add_track_status(df):
    """Decode the ``status`` text into boolean flags.

    ``status`` concatenates the track-status codes seen during the lap, for
    example ``"124"`` = clear + yellow + safety car. Codes used: 2 yellow flag,
    4 safety car, 5 red flag, 6/7 virtual safety car (deployed/ending).

    Returns
    -------
    pandas.DataFrame
        Copy with ``yellow``, ``safety_car``, ``red_flag``, ``vsc`` and
        ``neutralised`` (safety car, VSC or red flag) columns.
    """
    out = df.copy()
    status = out["status"].astype(str)
    out["yellow"] = status.str.contains("2")
    out["safety_car"] = status.str.contains("4")
    out["red_flag"] = status.str.contains("5")
    out["vsc"] = status.str.contains("6|7")
    out["neutralised"] = out[["safety_car", "red_flag", "vsc"]].any(axis=1)
    return out


def add_fuel_correction(df, start_fuel_kg=START_FUEL_KG, effect=FUEL_EFFECT_S_PER_KG):
    """Add ``fuel_corrected_s``: the lap time as if the tank were empty.

    Fuel burns linearly, so after lap ``n`` of ``N`` the remaining fuel is
    ``start_fuel_kg * (1 - n / N)``, and each kilogram costs ``effect`` seconds
    per lap. Without this correction cars look faster late in a stint only
    because they are lighter.

    Parameters
    ----------
    df : pandas.DataFrame
        Race lap table with ``lap`` and ``lap_time_s``.
    start_fuel_kg : float, optional
        Fuel at the start of the race.
    effect : float, optional
        Seconds lost per lap for each kilogram on board.
    """
    out = df.copy()
    remaining = start_fuel_kg * (1 - out["lap"] / out["lap"].max())
    out["fuel_corrected_s"] = out["lap_time_s"] - effect * remaining
    return out


def prepare_race_laps(raw):
    """Full cleaning pipeline for the race lap table.

    Adds: track-status flags, ``is_in_lap``, ``is_out_lap``, ``restart`` (first
    lap after a neutralised lap), ``fuel_corrected_s`` and ``is_clean`` (a lap
    that shows the true pace: not lap 1, not a pit lap, not under yellow flag,
    safety car, VSC or red flag, and not a restart lap).

    Returns
    -------
    pandas.DataFrame
        Laps sorted by driver and lap number.
    """
    df = clean_laps(raw, "race").sort_values(["driver", "lap"]).reset_index(drop=True)
    df = add_track_status(df)
    df["is_in_lap"] = df["pit_in_s"].notna()
    df["is_out_lap"] = df["pit_out_s"].notna()
    df["restart"] = df.groupby("driver")["neutralised"].shift(1).eq(True)
    df = add_fuel_correction(df)
    df["is_clean"] = (
        (df["lap"] > 1)
        & ~df["is_in_lap"] & ~df["is_out_lap"]
        & ~df["neutralised"] & ~df["yellow"] & ~df["restart"]
        & ~df["generated"]
        & df["lap_time_s"].notna()
    )
    return df


def prepare_qualifying_laps(raw):
    """Full cleaning pipeline for the qualifying lap table.

    Adds track-status flags, pit flags and ``is_valid_lap``: a timed push lap
    that is neither an in-lap/out-lap nor deleted (for example for track limits).

    Returns
    -------
    pandas.DataFrame
        Laps sorted by driver and lap number.
    """
    df = clean_laps(raw, "qualifying").sort_values(["driver", "lap"]).reset_index(drop=True)
    df = add_track_status(df)
    df["is_in_lap"] = df["pit_in_s"].notna()
    df["is_out_lap"] = df["pit_out_s"].notna()
    df["is_valid_lap"] = (
        df["lap_time_s"].notna() & ~df["is_in_lap"] & ~df["is_out_lap"] & ~df["deleted"] & ~df["generated"]
    )
    return df


def neutralised_ranges(laps):
    """Lap ranges run under safety car, VSC or red flag.

    Returns
    -------
    list of tuple
        ``(kind, first_lap, last_lap)`` for each run of consecutive laps.
    """
    ranges = []
    for kind, column in [("Safety car", "safety_car"), ("VSC", "vsc"), ("Red flag", "red_flag")]:
        flagged = np.sort(laps.loc[laps[column], "lap"].unique())
        if flagged.size == 0:
            continue
        breaks = np.where(np.diff(flagged) > 1)[0] + 1
        for run in np.split(flagged, breaks):
            ranges.append((kind, int(run[0]), int(run[-1])))
    return ranges
