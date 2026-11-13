"""Input/Output: read the TracingInsights JSON files and save processed tables.

Expected layout::

    data/raw/race/session_laptimes.json         # lap-by-lap race data
    data/raw/race/drivers.json                  # driver names, teams, colours
    data/raw/qualifying/session_laptimes.json   # lap-by-lap qualifying data

A lap file is one JSON object whose keys are columns and whose values are lists
of the same length (one entry per driver per lap), so it converts directly to a
DataFrame. No cleaning is done here: see :mod:`src.data_cleaning`.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_LAP_FIELDS = {
    "drv", "team", "lap", "time", "pos", "compound", "stint", "life", "status", "pin", "pout", "del", "ff1G",
}


def read_json(path):
    """Read a JSON file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def load_raw_laps(session, raw_dir=RAW_DIR):
    """Load the lap file of a session as a raw (uncleaned) DataFrame.

    Parameters
    ----------
    session : {"race", "qualifying"}
        Which session to load.
    raw_dir : str or Path, optional
        Folder that contains the ``race`` and ``qualifying`` folders.

    Returns
    -------
    pandas.DataFrame
        One row per driver per lap, with the original columns.

    Raises
    ------
    ValueError
        If required fields are missing (qualifying also needs ``qs``, the Q1/Q2/Q3 label).
    """
    data = read_json(Path(raw_dir) / session / "session_laptimes.json")
    required = set(REQUIRED_LAP_FIELDS) | ({"qs"} if session == "qualifying" else set())
    missing = required - set(data)
    if missing:
        raise ValueError(f"{session} lap file is missing fields: {sorted(missing)}")
    return pd.DataFrame(data)


def load_drivers(raw_dir=RAW_DIR):
    """Load driver names, teams and team colours from the race ``drivers.json``.

    Returns
    -------
    pandas.DataFrame
        Columns ``code``, ``name``, ``team``, ``number`` and ``color`` (#RRGGBB).
    """
    df = pd.DataFrame(read_json(Path(raw_dir) / "race" / "drivers.json")["drivers"])
    df["name"] = df["fn"] + " " + df["ln"]
    df["color"] = "#" + df["tc"]
    return df.rename(columns={"driver": "code", "dn": "number"})[["code", "name", "team", "number", "color"]]


def save_processed(df, name, processed_dir=PROCESSED_DIR):
    """Save a DataFrame as ``<processed_dir>/<name>.csv`` and return the path."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
