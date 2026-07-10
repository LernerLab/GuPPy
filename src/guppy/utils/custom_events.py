"""Parse, validate, and write user-supplied custom event timestamps.

This module backs the optional "Import Custom Events" GUI step. A custom event
is written as a GuPPy-compatible event CSV — a single column with the header
``timestamps`` and one timestamp (in seconds) per row — into a session folder.
``detect_acquisition_formats`` and ``CsvRecordingExtractor`` then surface it as a
store named after the file, with no further configuration.

The helpers here are internal (called by the GUI and exercised by tests); they
are not part of the advertised pipeline API.
"""

import os
import re

import pandas as pd

EVENT_CSV_COLUMN = "timestamps"


def parse_pasted_timestamps(text: str) -> list[float]:
    """Parse pasted timestamp text into a list of floats.

    The input is whatever the user copied out of a spreadsheet column (or typed),
    so tokens may be separated by newlines, commas, or other whitespace. Blank
    tokens are dropped. Order is preserved; the values are not sorted.

    Parameters
    ----------
    text : str
        Raw pasted text. Empty or whitespace-only input yields an empty list.

    Returns
    -------
    list of float
        Parsed timestamps in the order they appeared.

    Raises
    ------
    ValueError
        If a non-blank token cannot be parsed as a float; the offending token
        is named in the message.
    """
    tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
    timestamps = []
    for token in tokens:
        try:
            timestamps.append(float(token))
        except ValueError:
            raise ValueError(
                f"Could not parse {token!r} as a number. Paste one timestamp (in seconds) per row, "
                "with no header or text."
            )
    return timestamps


def is_sorted(timestamps: list[float]) -> bool:
    """Return whether ``timestamps`` is in non-decreasing order.

    Parameters
    ----------
    timestamps : list of float
        Timestamps to check.

    Returns
    -------
    bool
        True if every value is greater than or equal to the previous one.
    """
    return all(earlier <= later for earlier, later in zip(timestamps, timestamps[1:]))


def write_custom_event_csv(*, name: str, timestamps: list[float], folder_path: str, overwrite: bool = False) -> str:
    """Write timestamps as a GuPPy-compatible event CSV named after the event.

    The file is written as ``<name>.csv`` into ``folder_path`` with a single
    ``timestamps`` column. The filename (minus ``.csv``) becomes the store_id
    surfaced in the Label Stores GUI.

    Parameters
    ----------
    name : str
        Event name; also the output filename stem. Must be non-empty and contain
        no path separators.
    timestamps : list of float
        Event timestamps in seconds.
    folder_path : str
        Absolute path to the session directory the CSV is written into.
    overwrite : bool, optional
        If False (default) and a file named ``<name>.csv`` already exists, raise
        ``FileExistsError``. If True, replace it.

    Returns
    -------
    str
        Absolute path to the written CSV.

    Raises
    ------
    ValueError
        If ``name`` is empty or contains a path separator.
    FileExistsError
        If ``<name>.csv`` already exists and ``overwrite`` is False.
    """
    if not name or not name.strip():
        raise ValueError("Event name must be a non-empty string.")
    if os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"Event name {name!r} must not contain a path separator.")

    csv_path = os.path.join(folder_path, f"{name}.csv")
    if os.path.exists(csv_path) and not overwrite:
        raise FileExistsError(
            f"An event named {name!r} already exists at {csv_path}. Enable overwrite to replace it, "
            "or choose a different name."
        )

    pd.DataFrame({EVENT_CSV_COLUMN: timestamps}).to_csv(csv_path, index=False)
    return csv_path
