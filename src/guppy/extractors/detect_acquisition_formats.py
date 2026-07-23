import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _classify_csv_file(path: str) -> str:
    """
    Classify a single CSV file as belonging to one of three modalities.

    Parameters
    ----------
    path : str
        Absolute path to a CSV file.

    Returns
    -------
    str
        One of ``"doric"``, ``"npm"``, or ``"csv"``.
    """
    df = pd.read_csv(path, header=None, nrows=2, index_col=False, dtype=str)
    df = df.dropna(axis=1, how="all")
    header_values = np.array(df).flatten()
    non_numeric = [value for value in header_values if not _is_float(value)]

    # Doric CSV files have a 2-line all-string header (metadata + units rows) with no
    # numeric values in the first two rows at all.
    if len(non_numeric) == len(header_values):
        return "doric"

    # File has string headers (or numeric-only headers) — inspect column names to distinguish npm from csv.
    df = pd.read_csv(path, index_col=False)
    column_names = list(df.columns)

    # Doric v2 files store numeric values as column headers; treat them as headerless.
    if all(_is_float(name) for name in column_names):
        df = pd.read_csv(path, header=None)
        columns = np.array(list(df.columns), dtype=str)
    else:
        columns = np.array(column_names, dtype=str)

    if len(columns) == 1:
        if columns[0].lower() != "timestamps":
            message = (
                f"CSV file '{path}' has 1 column named '{columns[0]}', but the only-supported "
                "single-column CSV format requires the column to be named 'timestamps' (case-insensitive)."
            )
            logger.error(message)
            raise ValueError(message)
        return "csv"
    elif len(columns) == 3:
        required_columns = np.array(["timestamps", "data", "sampling_rate"])
        lowercase_columns = np.char.lower(columns)
        if (np.sort(required_columns) == np.sort(lowercase_columns)).all():
            return "csv"
        message = (
            f"CSV file '{path}' has columns {list(columns)}, but the 3-column CSV format "
            "requires column names 'timestamps', 'data', 'sampling_rate' (case-insensitive)."
        )
        logger.error(message)
        raise ValueError(message)
    elif len(columns) >= 2:
        return "npm"
    else:
        message = (
            f"CSV file '{path}' has {len(columns)} columns, which is not a recognized layout. "
            "Expected 1 column ('timestamps'), 2 columns (NPM event/data), or 3 columns "
            "('timestamps', 'data', 'sampling_rate')."
        )
        logger.error(message)
        raise ValueError(message)


def _is_float(value: object) -> bool:
    """Return True if *value* can be interpreted as a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_event_csv(path: str) -> bool:
    """
    Return True if the CSV file is an event_csv: a single column named 'timestamps'.

    Parameters
    ----------
    path : str
        Absolute path to a CSV file.

    Returns
    -------
    bool
    """
    df = pd.read_csv(path, nrows=0, index_col=False)
    columns = list(df.columns)
    return len(columns) == 1 and columns[0].lower() == "timestamps"


def detect_acquisition_formats(folder_path: str) -> set[str]:
    """
    Detect all acquisition formats present in a session folder.

    Returns every format that can supply data — photometry channels, TTL/event
    timestamps, or both.

    Parameters
    ----------
    folder_path : str
        Path to the session folder.

    Returns
    -------
    set of str
        Format strings for all sources found in the folder.
        Possible elements: ``"nwb"``, ``"tdt"``, ``"doric"``, ``"csv"``, ``"npm"``.
    """
    formats = set()

    # NWB .nwb files provide photometry channels via acquisition series
    if glob.glob(os.path.join(folder_path, "*.nwb")):
        formats.add("nwb")

    # TDT .tsq files provide both photometry stores and TTL event stores
    if glob.glob(os.path.join(folder_path, "*.tsq")):
        formats.add("tdt")

    # Doric .doric files provide both photometry channels and digital TTL channels
    if glob.glob(os.path.join(folder_path, "*.doric")):
        formats.add("doric")

    csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))

    # Multi-column CSV files can be NPM, Doric CSV exports, or 3-column data_csv files.
    # NPM demultiplexes its raw files in memory and never writes intermediates to the
    # folder, so each modality is detected independently of the others here.
    non_event_csv_paths = [csv_path for csv_path in csv_paths if not _is_event_csv(csv_path)]
    if non_event_csv_paths:
        labels = {_classify_csv_file(csv_path) for csv_path in non_event_csv_paths}
        if "npm" in labels:
            formats.add("npm")
        if "doric" in labels:
            formats.add("doric")
        if "csv" in labels:
            formats.add("csv")

    # Single-column timestamp CSVs are genuine external TTL files read by
    # CsvRecordingExtractor. NpmRecordingExtractor owns its own event streams in
    # memory, so single-column files no longer originate from NPM processing.
    if any(_is_event_csv(csv_path) for csv_path in csv_paths):
        formats.add("csv")

    return formats
