import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _classify_csv_file(path):
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
    df_arr = np.array(df).flatten()
    non_numeric = [el for el in df_arr if not _is_float(el)]

    # Doric CSV files have a 2-line all-string header (metadata + units rows) with no
    # numeric values in the first two rows at all.
    if len(non_numeric) == len(df_arr):
        return "doric"

    # File has string headers (or numeric-only headers) — inspect column names to distinguish npm from csv.
    df = pd.read_csv(path, index_col=False)
    colnames = list(df.columns)

    # Doric v2 files store numeric values as column headers; treat them as headerless.
    if all(_is_float(c) for c in colnames):
        df = pd.read_csv(path, header=None)
        cols = np.array(list(df.columns), dtype=str)
    else:
        cols = np.array(colnames, dtype=str)

    if len(cols) == 1:
        if cols[0].lower() != "timestamps":
            logger.error("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
            raise Exception("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
        return "csv"
    elif len(cols) == 3:
        arr1 = np.array(["timestamps", "data", "sampling_rate"])
        arr2 = np.char.lower(cols)
        if (np.sort(arr1) == np.sort(arr2)).all():
            return "csv"
        logger.error(
            "\033[1m" + "Column names should be timestamps, data and sampling_rate (all lower-cases)" + "\033[0m"
        )
        raise Exception(
            "\033[1m" + "Column names should be timestamps, data and sampling_rate (all lower-cases)" + "\033[0m"
        )
    elif len(cols) >= 2:
        return "npm"
    else:
        logger.error("Number of columns in csv file does not make sense.")
        raise Exception("Number of columns in csv file does not make sense.")


def _is_float(value):
    """Return True if *value* can be interpreted as a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_event_csv(path):
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
    cols = list(df.columns)
    return len(cols) == 1 and cols[0].lower() == "timestamps"


def detect_acquisition_formats(folder_path):
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
        Possible elements: ``"tdt"``, ``"doric"``, ``"csv"``, ``"npm"``.
    """
    formats = set()

    # TDT .tsq files provide both photometry stores and TTL event stores
    if glob.glob(os.path.join(folder_path, "*.tsq")):
        formats.add("tdt")

    # Doric .doric files provide both photometry channels and digital TTL channels
    if glob.glob(os.path.join(folder_path, "*.doric")):
        formats.add("doric")

    csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))

    # Multi-column CSV files can be NPM, Doric CSV exports, or 3-column data_csv files
    non_event_csv_paths = [p for p in csv_paths if not _is_event_csv(p)]
    has_npm_data = False
    if non_event_csv_paths:
        labels = {_classify_csv_file(p) for p in non_event_csv_paths}
        if "npm" in labels:
            formats.add("npm")
            has_npm_data = True
        if "doric" in labels:
            formats.add("doric")
        if not has_npm_data and "csv" in labels:
            formats.add("csv")

    # Single-column timestamp CSVs provide external event timestamps. When NPM data is
    # present, suppress only the NPM-generated split-event files (named event*.csv), since
    # those are owned by NpmRecordingExtractor. External event CSVs with other names are
    # still detected as "csv" and handled by CsvRecordingExtractor.
    event_csv_paths = [p for p in csv_paths if _is_event_csv(p)]
    if has_npm_data:
        external_event_csv_paths = [p for p in event_csv_paths if not os.path.basename(p).startswith("event")]
    else:
        external_event_csv_paths = event_csv_paths
    if external_event_csv_paths:
        formats.add("csv")

    return formats
