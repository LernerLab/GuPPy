import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from guppy.extractors import BaseRecordingExtractor
from guppy.extractors.detect_acquisition_formats import _classify_csv_file
from guppy.utils._hdf5_io import write_hdf5

logger = logging.getLogger(__name__)


class CsvRecordingExtractor(BaseRecordingExtractor):
    """
    Extractor for fiber photometry data stored in standard CSV files.

    Supports two CSV layouts:

    * **data_csv** — three columns: ``timestamps``, ``data``, ``sampling_rate``.
    * **event_csv** — one column: ``timestamps``.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the CSV files.
    """

    @classmethod
    def discover_events_and_flags(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from CSV files.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing CSV files.

        Returns
        -------
        events : list of str
            Names of all events/stores available in the dataset.
        flags : list of str
            Format indicators or file type flags.
        """
        logger.debug("If it exists, importing either NPM or Doric or csv file based on the structure of file")
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        # Only process files classified as standard CSV (event_csv or data_csv).
        # Skips NPM multi-column files and Doric CSV files when coexisting in the same folder.
        path = [p for p in path if _classify_csv_file(p) == "csv"]
        path = sorted(list(set(path)))
        flag = "None"
        event_from_filename = []
        flags = []
        for i in range(len(path)):
            extension = os.path.basename(path[i]).split(".")[-1]
            if extension != "csv":
                raise ValueError(f"Only .csv files are supported by CsvRecordingExtractor; got '{path[i]}'.")
            df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
            df = df.dropna(axis=1, how="all")
            header_values = np.array(df).flatten()
            check_all_str = []
            for element in header_values:
                try:
                    float(element)
                except:
                    check_all_str.append(i)
            if len(check_all_str) == len(header_values):
                raise ValueError(
                    f"CSV file '{path[i]}' appears to be a Doric .csv (all-string header rows). "
                    "CsvRecordingExtractor only supports standard .csv files; use the Doric extractor instead."
                )
            df = pd.read_csv(path[i], index_col=False)

            _, numeric_headers = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(numeric_headers) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                columns = np.array(list(df.columns), dtype=str)
            else:
                df = df
                columns_isstr = True
                columns = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(columns) == 1:
                if columns[0].lower() != "timestamps":
                    message = (
                        f"CSV file '{path[i]}' has 1 column named '{columns[0]}', but the only-supported "
                        "single-column CSV format requires the column to be named 'timestamps' (lower case)."
                    )
                    logger.error(message)
                    raise ValueError(message)
                else:
                    flag = "event_csv"
            elif len(columns) == 3:
                required_columns = np.array(["timestamps", "data", "sampling_rate"])
                lowercase_columns = np.char.lower(np.array(columns))
                if (np.sort(required_columns) == np.sort(lowercase_columns)).all() == False:
                    message = (
                        f"CSV file '{path[i]}' has columns {list(columns)}, but the 3-column CSV format "
                        "requires column names 'timestamps', 'data', 'sampling_rate' (all lower case)."
                    )
                    logger.error(message)
                    raise ValueError(message)
                else:
                    flag = "data_csv"
            elif len(columns) >= 2:
                raise ValueError(
                    f"CSV file '{path[i]}' has {len(columns)} columns {list(columns)}, which matches the "
                    "Neurophotometrics (NPM) layout. Set 'Acquisition System' to 'NPM' in the "
                    "Input Parameters GUI before re-running the pipeline."
                )

            flags.append(flag)
            logger.info(flag)
            name = os.path.basename(path[i]).split(".")[0]
            event_from_filename.append(name)

        logger.info("Importing of csv file is done.")
        return event_from_filename, flags

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @staticmethod
    def _check_header(df: pd.DataFrame) -> tuple[list[str], list[float]]:
        columns = list(df.columns)
        check_float = []
        for column in columns:
            try:
                check_float.append(float(column))
            except:
                pass

        return columns, check_float

    def count_samples(self, *, event: str) -> int:
        """Return the number of data rows in ``<event>.csv`` (excludes header)."""
        csv_path = os.path.join(self.folder_path, event + ".csv")
        if not os.path.exists(csv_path):
            return 0
        with open(csv_path, "rb") as file:
            total_lines = sum(1 for _ in file)
        return max(0, total_lines - 1)

    def _read_csv(self, event: str) -> pd.DataFrame:
        logger.debug(f"Trying to read data for {event} from csv file.")
        csv_path = os.path.join(self.folder_path, event + ".csv")
        if not os.path.exists(csv_path):
            message = f"No CSV file found for event '{event}' at '{csv_path}'."
            logger.error(message)
            raise FileNotFoundError(message)

        df = pd.read_csv(csv_path, index_col=False)
        return df

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data from CSV files for the specified events.

        Parameters
        ----------
        events : list of str
            Event names to read. Each name must correspond to a ``<name>.csv``
            file in ``folder_path``.
        outputPath : str
            Path to the output directory (unused by this extractor; required by
            the base-class interface).

        Returns
        -------
        list of dict
            One dictionary per event. Data CSVs produce dicts with keys
            ``store_id``, ``timestamps``, ``data``, and ``sampling_rate``;
            event CSVs produce dicts with keys ``store_id`` and ``timestamps``.
        """
        output_dicts = []
        for event in events:
            dataframe = self._read_csv(event=event)
            columns_lowercase = [column.lower() for column in dataframe.columns]
            if "data" in columns_lowercase:
                output_dicts.append(
                    {
                        "store_id": event,
                        "timestamps": dataframe["timestamps"].dropna().to_numpy(),
                        "data": dataframe["data"].dropna().to_numpy(),
                        "sampling_rate": dataframe["sampling_rate"].dropna().to_numpy()[:1],
                    }
                )
            else:
                output_dicts.append(
                    {
                        "store_id": event,
                        "timestamps": dataframe["timestamps"].dropna().to_numpy(),
                    }
                )
        return output_dicts

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Create a stubbed copy of the CSV folder with truncated data files.

        Copies the entire folder to `folder_path`, then replaces each CSV file
        with a version truncated to `duration_in_seconds`. The cutoff timestamp
        is computed as the first timestamp in the first data CSV plus
        `duration_in_seconds`. Both data CSVs (3-column) and event CSVs
        (1-column) are filtered to rows at or before the cutoff.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed folder. Created if it does
            not exist; overwritten if it already exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain in seconds. Default is 1.0.
        """
        folder_path = Path(folder_path)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(self.folder_path, folder_path)

        event_names, flags = CsvRecordingExtractor.discover_events_and_flags(self.folder_path)

        first_data_timestamp = None
        for event_name, flag in zip(event_names, flags):
            if flag == "data_csv":
                dataframe = pd.read_csv(Path(self.folder_path) / f"{event_name}.csv", index_col=False)
                first_data_timestamp = dataframe["timestamps"].iloc[0]
                break

        cutoff_timestamp = first_data_timestamp + duration_in_seconds

        for event_name, flag in zip(event_names, flags):
            csv_path = folder_path / f"{event_name}.csv"
            dataframe = pd.read_csv(csv_path, index_col=False)
            dataframe = dataframe[dataframe["timestamps"] <= cutoff_timestamp]
            dataframe.to_csv(csv_path, index=False)

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 files.

        Parameters
        ----------
        output_dicts : list of dict
            Data dictionaries as returned by :meth:`read`.
        outputPath : str
            Path to the output directory where HDF5 files are written.
        """
        for output_dict in output_dicts:
            store_id = output_dict["store_id"]
            for key, value in output_dict.items():
                if key == "store_id":
                    continue
                write_hdf5(value, store_id, outputPath, key)
