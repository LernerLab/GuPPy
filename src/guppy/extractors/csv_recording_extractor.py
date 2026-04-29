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

    @classmethod
    def discover_events_and_flags(cls, folder_path) -> tuple[list[str], list[str]]:
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
        flag_arr = []
        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            if ext != "csv":
                raise ValueError(f"Only .csv files are supported by CsvRecordingExtractor; got '{path[i]}'.")
            df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
            df = df.dropna(axis=1, how="all")
            df_arr = np.array(df).flatten()
            check_all_str = []
            for element in df_arr:
                try:
                    float(element)
                except:
                    check_all_str.append(i)
            if len(check_all_str) == len(df_arr):
                raise ValueError(
                    f"CSV file '{path[i]}' appears to be a Doric .csv (all-string header rows). "
                    "CsvRecordingExtractor only supports standard .csv files; use the Doric extractor instead."
                )
            df = pd.read_csv(path[i], index_col=False)

            _, value = cls._check_header(df)

            # check dataframe structure and read data accordingly
            if len(value) > 0:
                columns_isstr = False
                df = pd.read_csv(path[i], header=None)
                cols = np.array(list(df.columns), dtype=str)
            else:
                df = df
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)
            # check the structure of dataframe and assign flag to the type of file
            if len(cols) == 1:
                if cols[0].lower() != "timestamps":
                    message = (
                        f"CSV file '{path[i]}' has 1 column named '{cols[0]}', but the only-supported "
                        "single-column CSV format requires the column to be named 'timestamps' (lower case)."
                    )
                    logger.error(message)
                    raise ValueError(message)
                else:
                    flag = "event_csv"
            elif len(cols) == 3:
                arr1 = np.array(["timestamps", "data", "sampling_rate"])
                arr2 = np.char.lower(np.array(cols))
                if (np.sort(arr1) == np.sort(arr2)).all() == False:
                    message = (
                        f"CSV file '{path[i]}' has columns {list(cols)}, but the 3-column CSV format "
                        "requires column names 'timestamps', 'data', 'sampling_rate' (all lower case)."
                    )
                    logger.error(message)
                    raise ValueError(message)
                else:
                    flag = "data_csv"
            elif len(cols) >= 2:
                raise ValueError(
                    f"CSV file '{path[i]}' has {len(cols)} columns {list(cols)}, which matches the "
                    "Neurophotometrics (NPM) layout. Set 'Acquisition System' to 'NPM' in the "
                    "Input Parameters GUI before re-running the pipeline."
                )

            flag_arr.append(flag)
            logger.info(flag)
            name = os.path.basename(path[i]).split(".")[0]
            event_from_filename.append(name)

        logger.info("Importing of csv file is done.")
        return event_from_filename, flag_arr

    def __init__(self, folder_path):
        self.folder_path = folder_path

    @staticmethod
    def _check_header(df):
        arr = list(df.columns)
        check_float = []
        for i in arr:
            try:
                check_float.append(float(i))
            except:
                pass

        return arr, check_float

    def _read_csv(self, event):
        logger.debug(f"Trying to read data for {event} from csv file.")
        csv_path = os.path.join(self.folder_path, event + ".csv")
        if not os.path.exists(csv_path):
            message = f"No CSV file found for event '{event}' at '{csv_path}'."
            logger.error(message)
            raise FileNotFoundError(message)

        df = pd.read_csv(csv_path, index_col=False)
        return df

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        output_dicts = []
        for event in events:
            dataframe = self._read_csv(event=event)
            columns_lowercase = [col.lower() for col in dataframe.columns]
            if "data" in columns_lowercase:
                output_dicts.append(
                    {
                        "storename": event,
                        "timestamps": dataframe["timestamps"].dropna().to_numpy(),
                        "data": dataframe["data"].dropna().to_numpy(),
                        "sampling_rate": dataframe["sampling_rate"].dropna().to_numpy()[:1],
                    }
                )
            else:
                output_dicts.append(
                    {
                        "storename": event,
                        "timestamps": dataframe["timestamps"].dropna().to_numpy(),
                    }
                )
        return output_dicts

    def stub(self, *, folder_path, duration_in_seconds=1.0):
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
        for output_dict in output_dicts:
            storename = output_dict["storename"]
            for key, value in output_dict.items():
                if key == "storename":
                    continue
                write_hdf5(value, storename, outputPath, key)
