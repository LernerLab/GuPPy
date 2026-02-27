"""Base class for recording extractors."""

import glob
import logging
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Any

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseRecordingExtractor(ABC):
    """
    Abstract base class for recording extractors.

    Defines the interface contract for reading and saving fiber photometry
    data from various acquisition formats (TDT, Doric, CSV, NPM, etc.).
    """

    @classmethod
    @abstractmethod
    def discover_events_and_flags(cls) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from data files.

        Returns
        -------
        events : list of str
            Names of all events/stores available in the dataset.
        flags : list of str
            Format indicators or file type flags.
        """
        # NOTE: This method signature is intentionally minimal and flexible.
        # Different formats have different discovery requirements:
        # - TDT/CSV/Doric: need only folder_path parameter
        # - NPM: needs folder_path, num_ch, and optional inputParameters for interleaved channels
        # Each child class defines its own signature with the parameters it needs.
        pass

    @abstractmethod
    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data from source files for specified events.

        Parameters
        ----------
        events : list of str
            List of event/store names to extract from the data.
        outputPath : str
            Path to the output directory.

        Returns
        -------
        list of dict
            List of dictionaries containing extracted data. Each dictionary
            represents one event/store and contains keys such as 'storename',
            'timestamps', 'data', 'sampling_rate', etc.
        """
        pass

    @abstractmethod
    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 format.

        Parameters
        ----------
        output_dicts : list of dict
            List of data dictionaries from read().
        outputPath : str
            Path to the output directory.
        """
        pass

    @staticmethod
    def _write_hdf5(data: Any, storename: str, output_path: str, key: str) -> None:
        """
        Write data to HDF5 file.

        Parameters
        ----------
        data : array-like
            Data to write to the HDF5 file.
        storename : str
            Name of the store/event.
        output_path : str
            Directory path where HDF5 file will be written.
        key : str
            Key name for this data field in the HDF5 file.
        """
        # Replace invalid characters in storename to avoid filesystem errors
        storename = storename.replace("\\", "_")
        storename = storename.replace("/", "_")

        filepath = os.path.join(output_path, storename + ".hdf5")

        # Create new file if it doesn't exist
        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                if isinstance(data, np.ndarray):
                    f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
                else:
                    f.create_dataset(key, data=data)
        # Append to existing file
        else:
            with h5py.File(filepath, "r+") as f:
                if key in list(f.keys()):
                    if isinstance(data, np.ndarray):
                        f[key].resize(data.shape)
                        arr = f[key]
                        arr[:] = data
                    else:
                        arr = f[key]
                        arr[()] = data
                else:
                    if isinstance(data, np.ndarray):
                        f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
                    else:
                        f.create_dataset(key, data=data)


def read_and_save_event(extractor, event, outputPath):
    output_dicts = extractor.read(events=[event], outputPath=outputPath)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)
    logger.info("Data for event {} fetched and stored.".format(event))


def read_and_save_all_events(extractor, events, outputPath, numProcesses=mp.cpu_count()):
    logger.info("Reading data for event {} ...".format(events))

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(read_and_save_event, zip(repeat(extractor), events, repeat(outputPath)))
    logger.info("Time taken = {0:.5f}".format(time.time() - start))


def detect_modality(folder_path):
    # Check for TDT .tsq files
    paths = glob.glob(os.path.join(folder_path, "*.tsq"))
    if len(paths) > 1:
        logger.error("Two tsq files are present at the location.")
        raise Exception("Two tsq files are present at the location.")
    elif len(paths) == 1:
        return "tdt"

    # Check for doric .doric files
    paths = glob.glob(os.path.join(folder_path, "*.doric"))
    if len(paths) >= 1:
        return "doric"

    # Check for .csv files which could indicate doric, csv, or npm modalities.
    paths = glob.glob(os.path.join(folder_path, "*.csv"))
    if len(paths) == 0:
        logger.error("\033[1m" + "No .tsq, .doric, or .csv files found to determine modality." + "\033[0m")
        raise Exception("No .tsq, .doric, or .csv files found to determine modality.")
    for path in paths:
        df = pd.read_csv(path, header=None, nrows=2, index_col=False, dtype=str)
        df = df.dropna(axis=1, how="all")
        df_arr = np.array(df).flatten()
        check_all_str = []
        for i, element in enumerate(df_arr):
            try:
                float(element)
            except:
                check_all_str.append(i)
        if len(check_all_str) == len(df_arr):
            return "doric"
        else:
            df = pd.read_csv(path, index_col=False)
            colnames = list(df.columns)
            value = []
            for i in colnames:
                try:
                    value.append(float(i))
                except:
                    pass

            # check dataframe structure and read data accordingly
            if len(value) > 0:
                columns_isstr = False
                df = pd.read_csv(path, header=None)
                cols = np.array(list(df.columns), dtype=str)
            else:
                columns_isstr = True
                cols = np.array(list(df.columns), dtype=str)

            # check the structure of dataframe and assign flag to the type of file
            if len(cols) == 1:
                if cols[0].lower() != "timestamps":
                    logger.error("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
                    raise Exception("\033[1m" + "Column name should be timestamps (all lower-cases)" + "\033[0m")
                else:
                    return "csv"
            elif len(cols) == 3:
                arr1 = np.array(["timestamps", "data", "sampling_rate"])
                arr2 = np.char.lower(np.array(cols))
                if (np.sort(arr1) == np.sort(arr2)).all() == False:
                    logger.error(
                        "\033[1m"
                        + "Column names should be timestamps, data and sampling_rate (all lower-cases)"
                        + "\033[0m"
                    )
                    raise Exception(
                        "\033[1m"
                        + "Column names should be timestamps, data and sampling_rate (all lower-cases)"
                        + "\033[0m"
                    )
                else:
                    return "csv"
            elif len(cols) >= 2:
                return "npm"
            else:
                logger.error("Number of columns in csv file does not make sense.")
                raise Exception("Number of columns in csv file does not make sense.")
