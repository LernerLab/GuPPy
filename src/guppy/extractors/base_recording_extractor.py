"""Base class for recording extractors."""

import logging
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import h5py
import numpy as np

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


def read_and_save_all_events(event_to_extractor, outputPath, numProcesses=mp.cpu_count()):
    events = list(event_to_extractor.keys())
    logger.info("Reading data for event {} ...".format(events))

    start = time.time()
    args = [(extractor, event, outputPath) for event, extractor in event_to_extractor.items()]
    with mp.Pool(numProcesses) as p:
        p.starmap(read_and_save_event, args)
    logger.info("Time taken = {0:.5f}".format(time.time() - start))
