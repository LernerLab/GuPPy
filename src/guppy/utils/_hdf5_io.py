"""Shared h5py-native HDF5 read/write helpers.

These helpers implement GuPPy's "one file per event, multiple keyed datasets
per file" convention. Used by both the extractor layer (raw store dumps) and
the analysis layer (preprocessing, z-score, transients, etc.).
"""

import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def read_hdf5(event: str, filepath: str, key: str) -> np.ndarray:
    if event:
        event = event.replace("\\", "_")
        event = event.replace("/", "_")
        hdf5_path = os.path.join(filepath, event + ".hdf5")
    else:
        hdf5_path = filepath

    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            data = np.asarray(hdf5_file[key])
    else:
        message = f"HDF5 file '{hdf5_path}' does not exist (event={event!r}, key={key!r})."
        logger.error(message)
        raise FileNotFoundError(message)

    return data


def write_hdf5(data: np.ndarray | float | int | str | bool, storename: str, output_path: str, key: str) -> None:
    storename = storename.replace("\\", "_")
    storename = storename.replace("/", "_")
    hdf5_path = os.path.join(output_path, storename + ".hdf5")

    if not os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "w") as hdf5_file:
            if isinstance(data, np.ndarray):
                hdf5_file.create_dataset(key, data=data, maxshape=(None,), chunks=True)
            else:
                hdf5_file.create_dataset(key, data=data)
    else:
        with h5py.File(hdf5_path, "r+") as hdf5_file:
            if key in list(hdf5_file.keys()):
                if isinstance(data, np.ndarray):
                    hdf5_file[key].resize(data.shape)
                    dataset = hdf5_file[key]
                    dataset[:] = data
                else:
                    dataset = hdf5_file[key]
                    dataset[()] = data
            else:
                if isinstance(data, np.ndarray):
                    hdf5_file.create_dataset(key, data=data, maxshape=(None,), chunks=True)
                else:
                    hdf5_file.create_dataset(key, data=data)
