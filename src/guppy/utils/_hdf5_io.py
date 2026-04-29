"""Shared h5py-native HDF5 read/write helpers.

These helpers implement GuPPy's "one file per event, multiple keyed datasets
per file" convention. Used by both the extractor layer (raw store dumps) and
the analysis layer (preprocessing, z-score, transients, etc.).
"""

import logging
import os
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def read_hdf5(event, filepath, key):
    if event:
        event = event.replace("\\", "_")
        event = event.replace("/", "_")
        op = os.path.join(filepath, event + ".hdf5")
    else:
        op = filepath

    if os.path.exists(op):
        with h5py.File(op, "r") as f:
            arr = np.asarray(f[key])
    else:
        message = f"HDF5 file '{op}' does not exist (event={event!r}, key={key!r})."
        logger.error(message)
        raise FileNotFoundError(message)

    return arr


def write_hdf5(data: Any, event: str, filepath: str, key: str) -> None:
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    op = os.path.join(filepath, event + ".hdf5")

    if not os.path.exists(op):
        with h5py.File(op, "w") as f:
            if isinstance(data, np.ndarray):
                f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
            else:
                f.create_dataset(key, data=data)
    else:
        with h5py.File(op, "r+") as f:
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
