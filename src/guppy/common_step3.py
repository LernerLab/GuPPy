import glob
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from itertools import repeat

import h5py
import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16

logger = logging.getLogger(__name__)

# function to write data to a hdf5 file
def write_hdf5(data, event, filepath, key):

    # replacing \\ or / in storenames with _ (to avoid errors while saving data)
    event = event.replace("\\", "_")
    event = event.replace("/", "_")

    op = os.path.join(filepath, event + ".hdf5")

    # if file does not exist create a new file
    if not os.path.exists(op):
        with h5py.File(op, "w") as f:
            if type(data) is np.ndarray:
                f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
            else:
                f.create_dataset(key, data=data)

    # if file already exists, append data to it or add a new key to it
    else:
        with h5py.File(op, "r+") as f:
            if key in list(f.keys()):
                if type(data) is np.ndarray:
                    f[key].resize(data.shape)
                    arr = f[key]
                    arr[:] = data
                else:
                    arr = f[key]
                    arr = data
            else:
                if type(data) is np.ndarray:
                    f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
                else:
                    f.create_dataset(key, data=data)