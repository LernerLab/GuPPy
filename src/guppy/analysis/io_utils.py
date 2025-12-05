import fnmatch
import glob
import logging
import os
import re

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# Category: Analysis
# Reason: Utility function for path filtering - pure data transformation with no GUI or orchestration
def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


# Category: Analysis
# Reason: File system utility for case-insensitive file discovery - pure I/O helper with no orchestration
# find files by ignoring the case sensitivity
def find_files(path, glob_path, ignore_case=False):
    rule = (
        re.compile(fnmatch.translate(glob_path), re.IGNORECASE)
        if ignore_case
        else re.compile(fnmatch.translate(glob_path))
    )

    no_bytes_path = os.listdir(os.path.expanduser(path))
    str_path = []

    # converting byte object to string
    for x in no_bytes_path:
        try:
            str_path.append(x.decode("utf-8"))
        except:
            str_path.append(x)
    return [os.path.join(path, n) for n in str_path if rule.match(n)]


# Category: Analysis
# Reason: Simple file type detection utility - pure file system check with no orchestration
# check if dealing with TDT files or csv files
def check_TDT(filepath):
    path = glob.glob(os.path.join(filepath, "*.tsq"))
    if len(path) > 0:
        return True
    else:
        return False


# Category: Analysis
# Reason: I/O utility function for reading HDF5 files - pure file access with no business logic or orchestration
# function to read hdf5 file
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
        logger.error(f"{event}.hdf5 file does not exist")
        raise Exception("{}.hdf5 file does not exist".format(event))

    return arr


# Category: Analysis
# Reason: I/O utility function for writing HDF5 files - pure file access with no business logic or orchestration
# function to write hdf5 file
def write_hdf5(data, event, filepath, key):
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


# Category: Analysis
# Reason: Validation utility - checks file naming conventions and returns structured path array with no orchestration
# function to check if the naming convention for saving storeslist file was followed or not
def decide_naming_convention(filepath):
    path_1 = find_files(filepath, "control_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'control*'))

    path_2 = find_files(filepath, "signal_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'signal*'))

    path = sorted(path_1 + path_2, key=str.casefold)
    if len(path) % 2 != 0:
        logger.error("There are not equal number of Control and Signal data")
        raise Exception("There are not equal number of Control and Signal data")

    path = np.asarray(path).reshape(2, -1)

    return path


# Category: Analysis
# Reason: I/O utility that loads artifact coordinates from .npy file or provides default - pure file loading with simple logic
# function to read coordinates file which was saved by selecting chunks for artifacts removal
def fetchCoords(filepath, naming, data):

    path = os.path.join(filepath, "coordsForPreProcessing_" + naming + ".npy")

    if not os.path.exists(path):
        coords = np.array([0, data[-1]])
    else:
        coords = np.load(os.path.join(filepath, "coordsForPreProcessing_" + naming + ".npy"))[:, 0]

    if coords.shape[0] % 2 != 0:
        logger.error("Number of values in coordsForPreProcessing file is not even.")
        raise Exception("Number of values in coordsForPreProcessing file is not even.")

    coords = coords.reshape(-1, 2)

    return coords


# Category: Routing
# Reason: Organizes output folders for data combination - loops through numbered outputs and groups related folders
def get_all_stores_for_combining_data(folderNames):
    op = []
    for i in range(100):
        temp = []
        match = r"[\s\S]*" + "_output_" + str(i)
        for j in folderNames:
            temp.append(re.findall(match, j))
        temp = sorted(list(np.concatenate(temp).flatten()), key=str.casefold)
        if len(temp) > 0:
            op.append(temp)

    return op
