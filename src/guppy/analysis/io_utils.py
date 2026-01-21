import fnmatch
import glob
import logging
import os
import re

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


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


# check if dealing with TDT files or csv files
def check_TDT(filepath):
    path = glob.glob(os.path.join(filepath, "*.tsq"))
    if len(path) > 0:
        return True
    else:
        return False


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


def get_coords(filepath, name, tsNew, removeArtifacts):  # TODO: Make less redundant with fetchCoords
    if removeArtifacts == True:
        coords = fetchCoords(filepath, name, tsNew)
    else:
        dt = tsNew[1] - tsNew[0]
        coords = np.array([[tsNew[0] - dt, tsNew[-1] + dt]])
    return coords


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


# for combining data, reading storeslist file from both data and create a new storeslist array
def check_storeslistfile(folderNames):
    storesList = np.array([[], []])
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
                ),
                axis=1,
            )

    storesList = np.unique(storesList, axis=1)

    return storesList


def get_control_and_signal_channel_names(storesList):
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    channels_arr = []
    for i in range(names_for_storenames.shape[0]):
        if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
            channels_arr.append(names_for_storenames[i])

    channels_arr = sorted(channels_arr, key=str.casefold)
    try:
        channels_arr = np.asarray(channels_arr).reshape(2, -1)
    except:
        logger.error("Error in saving stores list file or spelling mistake for control or signal")
        raise Exception("Error in saving stores list file or spelling mistake for control or signal")

    return channels_arr


# function to read h5 file and make a dataframe from it
def read_Df(filepath, event, name):
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df


def make_dir_for_cross_correlation(filepath):
    op = os.path.join(filepath, "cross_correlation_output")
    if not os.path.exists(op):
        os.mkdir(op)
    return op
