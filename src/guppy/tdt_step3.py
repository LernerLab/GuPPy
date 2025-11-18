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

from guppy.common_step3 import write_hdf5

logger = logging.getLogger(__name__)

# function to execute readtev function using multiprocessing to make it faster
def execute_readtev(data, filepath, event, outputPath, numProcesses=mp.cpu_count()):

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
    # p = mp.Pool(mp.cpu_count())
    # p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
    # p.close()
    # p.join()
    logger.info("Time taken = {0:.5f}".format(time.time() - start))


# function to read tev file
def readtev(data, filepath, event, outputPath):

    logger.debug("Reading data for event {} ...".format(event))
    tevfilepath = glob.glob(os.path.join(filepath, "*.tev"))
    if len(tevfilepath) > 1:
        raise Exception("Two tev files are present at the location.")
    else:
        tevfilepath = tevfilepath[0]

    data["name"] = np.asarray(data["name"], dtype=str)

    allnames = np.unique(data["name"])

    index = []
    for i in range(len(allnames)):
        length = len(str(allnames[i]))
        if length < 4:
            index.append(i)

    allnames = np.delete(allnames, index, 0)

    eventNew = np.array(list(event))

    # logger.info(allnames)
    # logger.info(eventNew)
    row = ismember(data["name"], event)

    if sum(row) == 0:
        logger.info("\033[1m" + "Requested store name " + event + " not found (case-sensitive)." + "\033[0m")
        logger.info("\033[1m" + "File contains the following TDT store names:" + "\033[0m")
        logger.info("\033[1m" + str(allnames) + "\033[0m")
        logger.info("\033[1m" + "TDT store name " + str(event) + " not found." + "\033[0m")
        import_csv(filepath, event, outputPath)

        return 0

    allIndexesWhereEventIsPresent = np.where(row == 1)
    first_row = allIndexesWhereEventIsPresent[0][0]

    formatNew = data["format"][first_row] + 1

    table = np.array(
        [
            [0, 0, 0, 0],
            [0, "float", 1, np.float32],
            [0, "long", 1, np.int32],
            [0, "short", 2, np.int16],
            [0, "byte", 4, np.int8],
        ]
    )

    S = dict()

    S["storename"] = str(event)
    S["sampling_rate"] = data["frequency"][first_row]
    S["timestamps"] = np.asarray(data["timestamp"][allIndexesWhereEventIsPresent[0]])
    S["channels"] = np.asarray(data["chan"][allIndexesWhereEventIsPresent[0]])

    fp_loc = np.asarray(data["fp_loc"][allIndexesWhereEventIsPresent[0]])
    data_size = np.asarray(data["size"])

    if formatNew != 5:
        nsample = (data_size[first_row,] - 10) * int(table[formatNew, 2])
        S["data"] = np.zeros((len(fp_loc), nsample))
        for i in range(0, len(fp_loc)):
            with open(tevfilepath, "rb") as fp:
                fp.seek(fp_loc[i], os.SEEK_SET)
                S["data"][i, :] = np.fromfile(fp, dtype=table[formatNew, 3], count=nsample).reshape(
                    1, nsample, order="F"
                )
                # S['data'] = S['data'].swapaxes()
        S["npoints"] = nsample
    else:
        S["data"] = np.asarray(data["strobe"][allIndexesWhereEventIsPresent[0]])
        S["npoints"] = 1
        S["channels"] = np.tile(1, (S["data"].shape[0],))

    S["data"] = (S["data"].T).reshape(-1, order="F")

    save_dict_to_hdf5(S, event, outputPath)

    check_data(S, filepath, event, outputPath)

    logger.info("Data for event {} fetched and stored.".format(event))

# check if a particular element is there in an array or not
def ismember(arr, element):
    res = [1 if i == element else 0 for i in arr]
    return np.asarray(res)


# function to save data read from tev file to hdf5 file
def save_dict_to_hdf5(S, event, outputPath):
    write_hdf5(S["storename"], event, outputPath, "storename")
    write_hdf5(S["sampling_rate"], event, outputPath, "sampling_rate")
    write_hdf5(S["timestamps"], event, outputPath, "timestamps")

    write_hdf5(S["data"], event, outputPath, "data")
    write_hdf5(S["npoints"], event, outputPath, "npoints")
    write_hdf5(S["channels"], event, outputPath, "channels")


# function to check event data (checking whether event timestamps belongs to same event or multiple events)
def check_data(S, filepath, event, outputPath):
    # logger.info("Checking event storename data for creating multiple event names from single event storename...")
    new_event = event.replace("\\", "")
    new_event = event.replace("/", "")
    diff = np.diff(S["data"])
    arr = np.full(diff.shape[0], 1)

    storesList = np.genfromtxt(os.path.join(outputPath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)

    if diff.shape[0] == 0:
        return 0

    if S["sampling_rate"] == 0 and np.all(diff == diff[0]) == False:
        logger.info("\033[1m" + "Data in event {} belongs to multiple behavior".format(event) + "\033[0m")
        logger.debug(
            "\033[1m" + "Create timestamp files for individual new event and change the stores list file." + "\033[0m"
        )
        i_d = np.unique(S["data"])
        for i in range(i_d.shape[0]):
            new_S = dict()
            idx = np.where(S["data"] == i_d[i])[0]
            new_S["timestamps"] = S["timestamps"][idx]
            new_S["storename"] = new_event + str(int(i_d[i]))
            new_S["sampling_rate"] = S["sampling_rate"]
            new_S["data"] = S["data"]
            new_S["npoints"] = S["npoints"]
            new_S["channels"] = S["channels"]
            storesList = np.concatenate(
                (storesList, [[new_event + str(int(i_d[i]))], [new_event + "_" + str(int(i_d[i]))]]), axis=1
            )
            save_dict_to_hdf5(new_S, new_event + str(int(i_d[i])), outputPath)

        idx = np.where(storesList[0] == event)[0]
        storesList = np.delete(storesList, idx, axis=1)
        if not os.path.exists(os.path.join(outputPath, ".cache_storesList.csv")):
            os.rename(os.path.join(outputPath, "storesList.csv"), os.path.join(outputPath, ".cache_storesList.csv"))
        if idx.shape[0] == 0:
            pass
        else:
            np.savetxt(os.path.join(outputPath, "storesList.csv"), storesList, delimiter=",", fmt="%s")
        logger.info(
            "\033[1m"
            + "Timestamp files for individual new event are created \
	    			and the stores list file is changed."
            + "\033[0m"
        )