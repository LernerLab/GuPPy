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
from guppy.tdt_step3 import execute_readtev
from guppy.doric_step3 import execute_import_doric

logger = logging.getLogger(__name__)


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


def writeToFile(value: str):
    with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
        file.write(value)


# function to read event timestamps csv file.
def import_csv(filepath, event, outputPath):
    logger.debug("\033[1m" + "Trying to read data for {} from csv file.".format(event) + "\033[0m")
    if not os.path.exists(os.path.join(filepath, event + ".csv")):
        logger.error("\033[1m" + "No csv file found for event {}".format(event) + "\033[0m")
        raise Exception("\033[1m" + "No csv file found for event {}".format(event) + "\033[0m")

    df = pd.read_csv(os.path.join(filepath, event + ".csv"), index_col=False)
    data = df
    key = list(df.columns)

    if len(key) == 3:
        arr1 = np.array(["timestamps", "data", "sampling_rate"])
        arr2 = np.char.lower(np.array(key))
        if (np.sort(arr1) == np.sort(arr2)).all() == False:
            logger.error("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")
            raise Exception("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")

    if len(key) == 1:
        if key[0].lower() != "timestamps":
            logger.error("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")
            raise Exception("\033[1m" + "Column name should be timestamps" + "\033[0m")

    if len(key) != 3 and len(key) != 1:
        logger.error(
            "\033[1m"
            + "Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."
            + "\033[0m"
        )
        raise Exception(
            "\033[1m"
            + "Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."
            + "\033[0m"
        )

    for i in range(len(key)):
        write_hdf5(data[key[i]].dropna(), event, outputPath, key[i].lower())

    logger.info("\033[1m" + "Reading data for {} from csv file is completed.".format(event) + "\033[0m")

    return data, key


def execute_import_csv(filepath, event, outputPath, numProcesses=mp.cpu_count()):
    # logger.info("Reading data for event {} ...".format(event))

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(import_csv, zip(repeat(filepath), event, repeat(outputPath)))
    logger.info("Time taken = {0:.5f}".format(time.time() - start))





def find_string(regex, arr):
    for i in range(len(arr)):
        if regex.match(arr[i]):
            return i


# function to read data from 'tsq' and 'tev' files
def readRawData(inputParameters):

    logger.debug("### Reading raw data... ###")
    # get input parameters
    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    storesListPath = []
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            "Warning : # of cores parameter set is greater than the cores available \
			   available in your machine"
        )
        numProcesses = mp.cpu_count() - 1
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    writeToFile(str((storesListPath.shape[0] + 1) * 10) + "\n" + str(10) + "\n")
    step = 0
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        logger.debug(f"### Reading raw data for folder {folderNames[i]}")
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        modality = "doric"

        # read data corresponding to each storename selected by user while saving the storeslist file
        for j in range(len(storesListPath)):
            op = storesListPath[j]
            if os.path.exists(os.path.join(op, ".cache_storesList.csv")):
                storesList = np.genfromtxt(
                    os.path.join(op, ".cache_storesList.csv"), dtype="str", delimiter=","
                ).reshape(2, -1)
            else:
                storesList = np.genfromtxt(os.path.join(op, "storesList.csv"), dtype="str", delimiter=",").reshape(
                    2, -1
                )

            if modality == "tdt":
                execute_readtev(filepath, np.unique(storesList[0, :]), op, numProcesses)
            elif modality == "doric":
                execute_import_doric(filepath, storesList, modality, op)
            elif modality == "csv" or modality == "npm":
                execute_import_csv(filepath, np.unique(storesList[0, :]), op, numProcesses)
            else:
                raise ValueError("Modality not recognized. Please use 'tdt', 'csv', 'doric', or 'npm'.")

            writeToFile(str(10 + ((step + 1) * 10)) + "\n")
            step += 1
        logger.info(f"### Raw data for folder {folderNames[i]} fetched")
    logger.info("Raw data fetched and saved.")
    logger.info("#" * 400)


def main(input_parameters):
    logger.info("run")
    try:
        readRawData(input_parameters)
        logger.info("#" * 400)
    except Exception as e:
        with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
            file.write(str(-1) + "\n")
        logger.error(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
