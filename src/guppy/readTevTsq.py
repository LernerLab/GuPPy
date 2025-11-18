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
from guppy.csv_step3 import execute_import_csv

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

# function to read data from 'tsq' and 'tev' files
def readRawData(inputParameters):

    logger.debug("### Reading raw data... ###")
    # get input parameters
    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    modality = inputParameters["modality"]
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
