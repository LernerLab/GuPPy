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


# function to check if doric file exists
def check_doric(filepath):
    logger.debug("Checking if doric file exists")
    path = glob.glob(os.path.join(filepath, "*.csv")) + glob.glob(os.path.join(filepath, "*.doric"))

    flag_arr = []
    for i in range(len(path)):
        ext = os.path.basename(path[i]).split(".")[-1]
        if ext == "csv":
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    df = pd.read_csv(path[i], index_col=False, dtype=float)
                except:
                    df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
                    flag = "doric_csv"
                    flag_arr.append(flag)
        elif ext == "doric":
            flag = "doric_doric"
            flag_arr.append(flag)
        else:
            pass

    if len(flag_arr) > 1:
        logger.error("Two doric files are present at the same location")
        raise Exception("Two doric files are present at the same location")
    if len(flag_arr) == 0:
        logger.error("\033[1m" + "Doric file not found." + "\033[1m")
        return 0
    logger.info("Doric file found.")
    return flag_arr[0]


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


def access_data_doricV1(doric_file, storesList, outputPath):
    keys = list(doric_file["Traces"]["Console"].keys())
    for i in range(storesList.shape[1]):
        if "control" in storesList[1, i] or "signal" in storesList[1, i]:
            timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
            sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
            data = np.array(doric_file["Traces"]["Console"][storesList[0, i]][storesList[0, i]])
            write_hdf5(sampling_rate, storesList[0, i], outputPath, "sampling_rate")
            write_hdf5(timestamps, storesList[0, i], outputPath, "timestamps")
            write_hdf5(data, storesList[0, i], outputPath, "data")
        else:
            timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
            ttl = np.array(doric_file["Traces"]["Console"][storesList[0, i]][storesList[0, i]])
            indices = np.where(ttl <= 0)[0]
            diff_indices = np.where(np.diff(indices) > 1)[0]
            write_hdf5(timestamps[indices[diff_indices] + 1], storesList[0, i], outputPath, "timestamps")


def separate_last_element(arr):
    l = arr[-1]
    return arr[:-1], l


def find_string(regex, arr):
    for i in range(len(arr)):
        if regex.match(arr[i]):
            return i


def access_data_doricV6(doric_file, storesList, outputPath):
    data = [doric_file["DataAcquisition"]]
    res = []
    while len(data) != 0:
        members = len(data)
        while members != 0:
            members -= 1
            data, last_element = separate_last_element(data)
            if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                res.append(last_element.name)
            elif isinstance(last_element, h5py.Group):
                data.extend(reversed([last_element[k] for k in last_element.keys()]))

    decide_path = []
    for element in res:
        sep_values = element.split("/")
        if sep_values[-1] == "Values":
            if f"{sep_values[-3]}/{sep_values[-2]}" in storesList[0, :]:
                decide_path.append(element)
        else:
            if f"{sep_values[-2]}/{sep_values[-1]}" in storesList[0, :]:
                decide_path.append(element)

    for i in range(storesList.shape[1]):
        if "control" in storesList[1, i] or "signal" in storesList[1, i]:
            regex = re.compile("(.*?)" + str(storesList[0, i]) + "(.*?)")
            idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
            if len(idx) > 1:
                logger.error("More than one string matched (which should not be the case)")
                raise Exception("More than one string matched (which should not be the case)")
            idx = idx[0]
            data = np.array(doric_file[decide_path[idx]])
            timestamps = np.array(doric_file[decide_path[idx].rsplit("/", 1)[0] + "/Time"])
            sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
            write_hdf5(sampling_rate, storesList[0, i], outputPath, "sampling_rate")
            write_hdf5(timestamps, storesList[0, i], outputPath, "timestamps")
            write_hdf5(data, storesList[0, i], outputPath, "data")
        else:
            regex = re.compile("(.*?)" + storesList[0, i] + "$")
            idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
            if len(idx) > 1:
                logger.error("More than one string matched (which should not be the case)")
                raise Exception("More than one string matched (which should not be the case)")
            idx = idx[0]
            ttl = np.array(doric_file[decide_path[idx]])
            timestamps = np.array(doric_file[decide_path[idx].rsplit("/", 1)[0] + "/Time"])
            indices = np.where(ttl <= 0)[0]
            diff_indices = np.where(np.diff(indices) > 1)[0]
            write_hdf5(timestamps[indices[diff_indices] + 1], storesList[0, i], outputPath, "timestamps")


def execute_import_doric(filepath, storesList, flag, outputPath):

    if flag == "doric_csv":
        path = glob.glob(os.path.join(filepath, "*.csv"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric csv file present at the location")
            raise Exception("More than one Doric csv file present at the location")
        else:
            df = pd.read_csv(path[0], header=1, index_col=False)
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="any")
            df["Time(s)"] = df["Time(s)"] - df["Time(s)"].to_numpy()[0]
            for i in range(storesList.shape[1]):
                if "control" in storesList[1, i] or "signal" in storesList[1, i]:
                    timestamps = np.array(df["Time(s)"])
                    sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                    write_hdf5(sampling_rate, storesList[0, i], outputPath, "sampling_rate")
                    write_hdf5(df["Time(s)"].to_numpy(), storesList[0, i], outputPath, "timestamps")
                    write_hdf5(df[storesList[0, i]].to_numpy(), storesList[0, i], outputPath, "data")
                else:
                    ttl = df[storesList[0, i]]
                    indices = np.where(ttl <= 0)[0]
                    diff_indices = np.where(np.diff(indices) > 1)[0]
                    write_hdf5(
                        df["Time(s)"][indices[diff_indices] + 1].to_numpy(), storesList[0, i], outputPath, "timestamps"
                    )
    else:
        path = glob.glob(os.path.join(filepath, "*.doric"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric file present at the location")
            raise Exception("More than one Doric file present at the location")
        else:
            with h5py.File(path[0], "r") as f:
                if "Traces" in list(f.keys()):
                    keys = access_data_doricV1(f, storesList, outputPath)
                elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                    keys = access_data_doricV6(f, storesList, outputPath)


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
        modality = "tdt"

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
