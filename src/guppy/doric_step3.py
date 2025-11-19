import glob
import logging
import os
import re
import warnings

import h5py
import numpy as np
import pandas as pd

from guppy.common_step3 import write_hdf5

logger = logging.getLogger(__name__)


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


def execute_import_doric(filepath, storesList, flag, outputPath):
    flag = check_doric(filepath)

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
