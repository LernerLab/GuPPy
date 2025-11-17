import glob
import logging
import os
import tkinter as tk
from tkinter import StringVar, messagebox, ttk

import h5py
import numpy as np
import pandas as pd
import panel as pn

pn.extension()

logger = logging.getLogger(__name__)

def import_doric(filepath):

    logger.debug("If it exists, importing Doric file based on the structure of file")
    path = sorted(glob.glob(os.path.join(filepath, "*.csv"))) + sorted(glob.glob(os.path.join(filepath, "*.doric")))

    path = sorted(list(set(path)))
    flag = "None"
    event_from_filename = []
    flag_arr = []
    for i in range(len(path)):
        ext = os.path.basename(path[i]).split(".")[-1]
        if ext == "doric":
            key_names = read_doric(path[i])
            event_from_filename.extend(key_names)
            flag = "doric_doric"
        else:
            df = pd.read_csv(path[i], header=None, nrows=2, index_col=False, dtype=str)
            df = df.dropna(axis=1, how="all")
            df_arr = np.array(df).flatten()
            check_all_str = []
            for element in df_arr:
                try:
                    float(element)
                except:
                    check_all_str.append(i)
            assert len(check_all_str) == len(df_arr), "This file appears to be standard .csv. This function only supports doric .csv files."
            df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
            df = df.drop(["Time(s)"], axis=1)
            event_from_filename.extend(list(df.columns))
            flag = "doric_csv"
            logger.info(flag)
    logger.info("Importing of Doric file is done.")
    return event_from_filename, flag_arr


def read_doric(filepath):
    with h5py.File(filepath, "r") as f:
        if "Traces" in list(f.keys()):
            keys = access_keys_doricV1(f)
        elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
            keys = access_keys_doricV6(f)

    return keys

def access_keys_doricV6(doric_file):
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

    keys = []
    for element in res:
        sep_values = element.split("/")
        if sep_values[-1] == "Values":
            keys.append(f"{sep_values[-3]}/{sep_values[-2]}")
        else:
            keys.append(f"{sep_values[-2]}/{sep_values[-1]}")

    return keys


def access_keys_doricV1(doric_file):
    keys = list(doric_file["Traces"]["Console"].keys())
    keys.remove("Time(s)")

    return keys

def separate_last_element(arr):
    l = arr[-1]
    return arr[:-1], l