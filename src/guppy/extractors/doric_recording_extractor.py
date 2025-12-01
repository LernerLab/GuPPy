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


def execute_import_doric(folder_path, storesList, outputPath):
    extractor = DoricRecordingExtractor(folder_path=folder_path)
    output_dicts = extractor.read(storesList=storesList, outputPath=outputPath, folder_path=folder_path)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)


class DoricRecordingExtractor:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        logger.debug("If it exists, importing Doric file based on the structure of file")
        path = sorted(glob.glob(os.path.join(self.folder_path, "*.csv"))) + sorted(
            glob.glob(os.path.join(self.folder_path, "*.doric"))
        )

        path = sorted(list(set(path)))
        flag = "None"
        event_from_filename = []
        flag_arr = []
        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            if ext == "doric":
                key_names = self.read_doric(path[i])
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
                assert len(check_all_str) == len(
                    df_arr
                ), "This file appears to be standard .csv. This function only supports doric .csv files."
                df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
                df = df.drop(["Time(s)"], axis=1)
                event_from_filename.extend(list(df.columns))
                flag = "doric_csv"
                logger.info(flag)
        logger.info("Importing of Doric file is done.")

        self.events = event_from_filename
        self.flags = flag_arr

    def read_doric(self, filepath):
        with h5py.File(filepath, "r") as f:
            if "Traces" in list(f.keys()):
                keys = self.access_keys_doricV1(f)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                keys = self.access_keys_doricV6(f)

        return keys

    def access_keys_doricV6(self, doric_file):
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = self.separate_last_element(data)
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

    def access_keys_doricV1(self, doric_file):
        keys = list(doric_file["Traces"]["Console"].keys())
        keys.remove("Time(s)")

        return keys

    def separate_last_element(self, arr):
        l = arr[-1]
        return arr[:-1], l

    def check_doric(self, filepath):
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
                    except:  # TODO: fix this bare try-except
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

    def read_doric_csv(self, filepath, storesList, outputPath):
        path = glob.glob(os.path.join(filepath, "*.csv"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric csv file present at the location")
            raise Exception("More than one Doric csv file present at the location")

        df = pd.read_csv(path[0], header=1, index_col=False)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="any")
        df["Time(s)"] = df["Time(s)"] - df["Time(s)"].to_numpy()[0]

        output_dicts = []
        for i in range(storesList.shape[1]):
            if "control" in storesList[1, i] or "signal" in storesList[1, i]:
                timestamps = np.array(df["Time(s)"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                data = np.array(df[storesList[0, i]])
                storename = storesList[0, i]
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
            else:
                ttl = df[storesList[0, i]]
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = df["Time(s)"][indices[diff_indices] + 1].to_numpy()
                storename = storesList[0, i]
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def read_doric_doric(self, filepath, storesList, outputPath):
        path = glob.glob(os.path.join(filepath, "*.doric"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric file present at the location")
            raise Exception("More than one Doric file present at the location")
        with h5py.File(path[0], "r") as f:
            if "Traces" in list(f.keys()):
                output_dicts = self.access_data_doricV1(f, storesList, outputPath)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                output_dicts = self.access_data_doricV6(f, storesList, outputPath)
        return output_dicts

    def access_data_doricV6(self, doric_file, storesList, outputPath):
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = self.separate_last_element(data)
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

        output_dicts = []
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
                storename = storesList[0, i]
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
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
                timestamps = timestamps[indices[diff_indices] + 1]
                storename = storesList[0, i]
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def access_data_doricV1(self, doric_file, storesList, outputPath):
        keys = list(doric_file["Traces"]["Console"].keys())
        output_dicts = []
        for i in range(storesList.shape[1]):
            if "control" in storesList[1, i] or "signal" in storesList[1, i]:
                timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                data = np.array(doric_file["Traces"]["Console"][storesList[0, i]][storesList[0, i]])
                storename = storesList[0, i]
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
            else:
                timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
                ttl = np.array(doric_file["Traces"]["Console"][storesList[0, i]][storesList[0, i]])
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = timestamps[indices[diff_indices] + 1]
                storename = storesList[0, i]
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def save_dict_to_hdf5(self, S, outputPath):
        event = S["storename"]
        # write_hdf5(S["storename"], event, outputPath, "storename")
        write_hdf5(S["timestamps"], event, outputPath, "timestamps")

        if "sampling_rate" in S:
            write_hdf5(S["sampling_rate"], event, outputPath, "sampling_rate")
        if "data" in S:
            write_hdf5(S["data"], event, outputPath, "data")
        # write_hdf5(S["npoints"], event, outputPath, "npoints")
        # write_hdf5(S["channels"], event, outputPath, "channels")

    def read(self, storesList, outputPath, folder_path):
        flag = self.check_doric(folder_path)
        if flag == "doric_csv":
            output_dicts = self.read_doric_csv(folder_path, storesList, outputPath)
        elif flag == "doric_doric":
            output_dicts = self.read_doric_doric(folder_path, storesList, outputPath)
        else:
            logger.error("Doric file not found or not recognized.")
            raise FileNotFoundError("Doric file not found or not recognized.")

        return output_dicts

    def save(self, output_dicts, outputPath):
        for S in output_dicts:
            self.save_dict_to_hdf5(S=S, outputPath=outputPath)
