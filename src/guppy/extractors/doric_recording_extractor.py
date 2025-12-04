import glob
import logging
import os
import re
import warnings
from typing import Any

import h5py
import numpy as np
import pandas as pd

from guppy.extractors import BaseRecordingExtractor

logger = logging.getLogger(__name__)


def execute_import_doric(folder_path, storesList, outputPath):
    events = list(storesList[0, :])
    event_name_to_event_type = {storesList[0, i]: storesList[1, i] for i in range(storesList.shape[1])}

    extractor = DoricRecordingExtractor(folder_path=folder_path, event_name_to_event_type=event_name_to_event_type)
    output_dicts = extractor.read(events=events, outputPath=outputPath)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)


class DoricRecordingExtractor(BaseRecordingExtractor):
    # TODO: consolidate duplicate flag logic between the `discover_events_and_flags` and the `check_doric` method.

    @classmethod
    def discover_events_and_flags(cls, folder_path):
        """
        Discover available events and file format flags from Doric files.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing Doric files

        Returns
        -------
        events : list
            List of discovered event names
        flags : list
            List of format flags (e.g., 'doric_csv', 'doric_doric')
        """
        logger.debug("Discovering Doric events from file headers")
        path = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) + sorted(
            glob.glob(os.path.join(folder_path, "*.doric"))
        )

        path = sorted(list(set(path)))
        flag = "None"
        event_from_filename = []
        flag_arr = []

        for i in range(len(path)):
            ext = os.path.basename(path[i]).split(".")[-1]
            if ext == "doric":
                key_names = cls._read_doric_file(path[i])
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

        logger.info("Doric event discovery complete.")
        return event_from_filename, flag_arr

    def __init__(self, folder_path, event_name_to_event_type):
        self.folder_path = folder_path
        self._event_name_to_event_type = event_name_to_event_type

    @staticmethod
    def _read_doric_file(filepath):
        """Static helper to read Doric file headers for event discovery."""
        with h5py.File(filepath, "r") as f:
            if "Traces" in list(f.keys()):
                keys = DoricRecordingExtractor._access_keys_doricV1(f)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                keys = DoricRecordingExtractor._access_keys_doricV6(f)

        return keys

    @staticmethod
    def _access_keys_doricV6(doric_file):
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = DoricRecordingExtractor._separate_last_element(data)
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

    @staticmethod
    def _access_keys_doricV1(doric_file):
        keys = list(doric_file["Traces"]["Console"].keys())
        keys.remove("Time(s)")

        return keys

    @staticmethod
    def _separate_last_element(arr):
        l = arr[-1]
        return arr[:-1], l

    def _check_doric(self):
        logger.debug("Checking if doric file exists")
        path = glob.glob(os.path.join(self.folder_path, "*.csv")) + glob.glob(os.path.join(self.folder_path, "*.doric"))

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

    def _read_doric_csv(self, events):
        path = glob.glob(os.path.join(self.folder_path, "*.csv"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric csv file present at the location")
            raise Exception("More than one Doric csv file present at the location")

        df = pd.read_csv(path[0], header=1, index_col=False)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="any")
        df["Time(s)"] = df["Time(s)"] - df["Time(s)"].to_numpy()[0]

        output_dicts = []
        for event in events:
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                timestamps = np.array(df["Time(s)"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                data = np.array(df[event])
                storename = event
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
            else:
                ttl = df[event]
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = df["Time(s)"][indices[diff_indices] + 1].to_numpy()
                storename = event
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def _read_doric_doric(self, events):
        path = glob.glob(os.path.join(self.folder_path, "*.doric"))
        if len(path) > 1:
            logger.error("An error occurred : More than one Doric file present at the location")
            raise Exception("More than one Doric file present at the location")
        with h5py.File(path[0], "r") as f:
            if "Traces" in list(f.keys()):
                output_dicts = self._access_data_doricV1(f, events)
            elif list(f.keys()) == ["Configurations", "DataAcquisition"]:
                output_dicts = self._access_data_doricV6(f, events)
        return output_dicts

    def _access_data_doricV6(self, doric_file, events):
        data = [doric_file["DataAcquisition"]]
        res = []
        while len(data) != 0:
            members = len(data)
            while members != 0:
                members -= 1
                data, last_element = self._separate_last_element(data)
                if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
                    res.append(last_element.name)
                elif isinstance(last_element, h5py.Group):
                    data.extend(reversed([last_element[k] for k in last_element.keys()]))

        decide_path = []
        for element in res:
            sep_values = element.split("/")
            if sep_values[-1] == "Values":
                if f"{sep_values[-3]}/{sep_values[-2]}" in events:
                    decide_path.append(element)
            else:
                if f"{sep_values[-2]}/{sep_values[-1]}" in events:
                    decide_path.append(element)

        output_dicts = []
        for event in events:
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                regex = re.compile("(.*?)" + str(event) + "(.*?)")
                idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
                if len(idx) > 1:
                    logger.error("More than one string matched (which should not be the case)")
                    raise Exception("More than one string matched (which should not be the case)")
                idx = idx[0]
                data = np.array(doric_file[decide_path[idx]])
                timestamps = np.array(doric_file[decide_path[idx].rsplit("/", 1)[0] + "/Time"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                storename = event
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
            else:
                regex = re.compile("(.*?)" + event + "$")
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
                storename = event
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def _access_data_doricV1(self, doric_file, events):
        keys = list(doric_file["Traces"]["Console"].keys())
        output_dicts = []
        for event in events:
            event_type = self._event_name_to_event_type[event]
            if "control" in event_type or "signal" in event_type:
                timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
                sampling_rate = np.array([1 / (timestamps[-1] - timestamps[-2])])
                data = np.array(doric_file["Traces"]["Console"][event][event])
                storename = event
                S = {"storename": storename, "sampling_rate": sampling_rate, "timestamps": timestamps, "data": data}
                output_dicts.append(S)
            else:
                timestamps = np.array(doric_file["Traces"]["Console"]["Time(s)"]["Console_time(s)"])
                ttl = np.array(doric_file["Traces"]["Console"][event][event])
                indices = np.where(ttl <= 0)[0]
                diff_indices = np.where(np.diff(indices) > 1)[0]
                timestamps = timestamps[indices[diff_indices] + 1]
                storename = event
                S = {"storename": storename, "timestamps": timestamps}
                output_dicts.append(S)

        return output_dicts

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        flag = self._check_doric()
        if flag == "doric_csv":
            output_dicts = self._read_doric_csv(events)
        elif flag == "doric_doric":
            output_dicts = self._read_doric_doric(events)
        else:
            logger.error("Doric file not found or not recognized.")
            raise FileNotFoundError("Doric file not found or not recognized.")

        return output_dicts

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        for S in output_dicts:
            storename = S["storename"]
            self._write_hdf5(data=S["timestamps"], storename=storename, output_path=outputPath, key="timestamps")

            if "sampling_rate" in S:
                self._write_hdf5(
                    data=S["sampling_rate"], storename=storename, output_path=outputPath, key="sampling_rate"
                )
            if "data" in S:
                self._write_hdf5(data=S["data"], storename=storename, output_path=outputPath, key="data")
