import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
)

logger = logging.getLogger(__name__)


def eliminateData(filepath_to_timestamps, filepath_to_data, timeForLightsTurnOn, sampling_rate):

    arr = np.array([])
    ts_arr = np.array([])
    filepaths = list(filepath_to_timestamps.keys())
    for filepath in filepaths:
        ts = filepath_to_timestamps[filepath]
        data = filepath_to_data[filepath]

        if len(arr) == 0:
            arr = np.concatenate((arr, data))
            sub = ts[0] - timeForLightsTurnOn
            new_ts = ts - sub
            ts_arr = np.concatenate((ts_arr, new_ts))
        else:
            temp = data
            temp_ts = ts
            new_ts = temp_ts - (temp_ts[0] - ts_arr[-1])
            arr = np.concatenate((arr, temp))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    return arr, ts_arr


def eliminateTs(filepath_to_timestamps, filepath_to_ttl_timestamps, timeForLightsTurnOn, sampling_rate):

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    filepaths = list(filepath_to_timestamps.keys())
    for filepath in filepaths:
        tsNew = filepath_to_timestamps[filepath]
        ts = filepath_to_ttl_timestamps[filepath]
        if len(tsNew_arr) == 0:
            sub = tsNew[0] - timeForLightsTurnOn
            tsNew_arr = np.concatenate((tsNew_arr, tsNew - sub))
            ts_arr = np.concatenate((ts_arr, ts - sub))
        else:
            temp_tsNew = tsNew
            temp_ts = ts
            new_ts = temp_ts - (temp_tsNew[0] - tsNew_arr[-1])
            new_tsNew = temp_tsNew - (temp_tsNew[0] - tsNew_arr[-1])
            tsNew_arr = np.concatenate((tsNew_arr, new_tsNew + (1 / sampling_rate)))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

        # logger.info(event)
        # logger.info(ts_arr)
    return ts_arr


def combine_data(
    filepaths_to_combine: list[str],
    pair_name_to_filepath_to_timestamps: dict[str, dict[str, np.ndarray]],
    display_name_to_filepath_to_data: dict[str, dict[str, np.ndarray]],
    compound_name_to_filepath_to_ttl_timestamps: dict[str, dict[str, np.ndarray]],
    timeForLightsTurnOn,
    storesList,
    sampling_rate,
):
    # filepaths_to_combine = [folder1_output_i, folder2_output_i, ...]
    logger.debug("Processing timestamps for combining data...")

    names_for_storenames = storesList[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])

    pair_name_to_tsNew = {}
    display_name_to_data = {}
    compound_name_to_ttl_timestamps = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
        pair_name = name_1

        for i in range(len(names_for_storenames)):
            if (
                "control_" + pair_name.lower() in names_for_storenames[i].lower()
                or "signal_" + pair_name.lower() in names_for_storenames[i].lower()
            ):
                display_name = names_for_storenames[i]
                filepath_to_timestamps = pair_name_to_filepath_to_timestamps[pair_name]
                filepath_to_data = display_name_to_filepath_to_data[display_name]
                data, timestampNew = eliminateData(
                    filepath_to_timestamps,
                    filepath_to_data,
                    timeForLightsTurnOn,
                    sampling_rate,
                )
                pair_name_to_tsNew[pair_name] = timestampNew
                display_name_to_data[display_name] = data
            else:
                if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
                    continue
                compound_name = names_for_storenames[i] + "_" + pair_name
                filepath_to_timestamps = pair_name_to_filepath_to_timestamps[pair_name]
                filepath_to_ttl_timestamps = compound_name_to_filepath_to_ttl_timestamps[compound_name]

                ts = eliminateTs(
                    filepath_to_timestamps,
                    filepath_to_ttl_timestamps,
                    timeForLightsTurnOn,
                    sampling_rate,
                )
                compound_name_to_ttl_timestamps[compound_name] = ts

    return pair_name_to_tsNew, display_name_to_data, compound_name_to_ttl_timestamps
