import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def eliminateData(filepath_to_timestamps, filepath_to_data, timeForLightsTurnOn, event, sampling_rate, naming):

    arr = np.array([])
    ts_arr = np.array([])
    filepaths = list(filepath_to_timestamps.keys())
    for filepath in filepaths:
        ts = filepath_to_timestamps[filepath]
        data = filepath_to_data[filepath]
        # ts = read_hdf5("timeCorrection_" + naming, filepaths[i], "timestampNew")
        # data = read_hdf5(event, filepaths[i], "data").reshape(-1)

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


def eliminateTs(filepath_to_timestamps, filepath_to_ttl_timestamps, timeForLightsTurnOn, event, sampling_rate, naming):

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    filepaths = list(filepath_to_timestamps.keys())
    for filepath in filepaths:
        ts = filepath_to_timestamps[filepath]
        tsNew = filepath_to_ttl_timestamps[filepath]
        # tsNew = read_hdf5("timeCorrection_" + naming, filepath[i], "timestampNew")
        # if os.path.exists(os.path.join(filepath[i], event + "_" + naming + ".hdf5")):
        #     ts = read_hdf5(event + "_" + naming, filepath[i], "ts").reshape(-1)
        # else:
        #     ts = np.array([])

        # logger.info("total time : ", tsNew[-1])
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


def combine_data(filepath: list[list[str]], timeForLightsTurnOn, names_for_storenames, sampling_rate):
    # filepath = [[folder1_output_0, folder2_output_0], [folder1_output_1, folder2_output_1], ...]

    logger.debug("Processing timestamps for combining data...")

    names_for_storenames = names_for_storenames[1, :]

    for single_output_filepaths in filepath:
        # single_output_filepaths = [folder1_output_i, folder2_output_i, ...]

        path = decide_naming_convention(single_output_filepaths[0])

        pair_name_to_tsNew = {}
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
                    filepath_to_timestamps = {}
                    filepath_to_data = {}
                    for filepath in single_output_filepaths:
                        ts = read_hdf5("timeCorrection_" + pair_name, filepath, "timestampNew")
                        data = read_hdf5(names_for_storenames[i], filepath, "data").reshape(-1)
                        filepath_to_timestamps[filepath] = ts
                        filepath_to_data[filepath] = data

                    data, timestampNew = eliminateData(
                        filepath_to_timestamps,
                        filepath_to_data,
                        timeForLightsTurnOn,
                        names_for_storenames[i],
                        sampling_rate,
                        pair_name,
                    )
                    write_hdf5(data, names_for_storenames[i], single_output_filepaths[0], "data")
                    pair_name_to_tsNew[pair_name] = timestampNew
                else:
                    if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
                        continue
                    filepath_to_timestamps = {}
                    filepath_to_ttl_timestamps = {}
                    for filepath in single_output_filepaths:
                        tsNew = read_hdf5("timeCorrection_" + pair_name, filepath, "timestampNew")
                        if os.path.exists(os.path.join(filepath, names_for_storenames[i] + "_" + pair_name + ".hdf5")):
                            ts = read_hdf5(names_for_storenames[i] + "_" + pair_name, filepath, "ts").reshape(-1)
                        else:
                            ts = np.array([])
                        filepath_to_timestamps[filepath] = tsNew
                        filepath_to_ttl_timestamps[filepath] = ts

                    ts = eliminateTs(
                        filepath_to_timestamps,
                        filepath_to_ttl_timestamps,
                        timeForLightsTurnOn,
                        names_for_storenames[i],
                        sampling_rate,
                        pair_name,
                    )
                    write_hdf5(ts, names_for_storenames[i] + "_" + pair_name, single_output_filepaths[0], "ts")
        for pair_name, tsNew in pair_name_to_tsNew.items():
            write_hdf5(tsNew, "timeCorrection_" + pair_name, single_output_filepaths[0], "timestampNew")
