import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def eliminateData(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    arr = np.array([])
    ts_arr = np.array([])
    for i in range(len(filepath)):
        ts = read_hdf5("timeCorrection_" + naming, filepath[i], "timestampNew")
        data = read_hdf5(event, filepath[i], "data").reshape(-1)

        # index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]

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


def eliminateTs(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    for i in range(len(filepath)):
        tsNew = read_hdf5("timeCorrection_" + naming, filepath[i], "timestampNew")
        if os.path.exists(os.path.join(filepath[i], event + "_" + naming + ".hdf5")):
            ts = read_hdf5(event + "_" + naming, filepath[i], "ts").reshape(-1)
        else:
            ts = np.array([])

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


def combine_data(filepath, timeForLightsTurnOn, events, sampling_rate):

    logger.debug("Processing timestamps for combining data...")

    storesList = events[1, :]

    for k in range(len(filepath)):

        path = decide_naming_convention(filepath[k][0])

        pair_name_to_tsNew = {}
        for j in range(path.shape[1]):
            name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
            name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
            # dirname = os.path.dirname(path[i])
            if name_1[-1] == name_2[-1]:
                name = name_1[-1]

                for i in range(len(storesList)):
                    if (
                        "control_" + name.lower() in storesList[i].lower()
                        or "signal_" + name.lower() in storesList[i].lower()
                    ):
                        data, timestampNew = eliminateData(
                            filepath[k], timeForLightsTurnOn, storesList[i], sampling_rate, name
                        )
                        write_hdf5(data, storesList[i], filepath[k][0], "data")
                        pair_name_to_tsNew[name] = timestampNew
                    else:
                        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
                            continue
                        else:
                            ts = eliminateTs(filepath[k], timeForLightsTurnOn, storesList[i], sampling_rate, name)
                            write_hdf5(ts, storesList[i] + "_" + name, filepath[k][0], "ts")
        for pair_name, tsNew in pair_name_to_tsNew.items():
            write_hdf5(tsNew, "timeCorrection_" + pair_name, filepath[k][0], "timestampNew")
