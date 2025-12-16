import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
    fetchCoords,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def addingNaNtoChunksWithArtifacts(filepath, events):

    logger.debug("Replacing chunks with artifacts by NaN values.")
    storesList = events[1, :]

    path = decide_naming_convention(filepath)

    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        # dirname = os.path.dirname(path[i])
        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]
            ts = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
            coords = fetchCoords(filepath, name, ts)
            for i in range(len(storesList)):
                if (
                    "control_" + name.lower() in storesList[i].lower()
                    or "signal_" + name.lower() in storesList[i].lower()
                ):  # changes done
                    data = read_hdf5(storesList[i], filepath, "data").reshape(-1)
                    data = addingNaNValues(data=data, ts=ts, coords=coords)
                    write_hdf5(data, storesList[i], filepath, "data")
                else:
                    if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
                        continue
                    else:
                        ts = removeTTLs(filepath, storesList[i], name)
                        write_hdf5(ts, storesList[i] + "_" + name, filepath, "ts")

        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
    logger.info("Chunks with artifacts are replaced by NaN values.")


# main function to align timestamps for control, signal and event timestamps for artifacts removal
def processTimestampsForArtifacts(filepath, timeForLightsTurnOn, events):

    logger.debug("Processing timestamps to get rid of artifacts using concatenate method...")
    storesList = events[1, :]

    path = decide_naming_convention(filepath)

    timestamp_dict = dict()
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        # dirname = os.path.dirname(path[i])
        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            for i in range(len(storesList)):
                if (
                    "control_" + name.lower() in storesList[i].lower()
                    or "signal_" + name.lower() in storesList[i].lower()
                ):  # changes done
                    data, timestampNew = eliminateData(
                        filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name
                    )
                    write_hdf5(data, storesList[i], filepath, "data")
                else:
                    if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
                        continue
                    else:
                        ts = eliminateTs(filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name)
                        write_hdf5(ts, storesList[i] + "_" + name, filepath, "ts")

            # timestamp_dict[name] = timestampNew
            write_hdf5(timestampNew, "timeCorrection_" + name, filepath, "timestampNew")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
    logger.info("Timestamps processed, artifacts are removed and good chunks are concatenated.")


# helper function to process control and signal timestamps
def eliminateData(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    ts = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    data = read_hdf5(event, filepath, "data").reshape(-1)
    coords = fetchCoords(filepath, naming, ts)

    if (data == 0).all() == True:
        data = np.zeros(ts.shape[0])

    arr = np.array([])
    ts_arr = np.array([])
    for i in range(coords.shape[0]):

        index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]

        if len(arr) == 0:
            arr = np.concatenate((arr, data[index]))
            sub = ts[index][0] - timeForLightsTurnOn
            new_ts = ts[index] - sub
            ts_arr = np.concatenate((ts_arr, new_ts))
        else:
            temp = data[index]
            # new = temp + (arr[-1]-temp[0])
            temp_ts = ts[index]
            new_ts = temp_ts - (temp_ts[0] - ts_arr[-1])
            arr = np.concatenate((arr, temp))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    # logger.info(arr.shape, ts_arr.shape)
    return arr, ts_arr


# helper function to align event timestamps with the control and signal timestamps
def eliminateTs(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    tsNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    ts = read_hdf5(event + "_" + naming, filepath, "ts").reshape(-1)
    coords = fetchCoords(filepath, naming, tsNew)

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    for i in range(coords.shape[0]):
        tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]

        if len(tsNew_arr) == 0:
            sub = tsNew[tsNew_index][0] - timeForLightsTurnOn
            tsNew_arr = np.concatenate((tsNew_arr, tsNew[tsNew_index] - sub))
            ts_arr = np.concatenate((ts_arr, ts[ts_index] - sub))
        else:
            temp_tsNew = tsNew[tsNew_index]
            temp_ts = ts[ts_index]
            new_ts = temp_ts - (temp_tsNew[0] - tsNew_arr[-1])
            new_tsNew = temp_tsNew - (temp_tsNew[0] - tsNew_arr[-1])
            tsNew_arr = np.concatenate((tsNew_arr, new_tsNew + (1 / sampling_rate)))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    return ts_arr


# adding nan values to removed chunks
# when using artifacts removal method - replace with NaN
def addingNaNValues(*, data, ts, coords):

    if (data == 0).all() == True:
        data = np.zeros(ts.shape[0])

    arr = np.array([])
    ts_index = np.arange(ts.shape[0])
    for i in range(coords.shape[0]):

        index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        arr = np.concatenate((arr, index))

    nan_indices = list(set(ts_index).symmetric_difference(arr))
    data[nan_indices] = np.nan

    return data


# remove event TTLs which falls in the removed chunks
# when using artifacts removal method - replace with NaN
def removeTTLs(filepath, event, naming):
    tsNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    ts = read_hdf5(event + "_" + naming, filepath, "ts").reshape(-1)
    coords = fetchCoords(filepath, naming, tsNew)

    ts_arr = np.array([])
    for i in range(coords.shape[0]):
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        ts_arr = np.concatenate((ts_arr, ts[ts_index]))

    return ts_arr
