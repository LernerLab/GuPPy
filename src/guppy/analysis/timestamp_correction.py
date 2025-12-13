import logging
import os

import numpy as np
import pandas as pd

from .control_channel import helper_create_control_channel
from .io_utils import (
    check_TDT,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def write_corrected_timestamps(
    filepath, corrected_name_to_timestamps, name_to_timestamps, name_to_sampling_rate, name_to_correctionIndex
):
    for name, correctionIndex in name_to_correctionIndex.items():
        timestamps = name_to_timestamps[name]
        corrected_timestamps = corrected_name_to_timestamps[name]
        sampling_rate = name_to_sampling_rate[name]
        if sampling_rate.shape == ():  # numpy scalar
            sampling_rate = np.asarray([sampling_rate])
        name_1 = name.split("_")[-1]
        write_hdf5(np.asarray([timestamps[0]]), "timeCorrection_" + name_1, filepath, "timeRecStart")
        write_hdf5(corrected_timestamps, "timeCorrection_" + name_1, filepath, "timestampNew")
        write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
        write_hdf5(sampling_rate, "timeCorrection_" + name_1, filepath, "sampling_rate")


# function to correct timestamps after eliminating first few seconds of the data (for csv or TDT data depending on mode)
def timestampCorrection(
    filepath,
    timeForLightsTurnOn,
    storesList,
    name_to_timestamps,
    name_to_data,
    name_to_sampling_rate,
    name_to_npoints,
    mode,
):
    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    if mode not in ["tdt", "csv"]:
        logger.error("Mode should be either 'tdt' or 'csv'")
        raise ValueError("Mode should be either 'tdt' or 'csv'")
    name_to_timestamps = name_to_timestamps.copy()
    name_to_correctionIndex = {}
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)

    indices = check_cntrl_sig_length(arr, name_to_data)

    for i in range(arr.shape[1]):
        control_name = arr[0, i]
        signal_name = arr[1, i]
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

        # dirname = os.path.dirname(path[i])
        idx = np.where(names_for_storenames == indices[i])[0]

        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        name = names_for_storenames[idx][0]
        timestamp = name_to_timestamps[name]
        sampling_rate = name_to_sampling_rate[name]
        npoints = name_to_npoints[name]

        if mode == "tdt":
            timeRecStart = timestamp[0]
            timestamps = np.subtract(timestamp, timeRecStart)
            adder = np.arange(npoints) / sampling_rate
            lengthAdder = adder.shape[0]
            timestampNew = np.zeros((len(timestamps), lengthAdder))
            for i in range(lengthAdder):
                timestampNew[:, i] = np.add(timestamps, adder[i])
            timestampNew = (timestampNew.T).reshape(-1, order="F")
            correctionIndex = np.where(timestampNew >= timeForLightsTurnOn)[0]
            timestampNew = timestampNew[correctionIndex]
        elif mode == "csv":
            correctionIndex = np.where(timestamp >= timeForLightsTurnOn)[0]
            timestampNew = timestamp[correctionIndex]

        name_to_timestamps[name] = timestampNew
        name_to_correctionIndex[name] = correctionIndex

        arr = name_to_data[control_name]
        if (arr == 0).all() == True:
            arr = arr
        else:
            arr = arr[correctionIndex]
        write_hdf5(arr, control_name, filepath, "data")

        arr = name_to_data[signal_name]
        if (arr == 0).all() == True:
            arr = arr
        else:
            arr = arr[correctionIndex]
        write_hdf5(arr, signal_name, filepath, "data")

    logger.info("Timestamps corrected and converted to seconds.")
    return name_to_timestamps, name_to_correctionIndex


# function to check if naming convention was followed while saving storeslist file
# and apply timestamps correction using the function applyCorrection
def decide_naming_convention_and_applyCorrection(
    filepath,
    timeForLightsTurnOn,
    event,
    displayName,
    storesList,
    name_1_to_corrected_timestamps,
    name_1_to_timestamps,
    name_1_to_sampling_rate,
    name_1_to_correctionIndex,
    data,
    ttl_timestamps,
):

    logger.debug("Applying correction of timestamps to the data and event timestamps")
    arr = get_control_and_signal_channel_names(storesList)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
        else:
            corrected_timestamps = name_1_to_corrected_timestamps[name_1]
            timestamps = name_1_to_timestamps[name_1]
            timeRecStart = timestamps[0]
            sampling_rate = name_1_to_sampling_rate[name_1]
            correctionIndex = name_1_to_correctionIndex[name_1]
            applyCorrection(
                filepath,
                timeForLightsTurnOn,
                event,
                displayName,
                name_1,
                corrected_timestamps,
                sampling_rate,
                correctionIndex,
                timeRecStart,
                data,
                ttl_timestamps,
            )

    logger.info("Timestamps corrections applied to the data and event timestamps.")


def decide_naming_and_applyCorrection_signal_and_control(
    filepath,
    storesList,
    name_to_correctionIndex,
    name_to_data,
):
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)
    indices = check_cntrl_sig_length(arr, name_to_data)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

        idx = np.where(names_for_storenames == indices[i])[0]
        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        name = names_for_storenames[idx][0]
        correctionIndex = name_to_correctionIndex[name]
        control_name = arr[0, i]
        signal_name = arr[1, i]
        control_data = name_to_data[control_name]
        signal_data = name_to_data[signal_name]
        applyCorrection_signal_and_control(filepath, control_name, correctionIndex, control_data)
        applyCorrection_signal_and_control(filepath, signal_name, correctionIndex, signal_data)

    logger.info("Timestamps corrections applied to the data and event timestamps.")


def applyCorrection_signal_and_control(filepath, displayName, correctionIndex, data):
    arr = data
    if (arr == 0).all() == True:
        arr = arr
    else:
        arr = arr[correctionIndex]
    write_hdf5(arr, displayName, filepath, "data")


def decide_naming_and_applyCorrection_ttl(
    filepath,
    timeForLightsTurnOn,
    storesList,
    name_to_timestamps_ttl,
    name_to_timestamps,
    name_to_data,
):
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)
    indices = check_cntrl_sig_length(arr, name_to_data)

    for ttl_name, ttl_timestamps in name_to_timestamps_ttl.items():
        displayName = ttl_name
        for i in range(arr.shape[1]):
            name_1 = arr[0, i].split("_")[-1]
            name_2 = arr[1, i].split("_")[-1]
            if name_1 != name_2:
                logger.error("Error in naming convention of files or Error in storesList file")
                raise Exception("Error in naming convention of files or Error in storesList file")

            idx = np.where(names_for_storenames == indices[i])[0]
            if idx.shape[0] == 0:
                logger.error(f"{arr[0,i]} does not exist in the stores list file.")
                raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

            name = names_for_storenames[idx][0]
            timestamps = name_to_timestamps[name]
            timeRecStart = timestamps[0]
            applyCorrection_ttl(
                filepath,
                timeForLightsTurnOn,
                displayName,
                name_1,
                timeRecStart,
                ttl_timestamps,
            )

    logger.info("Timestamps corrections applied to the data and event timestamps.")


def applyCorrection_ttl(
    filepath,
    timeForLightsTurnOn,
    displayName,
    naming,
    timeRecStart,
    ttl_timestamps,
):
    cond = check_TDT(os.path.dirname(filepath))
    arr = ttl_timestamps
    if cond == True:
        res = (arr >= timeRecStart).all()
        if res == True:
            arr = np.subtract(arr, timeRecStart)
            arr = np.subtract(arr, timeForLightsTurnOn)
        else:
            arr = np.subtract(arr, timeForLightsTurnOn)
    else:
        arr = np.subtract(arr, timeForLightsTurnOn)
    write_hdf5(arr, displayName + "_" + naming, filepath, "ts")


# function to apply correction to control, signal and event timestamps
def applyCorrection(
    filepath,
    timeForLightsTurnOn,
    event,
    displayName,
    naming,
    corrected_timestamps,
    sampling_rate,
    correctionIndex,
    timeRecStart,
    data,
    ttl_timestamps,
):

    cond = check_TDT(os.path.dirname(filepath))

    timestampNew = corrected_timestamps
    if "control" in displayName.lower() or "signal" in displayName.lower():
        # TODO: double-check that this code is not reachable
        # split_name = displayName.split("_")[-1]
        # if split_name == naming:
        #     pass
        # else:
        #     correctionIndex = read_hdf5("timeCorrection_" + split_name, filepath, "correctionIndex")
        arr = data
        if (arr == 0).all() == True:
            arr = arr
        else:
            arr = arr[correctionIndex]
        write_hdf5(arr, displayName, filepath, "data")
    else:
        arr = ttl_timestamps
        if cond == True:
            res = (arr >= timeRecStart).all()
            if res == True:
                arr = np.subtract(arr, timeRecStart)
                arr = np.subtract(arr, timeForLightsTurnOn)
            else:
                arr = np.subtract(arr, timeForLightsTurnOn)
        else:
            arr = np.subtract(arr, timeForLightsTurnOn)
        write_hdf5(arr, displayName + "_" + naming, filepath, "ts")


# main function to create control channel using
# signal channel and save it to a file
def create_control_channel(filepath, arr, window=5001):

    storenames = arr[0, :]
    storesList = arr[1, :]

    for i in range(storesList.shape[0]):
        event_name, event = storesList[i], storenames[i]
        if "control" in event_name.lower() and "cntrl" in event.lower():
            logger.debug("Creating control channel from signal channel using curve-fitting")
            name = event_name.split("_")[-1]
            signal = read_hdf5("signal_" + name, filepath, "data")
            timestampNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
            sampling_rate = np.full(timestampNew.shape, np.nan)
            sampling_rate[0] = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            control = helper_create_control_channel(signal, timestampNew, window)

            write_hdf5(control, event_name, filepath, "data")
            d = {"timestamps": timestampNew, "data": control, "sampling_rate": sampling_rate}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(os.path.dirname(filepath), event.lower() + ".csv"), index=False)
            logger.info("Control channel from signal channel created using curve-fitting")


# function to check control and signal channel has same length
# if not, take a smaller length and do pre-processing
def check_cntrl_sig_length(channels_arr, name_to_data):

    indices = []
    for i in range(channels_arr.shape[1]):
        control_name = channels_arr[0, i]
        signal_name = channels_arr[1, i]
        control = name_to_data[control_name]
        signal = name_to_data[signal_name]
        if control.shape[0] < signal.shape[0]:
            indices.append(control_name)
        elif control.shape[0] > signal.shape[0]:
            indices.append(signal_name)
        else:
            indices.append(signal_name)

    return indices


def get_control_and_signal_channel_names(storesList):
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    channels_arr = []
    for i in range(names_for_storenames.shape[0]):
        if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
            channels_arr.append(names_for_storenames[i])

    channels_arr = sorted(channels_arr, key=str.casefold)
    try:
        channels_arr = np.asarray(channels_arr).reshape(2, -1)
    except:
        logger.error("Error in saving stores list file or spelling mistake for control or signal")
        raise Exception("Error in saving stores list file or spelling mistake for control or signal")

    return channels_arr


def read_control_and_signal(filepath, storesList):
    channels_arr = get_control_and_signal_channel_names(storesList)
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    name_to_data = {}
    name_to_timestamps = {}
    name_to_sampling_rate = {}
    name_to_npoints = {}

    for i in range(channels_arr.shape[1]):
        control_name = channels_arr[0, i]
        signal_name = channels_arr[1, i]
        idx_c = np.where(storesList == control_name)[0]
        idx_s = np.where(storesList == signal_name)[0]
        control_storename = storenames[idx_c[0]]
        signal_storename = storenames[idx_s[0]]

        control_data = read_hdf5(control_storename, filepath, "data")
        signal_data = read_hdf5(signal_storename, filepath, "data")
        control_timestamps = read_hdf5(control_storename, filepath, "timestamps")
        signal_timestamps = read_hdf5(signal_storename, filepath, "timestamps")
        control_sampling_rate = read_hdf5(control_storename, filepath, "sampling_rate")
        signal_sampling_rate = read_hdf5(signal_storename, filepath, "sampling_rate")
        try:  # TODO: define npoints for csv datasets
            control_npoints = read_hdf5(control_storename, filepath, "npoints")
            signal_npoints = read_hdf5(signal_storename, filepath, "npoints")
        except KeyError:  # npoints is not defined for csv datasets
            control_npoints = None
            signal_npoints = None

        name_to_data[control_name] = control_data
        name_to_data[signal_name] = signal_data
        name_to_timestamps[control_name] = control_timestamps
        name_to_timestamps[signal_name] = signal_timestamps
        name_to_sampling_rate[control_name] = control_sampling_rate
        name_to_sampling_rate[signal_name] = signal_sampling_rate
        name_to_npoints[control_name] = control_npoints
        name_to_npoints[signal_name] = signal_npoints

    return name_to_data, name_to_timestamps, name_to_sampling_rate, name_to_npoints


def read_ttl(filepath, storesList):
    channels_arr = get_control_and_signal_channel_names(storesList)
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    name_to_timestamps = {}
    for storename, name in zip(storenames, names_for_storenames):
        if name in channels_arr:
            continue
        timestamps = read_hdf5(storename, filepath, "timestamps")
        name_to_timestamps[name] = timestamps

    return name_to_timestamps
