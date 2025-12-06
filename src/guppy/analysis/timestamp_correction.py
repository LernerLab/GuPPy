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


# function to correct timestamps after eliminating first few seconds of the data (for csv data)
def timestampCorrection_csv(
    filepath, timeForLightsTurnOn, storesList, name_to_data, name_to_timestamps, name_to_sampling_rate
):
    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)

    indices = check_cntrl_sig_length(arr, name_to_data)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        idx = np.where(names_for_storenames == indices[i])[0]

        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        name = names_for_storenames[idx][0]
        timestamp = name_to_timestamps[name]
        sampling_rate = name_to_sampling_rate[name]

        if name_1 == name_2:
            correctionIndex = np.where(timestamp >= timeForLightsTurnOn)[0]
            timestampNew = timestamp[correctionIndex]
            # TODO: Pull out write operations into preprocess.py
            write_hdf5(timestampNew, "timeCorrection_" + name_1, filepath, "timestampNew")
            write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
            write_hdf5(np.asarray(sampling_rate), "timeCorrection_" + name_1, filepath, "sampling_rate")

        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrected and converted to seconds.")


# function to correct timestamps after eliminating first few seconds of the data (for TDT data)
def timestampCorrection_tdt(filepath, timeForLightsTurnOn, storesList):

    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    storenames = storesList[0, :]
    storesList = storesList[1, :]

    arr = []
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
            arr.append(storesList[i])

    arr = sorted(arr, key=str.casefold)

    try:
        arr = np.asarray(arr).reshape(2, -1)
    except:
        logger.error("Error in saving stores list file or spelling mistake for control or signal")
        raise Exception("Error in saving stores list file or spelling mistake for control or signal")

    indices = check_cntrl_sig_length(filepath, arr, storenames, storesList)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        idx = np.where(storesList == indices[i])[0]

        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        timestamp = read_hdf5(storenames[idx][0], filepath, "timestamps")
        npoints = read_hdf5(storenames[idx][0], filepath, "npoints")
        sampling_rate = read_hdf5(storenames[idx][0], filepath, "sampling_rate")

        if name_1 == name_2:
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

            write_hdf5(np.asarray([timeRecStart]), "timeCorrection_" + name_1, filepath, "timeRecStart")
            write_hdf5(timestampNew, "timeCorrection_" + name_1, filepath, "timestampNew")
            write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
            write_hdf5(np.asarray([sampling_rate]), "timeCorrection_" + name_1, filepath, "sampling_rate")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrected and converted to seconds.")
    # return timeRecStart, correctionIndex, timestampNew


# function to check if naming convention was followed while saving storeslist file
# and apply timestamps correction using the function applyCorrection
def decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, event, displayName, storesList):

    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storesList = storesList[1, :]

    arr = []
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
            arr.append(storesList[i])

    arr = sorted(arr, key=str.casefold)
    arr = np.asarray(arr).reshape(2, -1)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        if name_1 == name_2:
            applyCorrection(filepath, timeForLightsTurnOn, event, displayName, name_1)
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrections applied to the data and event timestamps.")


# function to apply correction to control, signal and event timestamps
def applyCorrection(filepath, timeForLightsTurnOn, event, displayName, naming):

    cond = check_TDT(os.path.dirname(filepath))

    if cond == True:
        timeRecStart = read_hdf5("timeCorrection_" + naming, filepath, "timeRecStart")[0]

    timestampNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    correctionIndex = read_hdf5("timeCorrection_" + naming, filepath, "correctionIndex")

    if "control" in displayName.lower() or "signal" in displayName.lower():
        split_name = displayName.split("_")[-1]
        if split_name == naming:
            pass
        else:
            correctionIndex = read_hdf5("timeCorrection_" + split_name, filepath, "correctionIndex")
        arr = read_hdf5(event, filepath, "data")
        if (arr == 0).all() == True:
            arr = arr
        else:
            arr = arr[correctionIndex]
        write_hdf5(arr, displayName, filepath, "data")
    else:
        arr = read_hdf5(event, filepath, "timestamps")
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

    # if isosbestic_control==False and 'control' in displayName.lower():
    # 	control = create_control_channel(filepath, displayName)
    # 	write_hdf5(control, displayName, filepath, 'data')


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

        name_to_data[control_name] = control_data
        name_to_data[signal_name] = signal_data
        name_to_timestamps[control_name] = control_timestamps
        name_to_timestamps[signal_name] = signal_timestamps
        name_to_sampling_rate[control_name] = control_sampling_rate
        name_to_sampling_rate[signal_name] = signal_sampling_rate

    return name_to_data, name_to_timestamps, name_to_sampling_rate
