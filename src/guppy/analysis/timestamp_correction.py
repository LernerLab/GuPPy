import logging

import numpy as np

from .io_utils import get_control_and_signal_channel_names

logger = logging.getLogger(__name__)


def correct_timestamps(
    timeForLightsTurnOn,
    storesList,
    name_to_timestamps,
    name_to_data,
    name_to_sampling_rate,
    name_to_npoints,
    name_to_timestamps_ttl,
    mode,
):
    name_to_corrected_timestamps, name_to_correctionIndex, name_to_corrected_data = timestampCorrection(
        timeForLightsTurnOn,
        storesList,
        name_to_timestamps,
        name_to_data,
        name_to_sampling_rate,
        name_to_npoints,
        mode=mode,
    )
    compound_name_to_corrected_ttl_timestamps = decide_naming_and_applyCorrection_ttl(
        timeForLightsTurnOn,
        storesList,
        name_to_timestamps_ttl,
        name_to_timestamps,
        name_to_data,
        mode=mode,
    )

    return (
        name_to_corrected_timestamps,
        name_to_correctionIndex,
        name_to_corrected_data,
        compound_name_to_corrected_ttl_timestamps,
    )


# function to correct timestamps after eliminating first few seconds of the data (for csv or TDT data depending on mode)
def timestampCorrection(
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
        message = f"Mode {mode!r} is not supported; must be either 'tdt' or 'csv'."
        logger.error(message)
        raise ValueError(message)
    name_to_corrected_timestamps = {}
    name_to_correctionIndex = {}
    name_to_corrected_data = {}
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    channels_arr = get_control_and_signal_channel_names(storesList)

    indices = check_cntrl_sig_length(channels_arr, name_to_data)

    for i in range(channels_arr.shape[1]):
        control_name = channels_arr[0, i]
        signal_name = channels_arr[1, i]
        name_1 = channels_arr[0, i].split("_")[-1]
        name_2 = channels_arr[1, i].split("_")[-1]
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in storesList: control channel '{control_name}' has suffix '{name_1}' "
                f"but signal channel '{signal_name}' has suffix '{name_2}'. Check the naming convention of "
                "your files and the storesList file, then re-run step 2."
            )
            logger.error(message)
            raise ValueError(message)

        # dirname = os.path.dirname(path[i])
        idx = np.where(names_for_storenames == indices[i])[0]

        if idx.shape[0] == 0:
            message = (
                f"Channel '{channels_arr[0, i]}' does not exist in the storesList file. "
                f"Available channel names: {sorted(set(names_for_storenames.tolist()))}."
            )
            logger.error(message)
            raise ValueError(message)

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

        for displayName in [control_name, signal_name]:
            name_to_corrected_timestamps[displayName] = timestampNew
            name_to_correctionIndex[displayName] = correctionIndex
            data = name_to_data[displayName]
            if (data == 0).all() == True:
                name_to_corrected_data[displayName] = data
            else:
                name_to_corrected_data[displayName] = data[correctionIndex]

    logger.info("Timestamps corrected and converted to seconds.")
    return name_to_corrected_timestamps, name_to_correctionIndex, name_to_corrected_data


def decide_naming_and_applyCorrection_ttl(
    timeForLightsTurnOn,
    storesList,
    name_to_timestamps_ttl,
    name_to_timestamps,
    name_to_data,
    mode,
):
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)
    indices = check_cntrl_sig_length(arr, name_to_data)

    compound_name_to_corrected_ttl_timestamps = {}
    for ttl_name, ttl_timestamps in name_to_timestamps_ttl.items():
        for i in range(arr.shape[1]):
            name_1 = arr[0, i].split("_")[-1]
            name_2 = arr[1, i].split("_")[-1]
            if name_1 != name_2:
                message = (
                    f"Pair name mismatch in storesList: control channel '{arr[0, i]}' has suffix "
                    f"'{name_1}' but signal channel '{arr[1, i]}' has suffix '{name_2}'. Check the "
                    "naming convention of your files and the storesList file, then re-run step 2."
                )
                logger.error(message)
                raise ValueError(message)

            idx = np.where(names_for_storenames == indices[i])[0]
            if idx.shape[0] == 0:
                message = (
                    f"Channel '{arr[0, i]}' does not exist in the storesList file. "
                    f"Available channel names: {sorted(set(names_for_storenames.tolist()))}."
                )
                logger.error(message)
                raise ValueError(message)

            name = names_for_storenames[idx][0]
            timestamps = name_to_timestamps[name]
            timeRecStart = timestamps[0]
            corrected_ttl_timestamps = applyCorrection_ttl(
                timeForLightsTurnOn,
                timeRecStart,
                ttl_timestamps,
                mode,
            )
            compound_name = ttl_name + "_" + name_1
            compound_name_to_corrected_ttl_timestamps[compound_name] = corrected_ttl_timestamps

    logger.info("Timestamps corrections applied to the data and event timestamps.")
    return compound_name_to_corrected_ttl_timestamps


def applyCorrection_ttl(
    timeForLightsTurnOn,
    timeRecStart,
    ttl_timestamps,
    mode,
):
    corrected_ttl_timestamps = ttl_timestamps
    if mode == "tdt":
        res = (corrected_ttl_timestamps >= timeRecStart).all()
        if res == True:
            corrected_ttl_timestamps = np.subtract(corrected_ttl_timestamps, timeRecStart)
            corrected_ttl_timestamps = np.subtract(corrected_ttl_timestamps, timeForLightsTurnOn)
        else:
            corrected_ttl_timestamps = np.subtract(corrected_ttl_timestamps, timeForLightsTurnOn)
    elif mode == "csv":
        corrected_ttl_timestamps = np.subtract(corrected_ttl_timestamps, timeForLightsTurnOn)
    return corrected_ttl_timestamps


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
