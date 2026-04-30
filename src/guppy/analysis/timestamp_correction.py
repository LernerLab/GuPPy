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
    """
    Apply timestamp correction to all channels and TTL stores.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset for the start of the recording; samples before this are dropped.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    name_to_timestamps : dict
        Display name → raw timestamp array.
    name_to_data : dict
        Display name → raw data array.
    name_to_sampling_rate : dict
        Display name → sampling-rate array.
    name_to_npoints : dict
        Display name → npoints array (or None for CSV data).
    name_to_timestamps_ttl : dict
        Display name → TTL timestamp array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    name_to_corrected_timestamps : dict
        Display name → corrected timestamp array.
    name_to_correctionIndex : dict
        Display name → index array applied to the raw timestamps.
    name_to_corrected_data : dict
        Display name → corrected data array.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
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


def timestampCorrection(
    timeForLightsTurnOn,
    storesList,
    name_to_timestamps,
    name_to_data,
    name_to_sampling_rate,
    name_to_npoints,
    mode,
):
    """
    Trim and realign control/signal timestamps, discarding samples before ``timeForLightsTurnOn``.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset; samples before this value are discarded.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    name_to_timestamps : dict
        Display name → raw timestamp array.
    name_to_data : dict
        Display name → raw data array.
    name_to_sampling_rate : dict
        Display name → sampling-rate value.
    name_to_npoints : dict
        Display name → npoints value (or None for CSV data).
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    name_to_corrected_timestamps : dict
        Display name → corrected timestamp array.
    name_to_correctionIndex : dict
        Display name → index array used to slice raw data.
    name_to_corrected_data : dict
        Display name → sliced data array.
    """
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
        # dirname = os.path.dirname(path[i])
        idx = np.where(names_for_storenames == indices[i])[0]

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
    """
    Apply timestamp correction to all TTL stores and pair them with channel suffixes.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset used as the new time zero.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    name_to_timestamps_ttl : dict
        TTL display name → raw TTL timestamp array.
    name_to_timestamps : dict
        Channel display name → raw photometry timestamp array.
    name_to_data : dict
        Channel display name → raw data array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)
    indices = check_cntrl_sig_length(arr, name_to_data)

    compound_name_to_corrected_ttl_timestamps = {}
    for ttl_name, ttl_timestamps in name_to_timestamps_ttl.items():
        for i in range(arr.shape[1]):
            name_1 = arr[0, i].split("_")[-1]

            idx = np.where(names_for_storenames == indices[i])[0]

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
    """
    Shift TTL timestamps to align with the corrected photometry time base.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset for the new time zero.
    timeRecStart : float
        Absolute start time of the recording (TDT only; ignored for CSV).
    ttl_timestamps : np.ndarray
        Raw TTL timestamp array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    corrected_ttl_timestamps : np.ndarray
        TTL timestamps shifted to the corrected time base.
    """
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


def check_cntrl_sig_length(channels_arr, name_to_data):
    """
    Identify the shorter channel in each control/signal pair.

    Parameters
    ----------
    channels_arr : np.ndarray
        Shape ``(2, N)`` array where row 0 is control names and row 1 is signal names.
    name_to_data : dict
        Display name → data array.

    Returns
    -------
    indices : list of str
        List of display names (one per pair) pointing to the channel with fewer samples.
        When lengths are equal the signal name is returned.
    """

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
