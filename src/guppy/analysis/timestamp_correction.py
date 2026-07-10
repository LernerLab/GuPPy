import logging

import numpy as np

from .io_utils import get_control_and_signal_channel_names

logger = logging.getLogger(__name__)


def correct_timestamps(
    timeForLightsTurnOn: float,
    store_array: np.ndarray,
    store_label_to_timestamps: dict[str, np.ndarray],
    store_label_to_data: dict[str, np.ndarray],
    store_label_to_sampling_rate: dict[str, np.ndarray],
    store_label_to_npoints: dict[str, np.ndarray | None],
    store_label_to_timestamps_ttl: dict[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Apply timestamp correction to all channels and TTL stores.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset for the start of the recording; samples before this are dropped.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].
    store_label_to_timestamps : dict
        Store label → raw timestamp array.
    store_label_to_data : dict
        Store label → raw data array.
    store_label_to_sampling_rate : dict
        Store label → sampling-rate array.
    store_label_to_npoints : dict
        Store label → npoints array (or None for CSV data).
    store_label_to_timestamps_ttl : dict
        Store label → TTL timestamp array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    store_label_to_corrected_timestamps : dict
        Store label → corrected timestamp array.
    store_label_to_correction_index : dict
        Store label → index array applied to the raw timestamps.
    store_label_to_corrected_data : dict
        Store label → corrected data array.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
    store_label_to_corrected_timestamps, store_label_to_correction_index, store_label_to_corrected_data = (
        timestampCorrection(
            timeForLightsTurnOn,
            store_array,
            store_label_to_timestamps,
            store_label_to_data,
            store_label_to_sampling_rate,
            store_label_to_npoints,
            mode=mode,
        )
    )
    compound_name_to_corrected_ttl_timestamps = decide_naming_and_applyCorrection_ttl(
        timeForLightsTurnOn,
        store_array,
        store_label_to_timestamps_ttl,
        store_label_to_timestamps,
        store_label_to_data,
        mode=mode,
    )

    return (
        store_label_to_corrected_timestamps,
        store_label_to_correction_index,
        store_label_to_corrected_data,
        compound_name_to_corrected_ttl_timestamps,
    )


def timestampCorrection(
    timeForLightsTurnOn: float,
    store_array: np.ndarray,
    store_label_to_timestamps: dict[str, np.ndarray],
    store_label_to_data: dict[str, np.ndarray],
    store_label_to_sampling_rate: dict[str, np.ndarray],
    store_label_to_npoints: dict[str, np.ndarray | None],
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Trim and realign control/signal timestamps, discarding samples before ``timeForLightsTurnOn``.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset; samples before this value are discarded.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].
    store_label_to_timestamps : dict
        Store label → raw timestamp array.
    store_label_to_data : dict
        Store label → raw data array.
    store_label_to_sampling_rate : dict
        Store label → sampling-rate value.
    store_label_to_npoints : dict
        Store label → npoints value (or None for CSV data).
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    store_label_to_corrected_timestamps : dict
        Store label → corrected timestamp array.
    store_label_to_correction_index : dict
        Store label → index array used to slice raw data.
    store_label_to_corrected_data : dict
        Store label → sliced data array.
    """
    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    if mode not in ["tdt", "csv"]:
        message = f"Mode {mode!r} is not supported; must be either 'tdt' or 'csv'."
        logger.error(message)
        raise ValueError(message)
    store_label_to_corrected_timestamps = {}
    store_label_to_correction_index = {}
    store_label_to_corrected_data = {}
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]
    control_signal_names = get_control_and_signal_channel_names(store_array)

    indices = check_cntrl_sig_length(control_signal_names, store_label_to_data)

    for i in range(control_signal_names.shape[1]):
        control_name = control_signal_names[0, i]
        signal_name = control_signal_names[1, i]
        match_index = np.where(store_labels == indices[i])[0]

        name = store_labels[match_index][0]
        timestamp = store_label_to_timestamps[name]
        sampling_rate = store_label_to_sampling_rate[name]
        npoints = store_label_to_npoints[name]

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
            store_label_to_corrected_timestamps[displayName] = timestampNew
            store_label_to_correction_index[displayName] = correctionIndex
            data = store_label_to_data[displayName]
            if (data == 0).all() == True:
                store_label_to_corrected_data[displayName] = data
            else:
                store_label_to_corrected_data[displayName] = data[correctionIndex]

    logger.info("Timestamps corrected and converted to seconds.")
    return store_label_to_corrected_timestamps, store_label_to_correction_index, store_label_to_corrected_data


def decide_naming_and_applyCorrection_ttl(
    timeForLightsTurnOn: float,
    store_array: np.ndarray,
    store_label_to_timestamps_ttl: dict[str, np.ndarray],
    store_label_to_timestamps: dict[str, np.ndarray],
    store_label_to_data: dict[str, np.ndarray],
    mode: str,
) -> dict[str, np.ndarray]:
    """
    Apply timestamp correction to all TTL stores and pair them with channel suffixes.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset used as the new time zero.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].
    store_label_to_timestamps_ttl : dict
        TTL display name → raw TTL timestamp array.
    store_label_to_timestamps : dict
        Channel display name → raw photometry timestamp array.
    store_label_to_data : dict
        Channel display name → raw data array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]
    control_signal_names = get_control_and_signal_channel_names(store_array)
    indices = check_cntrl_sig_length(control_signal_names, store_label_to_data)

    compound_name_to_corrected_ttl_timestamps = {}
    for ttl_name, ttl_timestamps in store_label_to_timestamps_ttl.items():
        for i in range(control_signal_names.shape[1]):
            name_1 = control_signal_names[0, i].split("_")[-1]

            match_index = np.where(store_labels == indices[i])[0]

            name = store_labels[match_index][0]
            timestamps = store_label_to_timestamps[name]
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
    timeForLightsTurnOn: float,
    timeRecStart: float,
    ttl_timestamps: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Shift TTL timestamps onto the recording-start time base.

    Events are placed on the same recording-start basis as the continuous
    ``timestampNew`` stream (see :func:`timestampCorrection`): for TDT the
    recording start (``timeRecStart``) is subtracted; for CSV the timestamps are
    already recording-relative and are returned unchanged. Events are *not*
    re-zeroed to ``timeForLightsTurnOn`` — keeping both streams on one shared
    origin so consumers can co-register them without per-stream offset bookkeeping.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset of the lights-on instant; retained for API compatibility
        but no longer used to shift events.
    timeRecStart : float
        Absolute start time of the recording (TDT only; ignored for CSV).
    ttl_timestamps : np.ndarray
        Raw TTL timestamp array.
    mode : str
        Acquisition format; one of ``'tdt'`` or ``'csv'``.

    Returns
    -------
    corrected_ttl_timestamps : np.ndarray
        TTL timestamps on the recording-start time base.
    """
    corrected_ttl_timestamps = ttl_timestamps
    if mode == "tdt":
        all_on_recording_clock = (corrected_ttl_timestamps >= timeRecStart).all()
        # When all TTLs are on the recording clock, rebase them to recording start.
        # Otherwise they are not on the recording clock; leave them as-is (rare path).
        if all_on_recording_clock == True:
            corrected_ttl_timestamps = np.subtract(corrected_ttl_timestamps, timeRecStart)
    return corrected_ttl_timestamps


def check_cntrl_sig_length(control_signal_names: np.ndarray, store_label_to_data: dict[str, np.ndarray]) -> list[str]:
    """
    Identify the shorter channel in each control/signal pair.

    Parameters
    ----------
    control_signal_names : np.ndarray
        Shape ``(2, N)`` array where row 0 is control names and row 1 is signal names.
    store_label_to_data : dict
        Store label → data array.

    Returns
    -------
    indices : list of str
        List of display names (one per pair) pointing to the channel with fewer samples.
        When lengths are equal the signal name is returned.
    """

    indices = []
    for i in range(control_signal_names.shape[1]):
        control_name = control_signal_names[0, i]
        signal_name = control_signal_names[1, i]
        control = store_label_to_data[control_name]
        signal = store_label_to_data[signal_name]
        if control.shape[0] < signal.shape[0]:
            indices.append(control_name)
        elif control.shape[0] > signal.shape[0]:
            indices.append(signal_name)
        else:
            indices.append(signal_name)

    return indices
