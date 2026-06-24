import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
)

logger = logging.getLogger(__name__)


def eliminateData(
    filepath_to_timestamps: dict[str, np.ndarray],
    filepath_to_data: dict[str, np.ndarray],
    timeForLightsTurnOn: float,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate data from multiple session files and realign their timestamps.

    Parameters
    ----------
    filepath_to_timestamps : dict
        Mapping from session filepath to corrected timestamp array.
    filepath_to_data : dict
        Mapping from session filepath to data array.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first session.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-session spacing.

    Returns
    -------
    arr : np.ndarray
        Concatenated data across all sessions.
    ts_arr : np.ndarray
        Realigned timestamps corresponding to ``arr``.
    """

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


def eliminateTs(
    filepath_to_timestamps: dict[str, np.ndarray],
    filepath_to_ttl_timestamps: dict[str, np.ndarray],
    timeForLightsTurnOn: float,
    sampling_rate: float,
) -> np.ndarray:
    """
    Realign TTL timestamps to match concatenated multi-session photometry timestamps.

    Parameters
    ----------
    filepath_to_timestamps : dict
        Mapping from session filepath to corrected photometry timestamp array.
    filepath_to_ttl_timestamps : dict
        Mapping from session filepath to TTL timestamp array to realign.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first session.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-session spacing.

    Returns
    -------
    ts_arr : np.ndarray
        Realigned TTL timestamps concatenated across all sessions.
    """

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    filepaths = list(filepath_to_timestamps.keys())
    for filepath in filepaths:
        tsNew = filepath_to_timestamps[filepath]
        ts = filepath_to_ttl_timestamps[filepath]
        # Both tsNew (continuous) and ts (events) are on the recording-start basis, so the
        # same per-session shift keeps them mutually aligned. Inter-session bridging below
        # uses differences only, which are basis-invariant.
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

    return ts_arr


def combine_data(
    filepaths_to_combine: list[str],
    pair_name_to_filepath_to_timestamps: dict[str, dict[str, np.ndarray]],
    display_name_to_filepath_to_data: dict[str, dict[str, np.ndarray]],
    compound_name_to_filepath_to_ttl_timestamps: dict[str, dict[str, np.ndarray]],
    timeForLightsTurnOn: float,
    storesList: np.ndarray,
    sampling_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Combine photometry data and TTL timestamps from multiple session files.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories to concatenate.
    pair_name_to_filepath_to_timestamps : dict
        ``{pair_name: {filepath: timestamps_array}}``.
    display_name_to_filepath_to_data : dict
        ``{display_name: {filepath: data_array}}``.
    compound_name_to_filepath_to_ttl_timestamps : dict
        ``{compound_name: {filepath: ttl_timestamps_array}}``.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    pair_name_to_tsNew : dict
        Pair name → combined and realigned timestamp array.
    display_name_to_data : dict
        Display name → combined data array.
    compound_name_to_ttl_timestamps : dict
        Compound TTL name → combined TTL timestamp array.
    """
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
            msg = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file suffix '{name_1}' does not match "
                f"signal file suffix '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(msg)
            raise ValueError(msg)
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
