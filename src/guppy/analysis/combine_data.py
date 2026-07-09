import logging
import os

import numpy as np

from .io_utils import (
    decide_naming_convention,
)
from .realignment import concatenate_and_realign_data, realign_ttl_timestamps

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
    concatenated_data : np.ndarray
        Concatenated data across all sessions.
    realigned_timestamps : np.ndarray
        Realigned timestamps corresponding to ``concatenated_data``.
    """

    segments = [(filepath_to_data[filepath], filepath_to_timestamps[filepath]) for filepath in filepath_to_timestamps]
    return concatenate_and_realign_data(segments, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


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
    realigned_ttl_timestamps : np.ndarray
        Realigned TTL timestamps concatenated across all sessions.
    """

    # Both tsNew (continuous) and ts (events) are on the recording-start basis, so the same
    # per-session shift keeps them mutually aligned; the inter-session bridging uses differences
    # only, which are basis-invariant.
    pairs = [
        (filepath_to_timestamps[filepath], filepath_to_ttl_timestamps[filepath]) for filepath in filepath_to_timestamps
    ]
    return realign_ttl_timestamps(pairs, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


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
            message = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file suffix '{name_1}' does not match "
                f"signal file suffix '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
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

                ttl_timestamps = eliminateTs(
                    filepath_to_timestamps,
                    filepath_to_ttl_timestamps,
                    timeForLightsTurnOn,
                    sampling_rate,
                )
                compound_name_to_ttl_timestamps[compound_name] = ttl_timestamps

    return pair_name_to_tsNew, display_name_to_data, compound_name_to_ttl_timestamps
