import logging

import numpy as np

from .realignment import concatenate_and_realign_data, realign_ttl_timestamps

logger = logging.getLogger(__name__)


def remove_artifacts(
    timeForLightsTurnOn: float,
    storesList: np.ndarray,
    pair_name_to_tsNew: dict[str, np.ndarray],
    pair_name_to_sampling_rate: dict[str, float],
    pair_name_to_coords: dict[str, np.ndarray],
    name_to_data: dict[str, np.ndarray],
    compound_name_to_ttl_timestamps: dict[str, np.ndarray],
    method: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, np.ndarray]]:
    """
    Remove artifacts from photometry data using the specified method.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset for when lights turned on; used as the new time zero.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    pair_name_to_tsNew : dict
        Mapping from pair name to corrected timestamp array.
    pair_name_to_sampling_rate : dict
        Mapping from pair name to sampling rate (Hz).
    pair_name_to_coords : dict
        Mapping from pair name to artifact-boundary coordinates array.
    name_to_data : dict
        Mapping from display name to raw data array.
    compound_name_to_ttl_timestamps : dict
        Mapping from compound TTL name to timestamp array.
    method : str
        Artifact removal method; one of ``'concatenate'`` or ``'replace with NaN'``.

    Returns
    -------
    name_to_corrected_data : dict
        Display name → corrected data array.
    pair_name_to_corrected_timestamps : dict or None
        Pair name → corrected timestamp array, or ``None`` for the NaN method.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
    if method == "concatenate":
        name_to_corrected_data, pair_name_to_corrected_timestamps, compound_name_to_corrected_ttl_timestamps = (
            processTimestampsForArtifacts(
                timeForLightsTurnOn,
                storesList,
                pair_name_to_tsNew,
                pair_name_to_sampling_rate,
                pair_name_to_coords,
                name_to_data,
                compound_name_to_ttl_timestamps,
            )
        )
        logger.info("Artifacts removed using concatenate method.")
    elif method == "replace with NaN":
        name_to_corrected_data, compound_name_to_corrected_ttl_timestamps = addingNaNtoChunksWithArtifacts(
            storesList,
            pair_name_to_tsNew,
            pair_name_to_coords,
            name_to_data,
            compound_name_to_ttl_timestamps,
        )
        pair_name_to_corrected_timestamps = None
        logger.info("Artifacts removed using NaN replacement method.")
    else:
        message = f"Invalid artifact removal method {method!r}. " "Expected one of: 'concatenate', 'replace with NaN'."
        logger.error(message)
        raise ValueError(message)

    return name_to_corrected_data, pair_name_to_corrected_timestamps, compound_name_to_corrected_ttl_timestamps


def addingNaNtoChunksWithArtifacts(
    storesList: np.ndarray,
    pair_name_to_tsNew: dict[str, np.ndarray],
    pair_name_to_coords: dict[str, np.ndarray],
    name_to_data: dict[str, np.ndarray],
    compound_name_to_ttl_timestamps: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Replace artifact chunks in control/signal data with NaN values.

    Parameters
    ----------
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    pair_name_to_tsNew : dict
        Mapping from pair name to corrected timestamp array.
    pair_name_to_coords : dict
        Mapping from pair name to artifact-boundary coordinates array.
    name_to_data : dict
        Mapping from display name to raw data array.
    compound_name_to_ttl_timestamps : dict
        Mapping from compound TTL name to timestamp array.

    Returns
    -------
    name_to_corrected_data : dict
        Display name → data array with artifact samples set to NaN.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → TTL timestamps with artifact-window events removed.
    """
    logger.debug("Replacing chunks with artifacts by NaN values.")
    names_for_storenames = storesList[1, :]
    pair_names = pair_name_to_tsNew.keys()

    name_to_corrected_data = {}
    compound_name_to_corrected_ttl_timestamps = {}
    for pair_name in pair_names:
        tsNew = pair_name_to_tsNew[pair_name]
        coords = pair_name_to_coords[pair_name]
        for i in range(len(names_for_storenames)):
            if (
                "control_" + pair_name.lower() in names_for_storenames[i].lower()
                or "signal_" + pair_name.lower() in names_for_storenames[i].lower()
            ):  # changes done
                data = name_to_data[names_for_storenames[i]].reshape(-1)
                data = addingNaNValues(data=data, ts=tsNew, coords=coords)
                name_to_corrected_data[names_for_storenames[i]] = data
            else:
                if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
                    continue
                ttl_name = names_for_storenames[i]
                compound_name = ttl_name + "_" + pair_name
                ts = compound_name_to_ttl_timestamps[compound_name].reshape(-1)
                ts = removeTTLs(ts=ts, coords=coords)
                compound_name_to_corrected_ttl_timestamps[compound_name] = ts
    logger.info("Chunks with artifacts are replaced by NaN values.")

    return name_to_corrected_data, compound_name_to_corrected_ttl_timestamps


def processTimestampsForArtifacts(
    timeForLightsTurnOn: float,
    storesList: np.ndarray,
    pair_name_to_tsNew: dict[str, np.ndarray],
    pair_name_to_sampling_rate: dict[str, float],
    pair_name_to_coords: dict[str, np.ndarray],
    name_to_data: dict[str, np.ndarray],
    compound_name_to_ttl_timestamps: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Concatenate non-artifact chunks and realign all timestamps.

    Parameters
    ----------
    timeForLightsTurnOn : float
        Seconds offset for when lights turned on; used as the new time zero.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].
    pair_name_to_tsNew : dict
        Mapping from pair name to corrected timestamp array.
    pair_name_to_sampling_rate : dict
        Mapping from pair name to sampling rate (Hz).
    pair_name_to_coords : dict
        Mapping from pair name to artifact-boundary coordinates array.
    name_to_data : dict
        Mapping from display name to raw data array.
    compound_name_to_ttl_timestamps : dict
        Mapping from compound TTL name to timestamp array.

    Returns
    -------
    name_to_corrected_data : dict
        Display name → concatenated data array with artifacts removed.
    pair_name_to_corrected_timestamps : dict
        Pair name → realigned timestamp array.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → realigned TTL timestamp array.
    """
    logger.debug("Processing timestamps to get rid of artifacts using concatenate method...")
    names_for_storenames = storesList[1, :]
    pair_names = pair_name_to_tsNew.keys()

    name_to_corrected_data = {}
    pair_name_to_corrected_timestamps = {}
    compound_name_to_corrected_ttl_timestamps = {}
    for pair_name in pair_names:
        sampling_rate = pair_name_to_sampling_rate[pair_name]
        tsNew = pair_name_to_tsNew[pair_name]
        coords = pair_name_to_coords[pair_name]

        for i in range(len(names_for_storenames)):
            if (
                "control_" + pair_name.lower() in names_for_storenames[i].lower()
                or "signal_" + pair_name.lower() in names_for_storenames[i].lower()
            ):  # changes done
                data = name_to_data[names_for_storenames[i]]
                data, timestampNew = eliminateData(
                    data=data,
                    ts=tsNew,
                    coords=coords,
                    timeForLightsTurnOn=timeForLightsTurnOn,
                    sampling_rate=sampling_rate,
                )
                name_to_corrected_data[names_for_storenames[i]] = data
                pair_name_to_corrected_timestamps[pair_name] = timestampNew
            else:
                if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
                    continue
                compound_name = names_for_storenames[i] + "_" + pair_name
                ts = compound_name_to_ttl_timestamps[compound_name]
                ts = eliminateTs(
                    ts=ts,
                    tsNew=tsNew,
                    coords=coords,
                    timeForLightsTurnOn=timeForLightsTurnOn,
                    sampling_rate=sampling_rate,
                )
                compound_name_to_corrected_ttl_timestamps[compound_name] = ts

    logger.info("Timestamps processed, artifacts are removed and good chunks are concatenated.")

    return (
        name_to_corrected_data,
        pair_name_to_corrected_timestamps,
        compound_name_to_corrected_ttl_timestamps,
    )


def eliminateData(
    *, data: np.ndarray, ts: np.ndarray, coords: np.ndarray, timeForLightsTurnOn: float, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate non-artifact data chunks and realign their timestamps.

    Parameters
    ----------
    data : np.ndarray
        1-D data array aligned with ``ts``.
    ts : np.ndarray
        1-D timestamp array.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first chunk.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-chunk spacing.

    Returns
    -------
    arr : np.ndarray
        Concatenated data from all good chunks.
    ts_arr : np.ndarray
        Realigned timestamps corresponding to ``arr``.
    """

    if (data == 0).all() == True:
        data = np.zeros(ts.shape[0])

    segments = []
    for i in range(coords.shape[0]):
        index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        segments.append((data[index], ts[index]))

    return concatenate_and_realign_data(segments, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


def eliminateTs(
    *, ts: np.ndarray, tsNew: np.ndarray, coords: np.ndarray, timeForLightsTurnOn: float, sampling_rate: float
) -> np.ndarray:
    """
    Realign TTL timestamps to match concatenated non-artifact photometry chunks.

    Parameters
    ----------
    ts : np.ndarray
        TTL timestamp array to realign.
    tsNew : np.ndarray
        Corrected photometry timestamp array used as the reference.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first chunk.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-chunk spacing.

    Returns
    -------
    ts_arr : np.ndarray
        Realigned TTL timestamps.
    """

    # tsNew (continuous) and ts (events) are both on the recording-start basis, matching the basis
    # of the artifact-removal coords, so both windowing comparisons are consistent.
    pairs = []
    for i in range(coords.shape[0]):
        tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        pairs.append((tsNew[tsNew_index], ts[ts_index]))

    return realign_ttl_timestamps(pairs, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


def addingNaNValues(*, data: np.ndarray, ts: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Set data samples outside the good-chunk windows to NaN.

    Parameters
    ----------
    data : np.ndarray
        1-D data array aligned with ``ts``.
    ts : np.ndarray
        1-D timestamp array.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.

    Returns
    -------
    data : np.ndarray
        Data array with artifact samples replaced by NaN.
    """

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


def removeTTLs(*, ts: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Keep only TTL timestamps that fall within good-chunk windows.

    Parameters
    ----------
    ts : np.ndarray
        TTL timestamp array to filter.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.

    Returns
    -------
    ts_arr : np.ndarray
        TTL timestamps that fall within the good-chunk windows.
    """
    ts_arr = np.array([])
    for i in range(coords.shape[0]):
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        ts_arr = np.concatenate((ts_arr, ts[ts_index]))

    return ts_arr
