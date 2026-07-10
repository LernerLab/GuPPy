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
            ):
                data = name_to_data[names_for_storenames[i]].reshape(-1)
                data = addingNaNValues(data=data, timestamps=tsNew, coords=coords)
                name_to_corrected_data[names_for_storenames[i]] = data
            else:
                if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
                    continue
                ttl_name = names_for_storenames[i]
                compound_name = ttl_name + "_" + pair_name
                ttl_timestamps = compound_name_to_ttl_timestamps[compound_name].reshape(-1)
                ttl_timestamps = removeTTLs(ttl_timestamps=ttl_timestamps, coords=coords)
                compound_name_to_corrected_ttl_timestamps[compound_name] = ttl_timestamps
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
            ):
                data = name_to_data[names_for_storenames[i]]
                data, timestampNew = eliminateData(
                    data=data,
                    timestamps=tsNew,
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
                ttl_timestamps = compound_name_to_ttl_timestamps[compound_name]
                ttl_timestamps = eliminateTs(
                    ttl_timestamps=ttl_timestamps,
                    tsNew=tsNew,
                    coords=coords,
                    timeForLightsTurnOn=timeForLightsTurnOn,
                    sampling_rate=sampling_rate,
                )
                compound_name_to_corrected_ttl_timestamps[compound_name] = ttl_timestamps

    logger.info("Timestamps processed, artifacts are removed and good chunks are concatenated.")

    return (
        name_to_corrected_data,
        pair_name_to_corrected_timestamps,
        compound_name_to_corrected_ttl_timestamps,
    )


def eliminateData(
    *, data: np.ndarray, timestamps: np.ndarray, coords: np.ndarray, timeForLightsTurnOn: float, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate non-artifact data chunks and realign their timestamps.

    Parameters
    ----------
    data : np.ndarray
        1-D data array aligned with ``timestamps``.
    timestamps : np.ndarray
        1-D timestamp array.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first chunk.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-chunk spacing.

    Returns
    -------
    concatenated_data : np.ndarray
        Concatenated data from all good chunks.
    realigned_timestamps : np.ndarray
        Realigned timestamps corresponding to ``concatenated_data``.
    """

    if (data == 0).all() == True:
        data = np.zeros(timestamps.shape[0])

    segments = []
    for i in range(coords.shape[0]):
        in_window_indices = np.where((timestamps > coords[i, 0]) & (timestamps < coords[i, 1]))[0]
        segments.append((data[in_window_indices], timestamps[in_window_indices]))

    return concatenate_and_realign_data(segments, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


def eliminateTs(
    *,
    ttl_timestamps: np.ndarray,
    tsNew: np.ndarray,
    coords: np.ndarray,
    timeForLightsTurnOn: float,
    sampling_rate: float,
) -> np.ndarray:
    """
    Realign TTL timestamps to match concatenated non-artifact photometry chunks.

    Parameters
    ----------
    ttl_timestamps : np.ndarray
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
    realigned_ttl_timestamps : np.ndarray
        Realigned TTL timestamps.
    """

    # tsNew (continuous) and ttl_timestamps (events) are both on the recording-start basis, matching
    # the basis of the artifact-removal coords, so both windowing comparisons are consistent.
    pairs = []
    for i in range(coords.shape[0]):
        reference_in_window_indices = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        ttl_in_window_indices = np.where((ttl_timestamps > coords[i, 0]) & (ttl_timestamps < coords[i, 1]))[0]
        pairs.append((tsNew[reference_in_window_indices], ttl_timestamps[ttl_in_window_indices]))

    return realign_ttl_timestamps(pairs, timeForLightsTurnOn=timeForLightsTurnOn, sampling_rate=sampling_rate)


def addingNaNValues(*, data: np.ndarray, timestamps: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Set data samples outside the good-chunk windows to NaN.

    Parameters
    ----------
    data : np.ndarray
        1-D data array aligned with ``timestamps``.
    timestamps : np.ndarray
        1-D timestamp array.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.

    Returns
    -------
    data : np.ndarray
        Data array with artifact samples replaced by NaN.
    """

    if (data == 0).all() == True:
        data = np.zeros(timestamps.shape[0])

    kept_indices = np.array([])
    all_indices = np.arange(timestamps.shape[0])
    for i in range(coords.shape[0]):

        chunk_indices = np.where((timestamps > coords[i, 0]) & (timestamps < coords[i, 1]))[0]
        kept_indices = np.concatenate((kept_indices, chunk_indices))

    nan_indices = list(set(all_indices).symmetric_difference(kept_indices))
    data[nan_indices] = np.nan

    return data


def removeTTLs(*, ttl_timestamps: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Keep only TTL timestamps that fall within good-chunk windows.

    Parameters
    ----------
    ttl_timestamps : np.ndarray
        TTL timestamp array to filter.
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.

    Returns
    -------
    kept_ttl_timestamps : np.ndarray
        TTL timestamps that fall within the good-chunk windows.
    """
    kept_ttl_timestamps = np.array([])
    for i in range(coords.shape[0]):
        in_window_indices = np.where((ttl_timestamps > coords[i, 0]) & (ttl_timestamps < coords[i, 1]))[0]
        kept_ttl_timestamps = np.concatenate((kept_ttl_timestamps, ttl_timestamps[in_window_indices]))

    return kept_ttl_timestamps
