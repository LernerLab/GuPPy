import numpy as np


def concatenate_and_realign_data(
    segments: list[tuple[np.ndarray, np.ndarray]], *, timeForLightsTurnOn: float, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ordered data segments and realign their timestamps into one continuous series.

    The first segment's time zero is set to ``timeForLightsTurnOn``; each subsequent segment is
    shifted so it starts one sample (``1 / sampling_rate``) after the previous segment ends.

    Parameters
    ----------
    segments : list of (np.ndarray, np.ndarray)
        Ordered ``(data_segment, timestamp_segment)`` pairs to stitch together.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first segment.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-segment spacing.

    Returns
    -------
    arr : np.ndarray
        Concatenated data across all segments.
    ts_arr : np.ndarray
        Realigned timestamps corresponding to ``arr``.
    """
    arr = np.array([])
    ts_arr = np.array([])
    for data_segment, ts_segment in segments:
        if len(arr) == 0:
            arr = np.concatenate((arr, data_segment))
            sub = ts_segment[0] - timeForLightsTurnOn
            new_ts = ts_segment - sub
            ts_arr = np.concatenate((ts_arr, new_ts))
        else:
            new_ts = ts_segment - (ts_segment[0] - ts_arr[-1])
            arr = np.concatenate((arr, data_segment))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    return arr, ts_arr


def realign_ttl_timestamps(
    pairs: list[tuple[np.ndarray, np.ndarray]], *, timeForLightsTurnOn: float, sampling_rate: float
) -> np.ndarray:
    """Realign TTL timestamps to match concatenated photometry segments.

    Mirrors :func:`concatenate_and_realign_data` but tracks the continuous photometry timestamps
    (``tsNew``) as the alignment reference and applies the same per-segment shift to the TTL
    timestamps (``ts``), so the two stay mutually aligned across the concatenation.

    Parameters
    ----------
    pairs : list of (np.ndarray, np.ndarray)
        Ordered ``(tsNew_segment, ts_segment)`` pairs, where ``tsNew_segment`` is the continuous
        photometry timestamp segment used as the reference and ``ts_segment`` is the TTL timestamp
        segment to realign.
    timeForLightsTurnOn : float
        Seconds offset used to set the new time zero for the first segment.
    sampling_rate : float
        Sampling rate in Hz; used to compute inter-segment spacing.

    Returns
    -------
    ts_arr : np.ndarray
        Realigned TTL timestamps concatenated across all segments.
    """
    ts_arr = np.array([])
    tsNew_arr = np.array([])
    for tsNew_segment, ts_segment in pairs:
        if len(tsNew_arr) == 0:
            sub = tsNew_segment[0] - timeForLightsTurnOn
            tsNew_arr = np.concatenate((tsNew_arr, tsNew_segment - sub))
            ts_arr = np.concatenate((ts_arr, ts_segment - sub))
        else:
            new_ts = ts_segment - (tsNew_segment[0] - tsNew_arr[-1])
            new_tsNew = tsNew_segment - (tsNew_segment[0] - tsNew_arr[-1])
            tsNew_arr = np.concatenate((tsNew_arr, new_tsNew + (1 / sampling_rate)))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    return ts_arr
