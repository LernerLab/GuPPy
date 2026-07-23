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
    concatenated_data : np.ndarray
        Concatenated data across all segments.
    realigned_timestamps : np.ndarray
        Realigned timestamps corresponding to ``concatenated_data``.
    """
    concatenated_data = np.array([])
    realigned_timestamps = np.array([])
    for data_segment, ts_segment in segments:
        if len(concatenated_data) == 0:
            concatenated_data = np.concatenate((concatenated_data, data_segment))
            offset = ts_segment[0] - timeForLightsTurnOn
            shifted_timestamps = ts_segment - offset
            realigned_timestamps = np.concatenate((realigned_timestamps, shifted_timestamps))
        else:
            shifted_timestamps = ts_segment - (ts_segment[0] - realigned_timestamps[-1])
            concatenated_data = np.concatenate((concatenated_data, data_segment))
            realigned_timestamps = np.concatenate((realigned_timestamps, shifted_timestamps + (1 / sampling_rate)))

    return concatenated_data, realigned_timestamps


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
    realigned_ttl_timestamps : np.ndarray
        Realigned TTL timestamps concatenated across all segments.
    """
    realigned_ttl_timestamps = np.array([])
    realigned_reference_timestamps = np.array([])
    for tsNew_segment, ts_segment in pairs:
        if len(realigned_reference_timestamps) == 0:
            offset = tsNew_segment[0] - timeForLightsTurnOn
            realigned_reference_timestamps = np.concatenate((realigned_reference_timestamps, tsNew_segment - offset))
            realigned_ttl_timestamps = np.concatenate((realigned_ttl_timestamps, ts_segment - offset))
        else:
            shifted_ttl_timestamps = ts_segment - (tsNew_segment[0] - realigned_reference_timestamps[-1])
            shifted_reference_timestamps = tsNew_segment - (tsNew_segment[0] - realigned_reference_timestamps[-1])
            realigned_reference_timestamps = np.concatenate(
                (realigned_reference_timestamps, shifted_reference_timestamps + (1 / sampling_rate))
            )
            realigned_ttl_timestamps = np.concatenate(
                (realigned_ttl_timestamps, shifted_ttl_timestamps + (1 / sampling_rate))
            )

    return realigned_ttl_timestamps
