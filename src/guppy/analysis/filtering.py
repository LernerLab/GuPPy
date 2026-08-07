"""Smoothing the raw channels before they are fit and normalized."""

import logging

import numpy as np
from scipy import signal as ss

logger = logging.getLogger(__name__)


def validate_chunk_lengths_for_filtering(tsNew: np.ndarray, coords: np.ndarray, filter_window: int) -> None:
    """
    Ensure every artifact-removal chunk is long enough for the moving-average filter.

    ``filterSignal`` filters each retained chunk with ``scipy.signal.filtfilt``, which
    zero-phase pads by ``padlen = 3 * filter_window`` and therefore requires each chunk to
    contain more than ``3 * filter_window`` samples. A large artifact can leave a short
    surviving chunk; without this check ``filtfilt`` raises an opaque
    "input vector x must be greater than padlen" error naming neither the chunk nor the cause.

    Parameters
    ----------
    tsNew : np.ndarray
        Corrected timestamp array aligned with the channel data.
    coords : np.ndarray
        Shape ``(N, 2)`` good-chunk boundary array from artifact removal.
    filter_window : int
        Moving-average filter window length; 0 disables filtering (no check needed).

    Raises
    ------
    ValueError
        If any retained chunk has ``<= 3 * filter_window`` samples.
    """
    if filter_window <= 1:
        return

    padlen = 3 * filter_window
    for i in range(coords.shape[0]):
        chunk_indices = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        if chunk_indices.shape[0] <= padlen:
            message = (
                f"retained segment [{coords[i, 0]}, {coords[i, 1]}]s has {chunk_indices.shape[0]} samples, but "
                f"the moving-average filter (window={filter_window}) needs more than {padlen}. Shrink the artifact "
                "selection over this segment, or lower the moving-average filter window."
            )
            logger.error(message)
            raise ValueError(message)


def filterSignal(filter_window: int, signal: np.ndarray) -> np.ndarray:
    """
    Apply a moving-average (uniform FIR) filter to a signal array.

    Parameters
    ----------
    filter_window : int
        Window length in samples; 0 returns ``signal`` unchanged; must be > 1 to filter.
    signal : np.ndarray
        1-D signal array.

    Returns
    -------
    np.ndarray
        Filtered signal array, or ``signal`` when ``filter_window`` is 0.
    """
    if filter_window == 0:
        return signal
    elif filter_window > 1:
        b = np.divide(np.ones((filter_window,)), filter_window)
        a = 1
        filtered_signal = ss.filtfilt(b, a, signal)
        return filtered_signal
    else:
        message = (
            f"filter_window={filter_window} is not a valid moving-average window. Use 0 to disable "
            "filtering, or a window length greater than 1."
        )
        logger.error(message)
        raise ValueError(message)


def filter_over_chunks(trace: np.ndarray, chunk_index_list: list[np.ndarray], filter_window: int) -> np.ndarray:
    """
    Filter each retained chunk of a trace independently.

    Filtering chunk by chunk keeps the filter from ever running across a removed segment, where it
    would smear the discontinuity into the samples on either side.

    Parameters
    ----------
    trace : np.ndarray
        Full-length channel trace.
    chunk_index_list : list of np.ndarray
        Indices of each retained chunk, from ``retained_chunk_indices``.
    filter_window : int
        Moving-average filter window length; 0 disables filtering.

    Returns
    -------
    np.ndarray
        Full-length filtered trace, NaN wherever no chunk covers the sample.
    """
    filtered = np.full(trace.shape[0], np.nan)
    for chunk_indices in chunk_index_list:
        filtered[chunk_indices] = filterSignal(filter_window, trace[chunk_indices])
    return filtered
