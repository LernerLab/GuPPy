import logging
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from ..utils.validation import validate_peak_windows

logger = logging.getLogger(__name__)


def compute_psth_peak_and_area(
    psth_mean: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float,
    peak_startPoint: Sequence[float],
    peak_endPoint: Sequence[float],
) -> OrderedDict:
    """
    Compute peak amplitude and area under the curve for each peak window.

    Parameters
    ----------
    psth_mean : np.ndarray
        2-D array of mean PSTH values (time-points × channels/trials).
    timestamps : np.ndarray
        1-D time axis (s) aligned with ``psth_mean`` rows.
    sampling_rate : float
        Sampling rate in Hz (currently unused; reserved for future use).
    peak_startPoint : array-like
        Start times (s) for each peak window.
    peak_endPoint : array-like
        End times (s) for each peak window.

    Returns
    -------
    peak_and_area : OrderedDict
        Ordered mapping of metric name → scalar or array value.  Keys follow
        the pattern ``'peak_pos_N'``, ``'peak_neg_N'``, and ``'area_N'`` for
        each validated window ``N``.
    """

    peak_startPoint, peak_endPoint = validate_peak_windows(peak_starts=peak_startPoint, peak_ends=peak_endPoint)

    peak_and_area = OrderedDict()

    if peak_startPoint.shape[0] == 0 or peak_endPoint.shape[0] == 0:
        peak_and_area["peak"] = np.nan
        peak_and_area["area"] = np.nan

    for i in range(peak_startPoint.shape[0]):
        startPtForPeak = np.where(timestamps >= peak_startPoint[i])[0]
        endPtForPeak = np.where(timestamps >= peak_endPoint[i])[0]
        if len(startPtForPeak) >= 1 and len(endPtForPeak) >= 1:
            peakPoint_pos = startPtForPeak[0] + np.argmax(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peakPoint_neg = startPtForPeak[0] + np.argmin(psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0)
            peak_and_area["peak_pos_" + str(i + 1)] = np.amax(psth_mean[peakPoint_pos], axis=0)
            peak_and_area["peak_neg_" + str(i + 1)] = np.amin(psth_mean[peakPoint_neg], axis=0)
            peak_and_area["area_" + str(i + 1)] = np.trapezoid(
                psth_mean[startPtForPeak[0] : endPtForPeak[0], :], axis=0
            )
        else:
            peak_and_area["peak_" + str(i + 1)] = np.nan
            peak_and_area["area_" + str(i + 1)] = np.nan

    return peak_and_area
