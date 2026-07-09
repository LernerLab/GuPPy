import logging
from typing import Literal

import numpy as np
import statsmodels.api as sm
from scipy import signal as ss

from .control_channel import helper_create_control_channel
from ..utils.validation import validate_window_bounds

logger = logging.getLogger(__name__)


def compute_z_score(
    control: np.ndarray,
    signal: np.ndarray,
    tsNew: np.ndarray,
    coords: np.ndarray,
    artifactsRemovalMethod: str,
    filter_window: int,
    isosbestic_control: bool,
    zscore_method: str,
    baseline_start: float,
    baseline_end: float,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Compute the z-score and dF/F for a control/signal channel pair.

    Parameters
    ----------
    control : np.ndarray
        Control channel data array.
    signal : np.ndarray
        Signal channel data array.
    tsNew : np.ndarray
        Corrected timestamp array aligned with ``control`` and ``signal``.
    coords : np.ndarray
        Shape ``(N, 2)`` good-chunk boundary array from artifact removal.
    artifactsRemovalMethod : str
        Artifact removal method used upstream; ``'concatenate'`` triggers NaN removal.
    filter_window : int
        Moving-average filter window length; 0 disables filtering.
    isosbestic_control : bool
        When False, a synthetic control is fit from the signal.
    zscore_method : str
        Z-score method; one of ``'standard z-score'``, ``'baseline z-score'``,
        or any other value (uses median/MAD).
    baseline_start : float
        Baseline window start (s); used only for ``'baseline z-score'``.
    baseline_end : float
        Baseline window end (s); used only for ``'baseline z-score'``.
    control_fit_method : str, optional
        Regression method for fitting the control to the signal; ``'IRWLS'``
        (default) or ``'OLS'``.

    Returns
    -------
    z_score_arr : np.ndarray
        Computed z-score array.
    norm_data_arr : np.ndarray
        Normalized dF/F array (NaN-filled where artifacts were removed).
    control_fit_arr : np.ndarray
        Fitted control channel array (NaN-filled where artifacts were removed).
    temp_control_arr : np.ndarray or None
        Synthetic control array (when ``isosbestic_control=False``); None otherwise.
    """
    if (control == 0).all() == True:
        control = np.zeros(tsNew.shape[0])

    z_score_arr = np.array([])
    norm_data_arr = np.full(tsNew.shape[0], np.nan)
    control_fit_arr = np.full(tsNew.shape[0], np.nan)
    temp_control_arr = np.full(tsNew.shape[0], np.nan)

    # for artifacts removal, each chunk which was selected by user is being processed individually and then
    # z-score is calculated
    for i in range(coords.shape[0]):
        tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        if isosbestic_control == False:
            control_arr = helper_create_control_channel(signal[tsNew_index], tsNew[tsNew_index], window=101)
            signal_arr = signal[tsNew_index]
            norm_data, control_fit = execute_controlFit_dff(
                control_arr, signal_arr, isosbestic_control, filter_window, control_fit_method
            )
            temp_control_arr[tsNew_index] = control_arr
            if i < coords.shape[0] - 1:
                blank_index = np.where((tsNew > coords[i, 1]) & (tsNew < coords[i + 1, 0]))[0]
                temp_control_arr[blank_index] = np.full(blank_index.shape[0], np.nan)
        else:
            control_arr = control[tsNew_index]
            signal_arr = signal[tsNew_index]
            norm_data, control_fit = execute_controlFit_dff(
                control_arr, signal_arr, isosbestic_control, filter_window, control_fit_method
            )
        norm_data_arr[tsNew_index] = norm_data
        control_fit_arr[tsNew_index] = control_fit

    if artifactsRemovalMethod == "concatenate":
        norm_data_arr = norm_data_arr[~np.isnan(norm_data_arr)]
        control_fit_arr = control_fit_arr[~np.isnan(control_fit_arr)]
    z_score = z_score_computation(norm_data_arr, tsNew, zscore_method, baseline_start, baseline_end)
    z_score_arr = np.concatenate((z_score_arr, z_score))

    # handle the case if there are chunks being cut in the front and the end
    if isosbestic_control == False:
        coords = coords.flatten()
        # front chunk
        idx = np.where((tsNew >= tsNew[0]) & (tsNew < coords[0]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
        # end chunk
        idx = np.where((tsNew > coords[-1]) & (tsNew <= tsNew[-1]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
    else:
        temp_control_arr = None

    return z_score_arr, norm_data_arr, control_fit_arr, temp_control_arr


def execute_controlFit_dff(
    control: np.ndarray,
    signal: np.ndarray,
    isosbestic_control: bool,
    filter_window: int,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter channels, fit the control to the signal, and compute dF/F.

    Parameters
    ----------
    control : np.ndarray
        Control channel data (or synthetic control when no isosbestic exists).
    signal : np.ndarray
        Signal channel data.
    isosbestic_control : bool
        When True, both channels are filtered before fitting.
        When False, only the signal is filtered.
    filter_window : int
        Moving-average filter window length; 0 disables filtering.
    control_fit_method : str, optional
        Regression method for fitting the control to the signal; ``'IRWLS'``
        (default) or ``'OLS'``.

    Returns
    -------
    norm_data : np.ndarray
        Normalized dF/F array.
    control_fit : np.ndarray
        Fitted control channel aligned with ``signal``.
    """

    if isosbestic_control == False:
        signal_smooth = filterSignal(filter_window, signal)
        control_fit = controlFit(control, signal_smooth, method=control_fit_method)
        norm_data = deltaFF(signal_smooth, control_fit)
    else:
        control_smooth = filterSignal(filter_window, control)
        signal_smooth = filterSignal(filter_window, signal)
        control_fit = controlFit(control_smooth, signal_smooth, method=control_fit_method)
        norm_data = deltaFF(signal_smooth, control_fit)

    return norm_data, control_fit


def deltaFF(signal: np.ndarray, control: np.ndarray) -> np.ndarray:
    """
    Compute dF/F as ``(signal - control) / control * 100``.

    Parameters
    ----------
    signal : np.ndarray
        Filtered signal channel.
    control : np.ndarray
        Fitted control channel.

    Returns
    -------
    normData : np.ndarray
        Percent dF/F array.
    """

    res = np.subtract(signal, control)
    normData = np.divide(res, control)
    normData = normData * 100

    return normData


def controlFit(control: np.ndarray, signal: np.ndarray, *, method: Literal["IRWLS", "OLS"] = "IRWLS") -> np.ndarray:
    """
    Fit a linear model from control to signal and return the fitted values.

    Parameters
    ----------
    control : np.ndarray
        Control channel array.
    signal : np.ndarray
        Signal channel array.
    method : str, optional
        Regression method used to fit the line. ``'IRWLS'`` (default) uses
        Iteratively Re-Weighted Least Squares with a Tukey bisquare norm, which
        down-weights outliers and matches ordinary least squares on clean data.
        ``'OLS'`` uses ordinary least squares.

    Returns
    -------
    arr : np.ndarray
        Fitted control values (linear projection onto the signal scale).
    """

    if method == "OLS":
        p = np.polyfit(control, signal, 1)
        arr = (p[0] * control) + p[1]
        return arr
    elif method == "IRWLS":
        design_matrix = sm.add_constant(control)
        results = sm.RLM(signal, design_matrix, M=sm.robust.norms.TukeyBiweight()).fit()
        return np.asarray(results.fittedvalues)
    else:
        raise ValueError(
            f"control fitting method '{method}' is not recognized. Use 'IRWLS' (robust, default) or 'OLS'."
        )


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
        raise ValueError(
            f"filter_window={filter_window} is not a valid moving-average window. "
            "Use 0 to disable filtering, or an integer greater than 1 for the window length in samples."
        )


def z_score_computation(
    dff: np.ndarray, timestamps: np.ndarray, zscore_method: str, baseline_start: float, baseline_end: float
) -> np.ndarray:
    """
    Convert a dF/F array to z-scores using the specified method.

    Parameters
    ----------
    dff : np.ndarray
        1-D dF/F array.
    timestamps : np.ndarray
        Timestamp array aligned with ``dff``; used only for ``'baseline z-score'``.
    zscore_method : str
        One of ``'standard z-score'`` (whole-session mean/std),
        ``'baseline z-score'`` (baseline-window mean/std), or any other value
        (robust median/MAD estimator).
    baseline_start : float
        Start of the baseline window (s); used only for ``'baseline z-score'``.
    baseline_end : float
        End of the baseline window (s); used only for ``'baseline z-score'``.

    Returns
    -------
    zscore : np.ndarray
        Z-scored signal array.
    """
    if zscore_method == "standard z-score":
        numerator = np.subtract(dff, np.nanmean(dff))
        zscore = np.divide(numerator, np.nanstd(dff))
    elif zscore_method == "baseline z-score":
        ts_min = float(np.nanmin(timestamps))
        ts_max = float(np.nanmax(timestamps))
        validate_window_bounds(
            start=baseline_start,
            end=baseline_end,
            ts_min=ts_min,
            ts_max=ts_max,
            start_name="baselineWindowStart",
            end_name="baselineWindowEnd",
            range_label="signal timespan",
        )

        idx = np.where((timestamps > baseline_start) & (timestamps < baseline_end))[0]
        if idx.shape[0] == 0:
            msg = (
                f"No signal samples found in the baseline window "
                f"({baseline_start}, {baseline_end})s; "
                f"signal timespan is [{ts_min:.4g}, {ts_max:.4g}]s — "
                f"choose baselineWindowStart and baselineWindowEnd within this range."
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            baseline_mean = np.nanmean(dff[idx])
            baseline_std = np.nanstd(dff[idx])
            numerator = np.subtract(dff, baseline_mean)
            zscore = np.divide(numerator, baseline_std)
    else:
        median = np.median(dff)
        mad = np.median(np.abs(dff - median))
        numerator = 0.6745 * (dff - median)
        zscore = np.divide(numerator, mad)

    return zscore
