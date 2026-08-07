import logging
from typing import Literal

import numpy as np

from .control_channel import helper_create_control_channel
from .control_fit import (
    estimate_baseline_epoch_model,
    execute_controlFit_dff,
    validate_chunk_lengths_for_filtering,
)
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
    control_fit_window_mode: Literal["full trace", "baseline epoch"] = "full trace",
    control_fit_window_start: float = 0,
    control_fit_window_end: float = 0,
    photobleaching_detrend: bool = False,
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
    control_fit_window_mode : {'full trace', 'baseline epoch'}, optional
        Fitting mode. ``'full trace'`` (default) estimates fit coefficients within
        each artifact-removal chunk (current behavior). ``'baseline epoch'`` estimates
        coefficients once from the fit window (isosbestic control only) and applies those
        frozen coefficients across every chunk, so a step-change (e.g. a drug injection)
        outside the window cannot corrupt the fit.
    control_fit_window_start : float, optional
        Fit-window start (s); used only in ``'baseline epoch'`` mode.
    control_fit_window_end : float, optional
        Fit-window end (s); used only in ``'baseline epoch'`` mode.
    photobleaching_detrend : bool, optional
        When True, the control fit gains an exponential term for the photobleaching the
        control channel does not see, so the fitted baseline absorbs the residual decay that
        would otherwise survive into the dF/F. Default is False.

    Returns
    -------
    z_scores : np.ndarray
        Computed z-score array.
    normalized_data : np.ndarray
        Normalized dF/F array (NaN-filled where artifacts were removed).
    fitted_control : np.ndarray
        Fitted control channel array (NaN-filled where artifacts were removed).
    synthetic_control : np.ndarray or None
        Synthetic control array (when ``isosbestic_control=False``); None otherwise.
    """
    if (control == 0).all() == True:
        control = np.zeros(tsNew.shape[0])

    validate_chunk_lengths_for_filtering(tsNew, coords, filter_window)

    z_scores = np.array([])
    normalized_data = np.full(tsNew.shape[0], np.nan)
    fitted_control = np.full(tsNew.shape[0], np.nan)
    synthetic_control = np.full(tsNew.shape[0], np.nan)
    # The photobleaching term is exponential in time, so the fit needs to know where each chunk
    # sits in the recording. Sample position stands in for time: the trace is uniformly sampled,
    # so the two axes differ only by the sampling rate.
    sample_index = np.arange(tsNew.shape[0], dtype=float)

    # In baseline-epoch mode, estimate the fit once from the fit window and reuse it for every
    # chunk below, instead of re-fitting per chunk.
    baseline_epoch_model = None
    if control_fit_window_mode == "baseline epoch":
        baseline_epoch_model = estimate_baseline_epoch_model(
            control,
            signal,
            tsNew,
            coords,
            filter_window,
            control_fit_method,
            control_fit_window_start,
            control_fit_window_end,
            photobleaching_detrend=photobleaching_detrend,
        )

    # for artifacts removal, each chunk which was selected by user is being processed individually and then
    # z-score is calculated
    for i in range(coords.shape[0]):
        chunk_indices = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        if isosbestic_control == False:
            control_segment = helper_create_control_channel(signal[chunk_indices], tsNew[chunk_indices], window=101)
            signal_segment = signal[chunk_indices]
            norm_data, control_fit = execute_controlFit_dff(
                control_segment, signal_segment, isosbestic_control, filter_window, control_fit_method
            )
            synthetic_control[chunk_indices] = control_segment
            if i < coords.shape[0] - 1:
                gap_indices = np.where((tsNew > coords[i, 1]) & (tsNew < coords[i + 1, 0]))[0]
                synthetic_control[gap_indices] = np.full(gap_indices.shape[0], np.nan)
        else:
            control_segment = control[chunk_indices]
            signal_segment = signal[chunk_indices]
            norm_data, control_fit = execute_controlFit_dff(
                control_segment,
                signal_segment,
                isosbestic_control,
                filter_window,
                control_fit_method,
                sample_index=sample_index[chunk_indices],
                model=baseline_epoch_model,
                photobleaching_detrend=photobleaching_detrend,
            )
        normalized_data[chunk_indices] = norm_data
        fitted_control[chunk_indices] = control_fit

    if artifactsRemovalMethod == "concatenate":
        normalized_data = normalized_data[~np.isnan(normalized_data)]
        fitted_control = fitted_control[~np.isnan(fitted_control)]
    z_score = z_score_computation(normalized_data, tsNew, zscore_method, baseline_start, baseline_end)
    z_scores = np.concatenate((z_scores, z_score))

    # handle the case if there are chunks being cut in the front and the end
    if isosbestic_control == False:
        coords = coords.flatten()
        # front chunk
        front_chunk_indices = np.where((tsNew >= tsNew[0]) & (tsNew < coords[0]))[0]
        synthetic_control[front_chunk_indices] = np.full(front_chunk_indices.shape[0], np.nan)
        # end chunk
        end_chunk_indices = np.where((tsNew > coords[-1]) & (tsNew <= tsNew[-1]))[0]
        synthetic_control[end_chunk_indices] = np.full(end_chunk_indices.shape[0], np.nan)
    else:
        synthetic_control = None

    return z_scores, normalized_data, fitted_control, synthetic_control


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

        baseline_indices = np.where((timestamps > baseline_start) & (timestamps < baseline_end))[0]
        if baseline_indices.shape[0] == 0:
            message = (
                f"No signal samples found in the baseline window "
                f"({baseline_start}, {baseline_end})s; "
                f"signal timespan is [{ts_min:.4g}, {ts_max:.4g}]s — "
                f"choose baselineWindowStart and baselineWindowEnd within this range."
            )
            logger.error(message)
            raise ValueError(message)
        else:
            baseline_mean = np.nanmean(dff[baseline_indices])
            baseline_std = np.nanstd(dff[baseline_indices])
            numerator = np.subtract(dff, baseline_mean)
            zscore = np.divide(numerator, baseline_std)
    else:
        median = np.median(dff)
        mad = np.median(np.abs(dff - median))
        numerator = 0.6745 * (dff - median)
        zscore = np.divide(numerator, mad)

    return zscore
