import logging
from typing import Literal

import numpy as np
import statsmodels.api as sm
from scipy import signal as ss
from scipy.optimize import minimize_scalar

from .control_channel import helper_create_control_channel
from ..utils.validation import validate_window_bounds

logger = logging.getLogger(__name__)

# When the control fit carries a photobleaching term, its decay constant is measured in samples,
# not seconds: a chunk is a contiguous run at a fixed sampling rate, so the sample index is the same
# axis up to that scale factor and the fitted curve is identical either way. It is searched over a
# log-spaced grid spanning this fraction of the chunk up to its full length. A constant longer than
# the chunk cannot be estimated from it — that is the regime where the exponential flattens into a
# straight line and the amplitude and offset are free to grow without bound as long as they cancel.
SHORTEST_DECAY_CONSTANT_FRACTION = 1 / 1000
DECAY_CONSTANT_GRID_SIZE = 50


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

    # In baseline-epoch mode, estimate the fit coefficients once from the fit window and reuse them
    # for every chunk below, instead of re-fitting per chunk.
    fit_coefficients = None
    if control_fit_window_mode == "baseline epoch":
        fit_coefficients = estimate_baseline_epoch_coefficients(
            control,
            signal,
            tsNew,
            coords,
            filter_window,
            control_fit_method,
            control_fit_window_start,
            control_fit_window_end,
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
                fit_coefficients=fit_coefficients,
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


def execute_controlFit_dff(
    control: np.ndarray,
    signal: np.ndarray,
    isosbestic_control: bool,
    filter_window: int,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
    *,
    fit_coefficients: tuple[float, float] | None = None,
    photobleaching_detrend: bool = False,
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
    fit_coefficients : tuple of float or None, optional
        Precomputed ``(slope, intercept)`` to apply instead of estimating from this
        segment. When None (default), coefficients are estimated from this segment
        (current per-chunk behavior).
    photobleaching_detrend : bool, optional
        When True, the fit gains an exponential term for the photobleaching the control
        channel does not see. Default is False.

    Returns
    -------
    norm_data : np.ndarray
        Normalized dF/F array.
    control_fit : np.ndarray
        Fitted control channel aligned with ``signal``.
    """

    if isosbestic_control == False:
        signal_smooth = filterSignal(filter_window, signal)
        if fit_coefficients is None:
            control_fit = controlFit(control, signal_smooth, method=control_fit_method)
        else:
            control_fit = apply_control_fit(control, *fit_coefficients)
        norm_data = deltaFF(signal_smooth, control_fit)
    else:
        control_smooth = filterSignal(filter_window, control)
        signal_smooth = filterSignal(filter_window, signal)
        if fit_coefficients is None:
            control_fit = controlFit(
                control_smooth,
                signal_smooth,
                method=control_fit_method,
                photobleaching_detrend=photobleaching_detrend,
            )
        else:
            control_fit = apply_control_fit(control_smooth, *fit_coefficients)
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

    difference = np.subtract(signal, control)
    normData = np.divide(difference, control)
    normData = normData * 100

    return normData


def estimate_control_fit_coefficients(
    control: np.ndarray, signal: np.ndarray, *, method: Literal["IRWLS", "OLS"] = "IRWLS"
) -> tuple[float, float]:
    """
    Estimate the linear control-to-signal fit coefficients.

    Parameters
    ----------
    control : np.ndarray
        Control channel array to fit from.
    signal : np.ndarray
        Signal channel array to fit to.
    method : str, optional
        Regression method used to fit the line. ``'IRWLS'`` (default) uses
        Iteratively Re-Weighted Least Squares with a Tukey bisquare norm, which
        down-weights outliers and matches ordinary least squares on clean data.
        ``'OLS'`` uses ordinary least squares.

    Returns
    -------
    slope : float
        Multiplicative coefficient mapping control onto the signal scale.
    intercept : float
        Additive offset of the linear fit.
    """

    intercept, slope = fit_linear_model(design_matrix=control, signal=signal, method=method)
    return float(slope), float(intercept)


def apply_control_fit(control: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """
    Apply linear fit coefficients to a control array.

    Parameters
    ----------
    control : np.ndarray
        Control channel array.
    slope : float
        Multiplicative coefficient from ``estimate_control_fit_coefficients``.
    intercept : float
        Additive offset from ``estimate_control_fit_coefficients``.

    Returns
    -------
    fitted_values : np.ndarray
        Control values projected onto the signal scale.
    """

    return (slope * control) + intercept


def fit_linear_model(
    *, design_matrix: np.ndarray, signal: np.ndarray, method: Literal["IRWLS", "OLS"] = "IRWLS"
) -> np.ndarray:
    """
    Fit ``signal ~ intercept + design_matrix`` and return the coefficients.

    Parameters
    ----------
    design_matrix : np.ndarray
        Regressor columns, without an intercept column; shape ``(n_samples, n_regressors)``.
    signal : np.ndarray
        Values to fit to.
    method : str, optional
        Regression method. ``'IRWLS'`` (default) uses Iteratively Re-Weighted Least Squares
        with a Tukey bisquare norm, which down-weights outliers and matches ordinary least
        squares on clean data. ``'OLS'`` uses ordinary least squares.

    Returns
    -------
    np.ndarray
        Coefficients, intercept first, then one per column of ``design_matrix``.
    """
    design_with_intercept = sm.add_constant(design_matrix)
    if method == "OLS":
        return np.linalg.lstsq(design_with_intercept, signal, rcond=None)[0]
    elif method == "IRWLS":
        return sm.RLM(signal, design_with_intercept, M=sm.robust.norms.TukeyBiweight()).fit().params
    else:
        raise ValueError(
            f"control fitting method '{method}' is not recognized. Use 'IRWLS' (robust, default) or 'OLS'."
        )


def _fit_with_decay_constant(
    control: np.ndarray,
    signal: np.ndarray,
    sample_index: np.ndarray,
    decay_constant: float,
    method: Literal["IRWLS", "OLS"],
) -> tuple[np.ndarray, float]:
    """Fit control plus bleaching with the decay constant fixed; return the fit and its error."""
    bleaching = np.exp(-sample_index / decay_constant)
    intercept, slope, amplitude = fit_linear_model(
        design_matrix=np.column_stack([control, bleaching]), signal=signal, method=method
    )
    fitted_values = intercept + slope * control + amplitude * bleaching
    return fitted_values, float(np.sum((signal - fitted_values) ** 2))


def fit_control_with_bleaching(
    control: np.ndarray, signal: np.ndarray, *, method: Literal["IRWLS", "OLS"] = "IRWLS"
) -> np.ndarray:
    """
    Fit ``signal = slope*control + intercept + amplitude*exp(-i/decay_constant)``.

    The exponential term stands for the photobleaching the control channel does not see, which no
    rescaling of the control can remove. It is estimated together with the linear coefficients
    rather than subtracted afterwards, because fitting the control on its own lets the slope
    absorb the decay — which is what leaves a residual in the dF/F in the first place.

    The model is linear once the decay constant is fixed, so the constant is found by a search
    over a log-spaced grid refined by a bounded scalar minimization, and the coefficients at each
    candidate come from an ordinary least-squares fit. Only the final fit uses ``method``: the
    error curve that locates the decay constant is insensitive to the robust weighting, and an
    iteratively re-weighted fit at every candidate would dominate the runtime on a long recording.

    Parameters
    ----------
    control : np.ndarray
        Control channel array.
    signal : np.ndarray
        Signal channel array.
    method : str, optional
        Regression method for the final fit; ``'IRWLS'`` (default) or ``'OLS'``.

    Returns
    -------
    fitted_values : np.ndarray
        Fitted baseline, aligned with ``signal``.
    """
    sample_index = np.arange(control.shape[0], dtype=float)
    length = float(sample_index[-1])

    def sum_of_squared_errors(decay_constant: float) -> float:
        return _fit_with_decay_constant(control, signal, sample_index, decay_constant, "OLS")[1]

    candidates = np.geomspace(length * SHORTEST_DECAY_CONSTANT_FRACTION, length, DECAY_CONSTANT_GRID_SIZE)
    errors = [sum_of_squared_errors(candidate) for candidate in candidates]
    best = int(np.argmin(errors))

    # Refine within the bracket the neighbouring grid points form, so the grid only has to get close.
    lower, upper = candidates[max(best - 1, 0)], candidates[min(best + 1, candidates.shape[0] - 1)]
    if lower < upper:
        decay_constant = float(minimize_scalar(sum_of_squared_errors, bounds=(lower, upper), method="bounded").x)
    else:
        decay_constant = float(candidates[best])

    logger.debug(f"Control fit bleaching term: decay constant {decay_constant:.4g} samples")
    return _fit_with_decay_constant(control, signal, sample_index, decay_constant, method)[0]


def controlFit(
    control: np.ndarray,
    signal: np.ndarray,
    *,
    method: Literal["IRWLS", "OLS"] = "IRWLS",
    photobleaching_detrend: bool = False,
) -> np.ndarray:
    """
    Fit a model from control to signal and return the fitted values.

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
    photobleaching_detrend : bool, optional
        When True, the fit gains an exponential decay term for the photobleaching the control
        channel does not see, so the fitted baseline absorbs the residual decay that would
        otherwise survive into the dF/F. Default is False.

    Returns
    -------
    fitted_values : np.ndarray
        Fitted control values, projected onto the signal scale.
    """

    if photobleaching_detrend == True:
        return fit_control_with_bleaching(control, signal, method=method)
    slope, intercept = estimate_control_fit_coefficients(control, signal, method=method)
    return apply_control_fit(control, slope, intercept)


def estimate_baseline_epoch_coefficients(
    control: np.ndarray,
    signal: np.ndarray,
    tsNew: np.ndarray,
    coords: np.ndarray,
    filter_window: int,
    control_fit_method: Literal["IRWLS", "OLS"],
    control_fit_window_start: float,
    control_fit_window_end: float,
) -> tuple[float, float]:
    """
    Estimate control-fit coefficients from the baseline fit window.

    The estimation set is the fit window intersected with the data retained by artifact
    removal, so an artifact inside the window cannot corrupt the coefficient estimate.

    Parameters
    ----------
    control, signal : np.ndarray
        Full-trace control and signal channel arrays.
    tsNew : np.ndarray
        Corrected timestamp array aligned with ``control`` and ``signal``.
    coords : np.ndarray
        Shape ``(N, 2)`` good-chunk boundary array from artifact removal.
    filter_window : int
        Moving-average filter window length; 0 disables filtering.
    control_fit_method : {'IRWLS', 'OLS'}
        Regression method for the fit.
    control_fit_window_start, control_fit_window_end : float
        Fit-window bounds (s).

    Returns
    -------
    slope, intercept : float
        Linear fit coefficients estimated from the fit window.

    Raises
    ------
    ValueError
        If the window bounds are invalid, or if no retained data falls inside the window.
    """
    validate_window_bounds(
        start=control_fit_window_start,
        end=control_fit_window_end,
        ts_min=float(tsNew[0]),
        ts_max=float(tsNew[-1]),
        start_name="controlFitWindowStart",
        end_name="controlFitWindowEnd",
        range_label="signal timespan",
    )

    retained_indices = np.concatenate(
        [np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0] for i in range(coords.shape[0])]
    )
    window_indices = np.where((tsNew >= control_fit_window_start) & (tsNew <= control_fit_window_end))[0]
    estimation_indices = np.intersect1d(window_indices, retained_indices)
    if estimation_indices.size == 0:
        message = (
            f"control fit window [{control_fit_window_start}, {control_fit_window_end}]s contains no data after "
            f"artifact removal within the retained range [{tsNew[0]}, {tsNew[-1]}]s; no points remain to estimate "
            "the fit. Widen the window or adjust artifact removal so the fit window overlaps retained data."
        )
        logger.error(message)
        raise ValueError(message)

    control_smooth = filterSignal(filter_window, control)
    signal_smooth = filterSignal(filter_window, signal)
    return estimate_control_fit_coefficients(
        control_smooth[estimation_indices], signal_smooth[estimation_indices], method=control_fit_method
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
