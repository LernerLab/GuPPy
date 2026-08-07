"""Fitting the control channel to the signal channel, and normalizing against that fit.

The fitted control is the estimated baseline the signal is measured against: dF/F divides by it,
and everything downstream is derived from the result. Two independent choices shape it:

* **Which regressors.** The control alone, or the control plus an exponential decay term standing
  for the photobleaching the control channel does not see.
* **Which samples estimate the coefficients.** The segment being corrected, or a baseline epoch
  chosen by the user.

Both are arguments to :func:`estimate_control_fit`, so all four combinations compose. The estimated
coefficients travel as a :class:`ControlFitModel`, which :func:`apply_control_fit` evaluates over
whatever segment is being corrected — so a model estimated from a baseline epoch can be applied
across the whole recording.
"""

import logging
from typing import Literal, NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import signal as ss
from scipy.optimize import minimize_scalar

from ..utils.validation import validate_window_bounds

logger = logging.getLogger(__name__)

# The decay constant is searched in log space, since it ranges over orders of magnitude, and is
# bounded only to keep the search finite. The upper bound sits well past the estimation span so it
# does not act as a constraint: beyond a few multiples of the span every larger constant describes
# the same near-straight line, so widening it further changes nothing.
SHORTEST_DECAY_CONSTANT_FRACTION = 1 / 1000
LONGEST_DECAY_CONSTANT_MULTIPLE = 10


class ControlFitModel(NamedTuple):
    """Coefficients of a fitted control-to-signal model.

    Attributes
    ----------
    slope : float
        Multiplicative coefficient mapping the control onto the signal scale.
    intercept : float
        Additive offset of the fit.
    bleaching_amplitude : float
        Size of the photobleaching term at sample 0; 0.0 when the model has no such term.
    decay_constant : float or None
        Decay constant of the photobleaching term, in samples; None when the model has no
        such term.
    """

    slope: float
    intercept: float
    bleaching_amplitude: float = 0.0
    decay_constant: float | None = None


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
) -> tuple[ControlFitModel, float]:
    """Fit control plus bleaching with the decay constant held fixed; return model and error."""
    bleaching = np.exp(-sample_index / decay_constant)
    intercept, slope, amplitude = fit_linear_model(
        design_matrix=np.column_stack([control, bleaching]), signal=signal, method=method
    )
    fitted_values = intercept + slope * control + amplitude * bleaching
    model = ControlFitModel(float(slope), float(intercept), float(amplitude), float(decay_constant))
    return model, float(np.sum((signal - fitted_values) ** 2))


def _estimate_decay_constant(control: np.ndarray, signal: np.ndarray, sample_index: np.ndarray) -> float:
    """Find the decay constant minimizing the fit error, searching in log space."""
    span = float(sample_index[-1] - sample_index[0])

    def sum_of_squared_errors(log_decay_constant: float) -> float:
        return _fit_with_decay_constant(control, signal, sample_index, np.exp(log_decay_constant), "OLS")[1]

    result = minimize_scalar(
        sum_of_squared_errors,
        bounds=(
            np.log(span * SHORTEST_DECAY_CONSTANT_FRACTION),
            np.log(span * LONGEST_DECAY_CONSTANT_MULTIPLE),
        ),
        method="bounded",
    )
    return float(np.exp(result.x))


def estimate_control_fit(
    control: np.ndarray,
    signal: np.ndarray,
    *,
    sample_index: np.ndarray,
    method: Literal["IRWLS", "OLS"] = "IRWLS",
    photobleaching_detrend: bool = False,
) -> ControlFitModel:
    """
    Estimate the control-to-signal fit from the samples given.

    Pass the samples the coefficients should be estimated from: the whole segment being
    corrected for a per-segment fit, or a subset for a baseline-epoch fit. ``sample_index``
    positions those samples on the recording's sample axis, so a model estimated from one span
    evaluates correctly over another.

    Parameters
    ----------
    control : np.ndarray
        Control channel values to fit from.
    signal : np.ndarray
        Signal channel values to fit to.
    sample_index : np.ndarray
        Position of each sample on the recording's sample axis. Only the photobleaching term
        depends on it; the linear coefficients do not.
    method : str, optional
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``.
    photobleaching_detrend : bool, optional
        When True, the model gains an exponential decay term for the photobleaching the control
        channel does not see. It is estimated together with the linear coefficients, since
        fitting the control on its own lets its slope absorb the decay. Default is False.

    Returns
    -------
    ControlFitModel
        The estimated coefficients.
    """
    if photobleaching_detrend == False:
        intercept, slope = fit_linear_model(design_matrix=control, signal=signal, method=method)
        return ControlFitModel(float(slope), float(intercept))

    # The model is linear once the decay constant is fixed, so only that one parameter needs a
    # search. Its candidates are scored with ordinary least squares: the error curve locating it is
    # insensitive to the robust weighting, and an iteratively re-weighted fit at every candidate
    # would dominate the runtime on a long recording. The final fit below uses the chosen method.
    decay_constant = _estimate_decay_constant(control, signal, sample_index)
    model, _ = _fit_with_decay_constant(control, signal, sample_index, decay_constant, method)
    logger.debug(f"Control fit bleaching term: decay constant {model.decay_constant:.4g} samples")
    return model


def apply_control_fit(model: ControlFitModel, control: np.ndarray, sample_index: np.ndarray) -> np.ndarray:
    """
    Evaluate a fitted model over a segment.

    Parameters
    ----------
    model : ControlFitModel
        Coefficients from :func:`estimate_control_fit`.
    control : np.ndarray
        Control channel values for the segment.
    sample_index : np.ndarray
        Position of each sample on the recording's sample axis, on the same basis used to
        estimate the model.

    Returns
    -------
    np.ndarray
        The fitted baseline, aligned with ``control``.
    """
    fitted_values = model.intercept + model.slope * control
    if model.decay_constant is not None:
        fitted_values = fitted_values + model.bleaching_amplitude * np.exp(-sample_index / model.decay_constant)
    return fitted_values


def controlFit(
    control: np.ndarray,
    signal: np.ndarray,
    *,
    method: Literal["IRWLS", "OLS"] = "IRWLS",
    photobleaching_detrend: bool = False,
) -> np.ndarray:
    """
    Fit a model from control to signal over one segment and return the fitted values.

    Parameters
    ----------
    control : np.ndarray
        Control channel array.
    signal : np.ndarray
        Signal channel array.
    method : str, optional
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``.
    photobleaching_detrend : bool, optional
        When True, the fit gains an exponential decay term for the photobleaching the control
        channel does not see. Default is False.

    Returns
    -------
    fitted_values : np.ndarray
        Fitted control values, projected onto the signal scale.
    """
    sample_index = np.arange(control.shape[0], dtype=float)
    model = estimate_control_fit(
        control,
        signal,
        sample_index=sample_index,
        method=method,
        photobleaching_detrend=photobleaching_detrend,
    )
    return apply_control_fit(model, control, sample_index)


def estimate_baseline_epoch_model(
    control: np.ndarray,
    signal: np.ndarray,
    tsNew: np.ndarray,
    coords: np.ndarray,
    filter_window: int,
    control_fit_method: Literal["IRWLS", "OLS"],
    control_fit_window_start: float,
    control_fit_window_end: float,
    *,
    photobleaching_detrend: bool = False,
) -> ControlFitModel:
    """
    Estimate the control fit from the baseline fit window.

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
    photobleaching_detrend : bool, optional
        When True, the model gains an exponential decay term. Default is False.

    Returns
    -------
    ControlFitModel
        Coefficients estimated from the fit window, on the full trace's sample axis.

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
    sample_index = np.arange(tsNew.shape[0], dtype=float)
    return estimate_control_fit(
        control_smooth[estimation_indices],
        signal_smooth[estimation_indices],
        sample_index=sample_index[estimation_indices],
        method=control_fit_method,
        photobleaching_detrend=photobleaching_detrend,
    )


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


def execute_controlFit_dff(
    control: np.ndarray,
    signal: np.ndarray,
    isosbestic_control: bool,
    filter_window: int,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
    *,
    sample_index: np.ndarray | None = None,
    model: ControlFitModel | None = None,
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
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``.
    sample_index : np.ndarray or None, optional
        Position of each sample on the recording's sample axis. Defaults to a range starting
        at 0, which is correct whenever the model is estimated from this same segment.
    model : ControlFitModel or None, optional
        Precomputed coefficients to apply instead of estimating from this segment. When None
        (default), coefficients are estimated from this segment.
    photobleaching_detrend : bool, optional
        When True, an estimated fit gains an exponential decay term. Ignored when ``model``
        is given, since that model already carries its own terms. Default is False.

    Returns
    -------
    norm_data : np.ndarray
        Normalized dF/F array.
    control_fit : np.ndarray
        Fitted control channel aligned with ``signal``.
    """
    if sample_index is None:
        sample_index = np.arange(control.shape[0], dtype=float)

    if isosbestic_control == False:
        control_smooth = control
    else:
        control_smooth = filterSignal(filter_window, control)
    signal_smooth = filterSignal(filter_window, signal)

    if model is None:
        model = estimate_control_fit(
            control_smooth,
            signal_smooth,
            sample_index=sample_index,
            method=control_fit_method,
            photobleaching_detrend=photobleaching_detrend,
        )
    control_fit = apply_control_fit(model, control_smooth, sample_index)

    return deltaFF(signal_smooth, control_fit), control_fit
