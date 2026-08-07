"""Fitting the control channel to the signal channel.

The fitted control is the estimated baseline the signal is measured against: dF/F divides by it,
and everything downstream is derived from the result. Two independent choices shape it:

* **Which regressors.** The control alone, or the control plus an exponential decay term standing
  for the photobleaching the control channel does not see.
* **Which samples the model is fit on.** The segment being corrected, or a baseline epoch chosen
  by the user.

The first is the ``photobleaching_detrend`` argument to :func:`fit`; the second is the ``indices``
argument shared by :func:`fit` and :func:`predict`, with :func:`select_fit_window_indices` picking
them for a baseline epoch. So the two compose without either knowing about the other.

:func:`execute` composes them over a whole recording, fitting either per retained chunk or once
on a baseline epoch. It takes channels that have already been smoothed, and a control that has
already been chosen — whether that control was measured at the isosbestic wavelength or synthesized
from the signal is the caller's concern, not this module's.

Both :func:`fit` and :func:`predict` take the full trace and select from it, rather than a segment.
The photobleaching term is exponential in time, so it needs an origin, and taking the whole trace
makes that origin the start of the recording for every caller. That is what lets a model fit on a
baseline epoch be applied across the rest of the recording: the two calls share an axis without
the caller having to construct or carry one. (Sample position stands in for time throughout — the
trace is uniformly sampled, so the two axes differ only by the sampling rate.)
"""

import logging
from typing import Literal, NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

from ..utils.validation import validate_window_bounds

logger = logging.getLogger(__name__)

# The decay constant is searched in log space, since it ranges over orders of magnitude, and is
# bounded only to keep the search finite. The upper bound sits well past the fitted span so it does
# not act as a constraint: beyond a few multiples of the span every larger constant describes the
# same near-straight line, so widening it further changes nothing.
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


def predict(model: ControlFitModel, control: np.ndarray, *, indices: np.ndarray | None = None) -> np.ndarray:
    """
    Evaluate a fitted model over part of the recording.

    Parameters
    ----------
    model : ControlFitModel
        Coefficients from :func:`fit`.
    control : np.ndarray
        Full control channel trace.
    indices : np.ndarray or None, optional
        Indices of the samples to predict at. Default is the whole trace.

    Returns
    -------
    np.ndarray
        The predicted baseline, aligned with ``indices``.
    """
    indices = np.arange(control.shape[0]) if indices is None else indices

    predicted = model.intercept + model.slope * control[indices]
    if model.decay_constant is not None:
        predicted = predicted + model.bleaching_amplitude * np.exp(-indices / model.decay_constant)
    return predicted


def _fit_at_decay_constant(
    control: np.ndarray,
    signal: np.ndarray,
    sample_index: np.ndarray,
    decay_constant: float | None,
    method: Literal["IRWLS", "OLS"],
) -> tuple[ControlFitModel, np.ndarray]:
    """Fit the coefficients with the decay constant held fixed, or absent when it is None.

    Every coefficient enters the model linearly once the decay constant is fixed, so this is a
    single least-squares solve: no starting guess, no iteration, and the same code path whether
    or not there is a bleaching term.

    Returns the coefficients along with the values they predict on the samples fit, which the
    solve produces anyway and which the decay-constant search scores.
    """
    if decay_constant is None:
        regressors = control
    else:
        regressors = np.column_stack([control, np.exp(-sample_index / decay_constant)])

    # sm.add_constant prepends the intercept column, so the coefficients come back intercept-first.
    design_matrix = sm.add_constant(regressors)
    if method == "OLS":
        coefficients = np.linalg.lstsq(design_matrix, signal, rcond=None)[0]
    elif method == "IRWLS":
        coefficients = sm.RLM(signal, design_matrix, M=sm.robust.norms.TukeyBiweight()).fit().params
    else:
        raise ValueError(
            f"control fitting method '{method}' is not recognized. Use 'IRWLS' (robust, default) or 'OLS'."
        )

    intercept, slope, *bleaching_amplitude = coefficients
    model = ControlFitModel(
        slope=float(slope),
        intercept=float(intercept),
        bleaching_amplitude=float(bleaching_amplitude[0]) if bleaching_amplitude else 0.0,
        decay_constant=decay_constant,
    )
    return model, design_matrix @ coefficients


def _search_decay_constant(control: np.ndarray, signal: np.ndarray, sample_index: np.ndarray) -> float:
    """Find the decay constant minimizing the fit error, searching in log space.

    The decay constant is the one parameter the model is not linear in, so it is the only one that
    needs a search; every other coefficient is solved exactly at each candidate.

    Candidates are scored with ordinary least squares even when the final fit is robust: the error
    curve locating the constant is insensitive to the weighting, and an iteratively re-weighted fit
    at every candidate would dominate the runtime on a long recording.
    """
    span = float(sample_index[-1] - sample_index[0])

    def sum_of_squared_errors(log_decay_constant: float) -> float:
        decay_constant = float(np.exp(log_decay_constant))
        _, predicted = _fit_at_decay_constant(control, signal, sample_index, decay_constant, "OLS")
        return float(np.sum((signal - predicted) ** 2))

    result = minimize_scalar(
        sum_of_squared_errors,
        bounds=(
            np.log(span * SHORTEST_DECAY_CONSTANT_FRACTION),
            np.log(span * LONGEST_DECAY_CONSTANT_MULTIPLE),
        ),
        method="bounded",
    )
    return float(np.exp(result.x))


def fit(
    control: np.ndarray,
    signal: np.ndarray,
    *,
    indices: np.ndarray | None = None,
    method: Literal["IRWLS", "OLS"] = "IRWLS",
    photobleaching_detrend: bool = False,
) -> ControlFitModel:
    """
    Fit the control-to-signal model.

    Parameters
    ----------
    control : np.ndarray
        Full control channel trace.
    signal : np.ndarray
        Full signal channel trace.
    indices : np.ndarray or None, optional
        Indices of the samples to fit on: the segment being corrected for a per-segment fit, or
        the baseline epoch for a frozen one. Default is the whole trace.
    method : str, optional
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``.
    photobleaching_detrend : bool, optional
        When True, the model gains an exponential decay term for the photobleaching the control
        channel does not see. It is fit together with the linear coefficients, since fitting the
        control on its own lets its slope absorb the decay. Default is False.

    Returns
    -------
    ControlFitModel
        The fitted coefficients.
    """
    indices = np.arange(control.shape[0]) if indices is None else indices
    control, signal = control[indices], signal[indices]

    decay_constant = None
    if photobleaching_detrend == True:
        decay_constant = _search_decay_constant(control, signal, indices)
        logger.debug(f"Control fit bleaching term: decay constant {decay_constant:.4g} samples")

    model, _ = _fit_at_decay_constant(control, signal, indices, decay_constant, method)
    return model


def select_fit_window_indices(
    tsNew: np.ndarray,
    chunk_index_list: list[np.ndarray],
    control_fit_window_start: float,
    control_fit_window_end: float,
) -> np.ndarray:
    """
    Pick the samples a baseline-epoch model is fit on.

    The fit set is the fit window intersected with the data retained by artifact removal, so an
    artifact inside the window cannot corrupt the coefficients.

    Parameters
    ----------
    tsNew : np.ndarray
        Corrected timestamp array aligned with the channel data.
    chunk_index_list : list of np.ndarray
        Indices of each retained chunk, from ``retained_chunk_indices``.
    control_fit_window_start, control_fit_window_end : float
        Fit-window bounds (s).

    Returns
    -------
    np.ndarray
        Indices into ``tsNew`` of the samples to fit on.

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

    window_indices = np.where((tsNew >= control_fit_window_start) & (tsNew <= control_fit_window_end))[0]
    fit_indices = np.intersect1d(window_indices, np.concatenate(chunk_index_list))
    if fit_indices.size == 0:
        message = (
            f"control fit window [{control_fit_window_start}, {control_fit_window_end}]s contains no data after "
            f"artifact removal within the retained range [{tsNew[0]}, {tsNew[-1]}]s; no points remain to estimate "
            "the fit. Widen the window or adjust artifact removal so the fit window overlaps retained data."
        )
        logger.error(message)
        raise ValueError(message)

    return fit_indices


def execute(
    control: np.ndarray,
    signal: np.ndarray,
    chunk_index_list: list[np.ndarray],
    *,
    fit_indices: np.ndarray | None = None,
    method: Literal["IRWLS", "OLS"] = "IRWLS",
    photobleaching_detrend: bool = False,
) -> np.ndarray:
    """
    Produce the fitted baseline across every retained chunk of a recording.

    Parameters
    ----------
    control : np.ndarray
        Full smoothed control channel trace.
    signal : np.ndarray
        Full smoothed signal channel trace.
    chunk_index_list : list of np.ndarray
        Indices of each retained chunk, from ``retained_chunk_indices``.
    fit_indices : np.ndarray or None, optional
        Samples to fit a single frozen model on, applied to every chunk. Default is None, which
        fits each chunk on itself instead.
    method : str, optional
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``.
    photobleaching_detrend : bool, optional
        When True, the fit gains an exponential decay term. Default is False.

    Returns
    -------
    np.ndarray
        Full-length fitted baseline, NaN wherever no chunk covers the sample.
    """
    frozen_model = None
    if fit_indices is not None:
        frozen_model = fit(
            control,
            signal,
            indices=fit_indices,
            method=method,
            photobleaching_detrend=photobleaching_detrend,
        )

    fitted_control = np.full(control.shape[0], np.nan)
    for chunk_indices in chunk_index_list:
        model = frozen_model
        if model is None:
            model = fit(
                control,
                signal,
                indices=chunk_indices,
                method=method,
                photobleaching_detrend=photobleaching_detrend,
            )
        fitted_control[chunk_indices] = predict(model, control, indices=chunk_indices)

    return fitted_control
