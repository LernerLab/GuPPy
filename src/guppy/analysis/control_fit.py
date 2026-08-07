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

The regressor choice decides how the fit is solved. Without the bleaching term every coefficient
is linear, so the fit is a least-squares solve with an exact answer. The bleaching term puts a
coefficient inside an exponent, which makes the whole fit nonlinear and hands it to ``curve_fit``.
That is also why the robust ``'IRWLS'`` method does not apply when detrending — ``curve_fit`` has
no robust norm — and the two are refused as a combination upstream.

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
from scipy.optimize import curve_fit

from ..utils.validation import validate_window_bounds

logger = logging.getLogger(__name__)

# The decay constant is bounded relative to the span of data being fit. The upper bound sits well
# past that span so it does not act as a constraint on any decay the data can actually show: beyond
# a few multiples of the span every larger constant describes the same near-straight line. Without
# it, though, an unidentifiable trace lets the offset and the amplitude run away in opposite
# directions chasing that straight line.
SHORTEST_DECAY_CONSTANT_FRACTION = 1 / 1000
LONGEST_DECAY_CONSTANT_MULTIPLE = 10
INITIAL_DECAY_CONSTANT_FRACTION = 1 / 4


class ControlFitModel(NamedTuple):
    """Coefficients of a fitted control-to-signal model.

    Attributes
    ----------
    slope : float
        Multiplicative coefficient mapping the control onto the signal scale.
    intercept : float
        Additive offset of the fit.
    bleaching_amplitude : float
        Size of the photobleaching term at ``sample_origin``; 0.0 when the model has no such term.
    decay_constant : float or None
        Decay constant of the photobleaching term, in samples; None when the model has no
        such term.
    sample_origin : float
        Sample the photobleaching term decays from — the first sample the model was fit on.
    """

    slope: float
    intercept: float
    bleaching_amplitude: float = 0.0
    decay_constant: float | None = None
    sample_origin: float = 0.0


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
        elapsed = indices - model.sample_origin
        predicted = predicted + model.bleaching_amplitude * np.exp(-elapsed / model.decay_constant)
    return predicted


def _fit_without_bleaching(control: np.ndarray, signal: np.ndarray, method: Literal["IRWLS", "OLS"]) -> ControlFitModel:
    """Fit ``signal ~ intercept + slope * control``.

    The model is linear in both coefficients, so this is solved exactly in one step rather than
    searched for.
    """
    # sm.add_constant prepends the intercept column, so the coefficients come back intercept-first.
    design_matrix = sm.add_constant(control)
    if method == "OLS":
        intercept, slope = np.linalg.lstsq(design_matrix, signal, rcond=None)[0]
    elif method == "IRWLS":
        intercept, slope = sm.RLM(signal, design_matrix, M=sm.robust.norms.TukeyBiweight()).fit().params
    else:
        raise ValueError(
            f"control fitting method '{method}' is not recognized. Use 'IRWLS' (robust, default) or 'OLS'."
        )

    return ControlFitModel(slope=float(slope), intercept=float(intercept))


def _bleaching_model(
    regressors: tuple[np.ndarray, np.ndarray],
    intercept: float,
    slope: float,
    bleaching_amplitude: float,
    decay_constant: float,
) -> np.ndarray:
    """``intercept + slope * control + bleaching_amplitude * exp(-elapsed / decay_constant)``."""
    control, elapsed = regressors
    return intercept + slope * control + bleaching_amplitude * np.exp(-elapsed / decay_constant)


def _fit_with_bleaching(control: np.ndarray, signal: np.ndarray, sample_index: np.ndarray) -> ControlFitModel:
    """Fit the same model plus an exponential decay term, by nonlinear least squares.

    The decay constant sits inside an exponent, so unlike every other coefficient it cannot be
    solved for directly. That makes the whole fit nonlinear, and ``curve_fit`` needs a starting
    guess: the linear fit supplies the intercept and slope, its residual at the first sample
    supplies the amplitude, and the decay constant starts at a fraction of the span.

    Elapsed samples are measured from the first sample fit rather than from the start of the
    recording. Anchored at the recording start, a segment far into a long session would evaluate
    the exponential deep in its tail, where it underflows to zero and takes the gradient with it,
    and the fit would return the starting guess unchanged.
    """
    elapsed = sample_index - sample_index[0]
    span = float(elapsed[-1])

    linear_fit = _fit_without_bleaching(control, signal, "OLS")
    residual = signal - predict(linear_fit, control)
    initial_guess = [
        linear_fit.intercept,
        linear_fit.slope,
        float(residual[0]),
        span * INITIAL_DECAY_CONSTANT_FRACTION,
    ]

    coefficients, _ = curve_fit(
        _bleaching_model,
        (control, elapsed),
        signal,
        p0=initial_guess,
        bounds=(
            [-np.inf, -np.inf, -np.inf, span * SHORTEST_DECAY_CONSTANT_FRACTION],
            [np.inf, np.inf, np.inf, span * LONGEST_DECAY_CONSTANT_MULTIPLE],
        ),
    )
    intercept, slope, bleaching_amplitude, decay_constant = coefficients
    logger.debug(f"Control fit bleaching term: decay constant {decay_constant:.4g} samples")

    return ControlFitModel(
        slope=float(slope),
        intercept=float(intercept),
        bleaching_amplitude=float(bleaching_amplitude),
        decay_constant=float(decay_constant),
        sample_origin=float(sample_index[0]),
    )


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
        Regression method; ``'IRWLS'`` (default) or ``'OLS'``. Ignored when
        ``photobleaching_detrend`` is True, which is fit by ordinary least squares.
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

    if photobleaching_detrend == True:
        return _fit_with_bleaching(control[indices], signal[indices], indices)
    return _fit_without_bleaching(control[indices], signal[indices], method)


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
