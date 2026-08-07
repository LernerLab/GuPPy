import numpy as np
import pytest

from guppy.analysis.control_fit import (
    ControlFitModel,
    apply_control_fit,
    controlFit,
    deltaFF,
    estimate_control_fit,
    execute_controlFit_dff,
    filterSignal,
    fit_linear_model,
    validate_chunk_lengths_for_filtering,
)

# ── validate_chunk_lengths_for_filtering ──────────────────────────────────────


def test_validate_chunk_lengths_raises_on_short_chunk():
    # filter_window=100 -> padlen=300, so a chunk must have > 300 samples.
    tsNew = np.arange(250.0)  # 250 samples, all inside one chunk
    coords = np.array([[-0.5, 249.5]])
    with pytest.raises(ValueError) as exception_info:
        validate_chunk_lengths_for_filtering(tsNew, coords, 100)
    message = str(exception_info.value)
    assert "250 samples" in message
    assert "more than 300" in message


def test_validate_chunk_lengths_passes_when_all_chunks_long_enough():
    tsNew = np.arange(400.0)  # 400 samples > padlen 300
    coords = np.array([[-0.5, 399.5]])
    # Should not raise.
    validate_chunk_lengths_for_filtering(tsNew, coords, 100)


def test_validate_chunk_lengths_no_check_when_filtering_disabled():
    tsNew = np.arange(10.0)  # far shorter than any padlen
    coords = np.array([[-0.5, 9.5]])
    # filter_window=0 disables filtering, so no length requirement applies.
    validate_chunk_lengths_for_filtering(tsNew, coords, 0)


# ── filterSignal ──────────────────────────────────────────────────────────────


def test_filter_signal_window_zero_returns_original(uniform_signal):
    result = filterSignal(0, uniform_signal)
    np.testing.assert_array_equal(result, uniform_signal)


def test_filter_signal_window_greater_than_one_smooths(uniform_signal):
    result = filterSignal(11, uniform_signal)
    assert result.var() < uniform_signal.var()


def test_filter_signal_window_one_raises():
    signal = np.ones(100)
    with pytest.raises(ValueError) as exception_info:
        filterSignal(1, signal)
    message = str(exception_info.value)
    assert "filter_window=1" in message
    assert "Use 0 to disable" in message


# ── deltaFF ───────────────────────────────────────────────────────────────────


def test_delta_ff_equal_signal_and_control_returns_zeros():
    signal = np.array([1.0, 2.0, 3.0])
    control = np.array([1.0, 2.0, 3.0])
    result = deltaFF(signal, control)
    np.testing.assert_allclose(result, np.zeros(3))


def test_delta_ff_double_signal_returns_one_hundred():
    control = np.array([1.0, 2.0, 3.0])
    signal = 2.0 * control
    result = deltaFF(signal, control)
    np.testing.assert_allclose(result, np.full(3, 100.0))


# ── controlFit ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["IRWLS", "OLS"])
def test_control_fit_output_is_linear_transform_of_control(method):
    rng = np.random.default_rng(seed=0)
    control = rng.standard_normal(500)
    signal = 2.5 * control + 1.0 + 0.01 * rng.standard_normal(500)
    result = controlFit(control, signal, method=method)
    # result should be a linear function of control; residuals should be small
    residuals = signal - result
    assert residuals.std() < 0.1


def test_control_fit_ols_known_signal_returns_exact_fit():
    # signal = 3.0 * control + 0.5; ordinary least squares should recover this exactly
    control = np.array([0.0, 1.0, 2.0])
    signal = np.array([0.5, 3.5, 6.5])
    result = controlFit(control, signal, method="OLS")
    np.testing.assert_allclose(result, np.array([0.5, 3.5, 6.5]), atol=1e-10)


def test_control_fit_unknown_method_raises():
    control = np.array([0.0, 1.0, 2.0])
    signal = np.array([0.5, 3.5, 6.5])
    with pytest.raises(ValueError) as exception_info:
        controlFit(control, signal, method="quadratic")
    message = str(exception_info.value)
    assert "quadratic" in message
    assert "IRWLS" in message


def test_control_fit_irwls_downweights_outliers_while_ols_is_pulled_off():
    # Clean line: signal = 2.0 * control + 1.0, with a handful of large outliers added.
    # IRWLS should recover the clean line (slope ~2, intercept ~1); ordinary least
    # squares is dragged toward the outliers.
    rng = np.random.default_rng(seed=7)
    control = np.linspace(0.0, 99.0, 100)
    signal = 2.0 * control + 1.0 + rng.standard_normal(100) * 0.05
    # Place the outliers in the upper half of the x-range so they lever the
    # least-squares slope (balanced outliers would only shift the intercept).
    outlier_indices = [70, 80, 90, 99]
    signal[outlier_indices] += 200.0

    irwls_fit = controlFit(control, signal, method="IRWLS")
    ols_fit = controlFit(control, signal, method="OLS")

    # Recover slope/intercept from the fitted values (fit = slope * control + intercept).
    irwls_slope = (irwls_fit[-1] - irwls_fit[0]) / (control[-1] - control[0])
    irwls_intercept = irwls_fit[0] - irwls_slope * control[0]
    ols_slope = (ols_fit[-1] - ols_fit[0]) / (control[-1] - control[0])

    # IRWLS stays close to the true clean line.
    np.testing.assert_allclose(irwls_slope, 2.0, atol=0.05)
    np.testing.assert_allclose(irwls_intercept, 1.0, atol=1.0)
    # Ordinary least squares is visibly pulled off the true slope by the outliers,
    # and IRWLS is markedly closer to the truth than OLS.
    assert abs(ols_slope - 2.0) > abs(irwls_slope - 2.0)
    assert abs(ols_slope - 2.0) > 0.2


# ── execute_controlFit_dff ────────────────────────────────────────────────────


def test_execute_control_fit_dff_isosbestic_true_signal_proportional_to_control():
    # signal = 1.5 * control → perfect linear fit → control_fit == signal → norm_data == 0
    control = np.array([1.0, 2.0, 3.0])
    signal = 1.5 * control
    norm_data, control_fit = execute_controlFit_dff(
        control, signal, isosbestic_control=True, filter_window=0, control_fit_method="OLS"
    )
    np.testing.assert_allclose(norm_data, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(control_fit, np.array([1.5, 3.0, 4.5]), atol=1e-10)


def test_execute_control_fit_dff_isosbestic_false_signal_offset_from_control():
    # signal = control + 1.0 → perfect linear fit → control_fit == signal → norm_data == 0
    control = np.array([1.0, 2.0, 3.0])
    signal = control + 1.0
    norm_data, control_fit = execute_controlFit_dff(
        control, signal, isosbestic_control=False, filter_window=0, control_fit_method="OLS"
    )
    np.testing.assert_allclose(norm_data, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(control_fit, np.array([2.0, 3.0, 4.0]), atol=1e-10)


# ── fit_linear_model ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["OLS", "IRWLS"])
def test_fit_linear_model_recovers_two_regressors(method):
    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    second = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    # signal = 2*first + 5*second + 1
    signal = np.array([6.0, 3.0, 10.0, 7.0, 14.0])

    intercept, first_coefficient, second_coefficient = fit_linear_model(
        design_matrix=np.column_stack([first, second]), signal=signal, method=method
    )

    np.testing.assert_allclose([intercept, first_coefficient, second_coefficient], [1.0, 2.0, 5.0], atol=1e-8)


def test_fit_linear_model_unknown_method_raises():
    with pytest.raises(ValueError, match="is not recognized"):
        fit_linear_model(design_matrix=np.arange(5.0), signal=np.arange(5.0), method="quadratic")


# ── the photobleaching term ───────────────────────────────────────────────────

# A control that varies enough for the regression to be well determined, plus bleaching the control
# does not carry: signal = 2*control + 5 + 30*exp(-i/500) over 5000 samples.
BLEACHING_SAMPLE_INDEX = np.arange(5000.0)
BLEACHING_CONTROL = 100.0 + 10.0 * np.sin(2.0 * np.pi * BLEACHING_SAMPLE_INDEX / 700.0)
BLEACHING_SIGNAL = 2.0 * BLEACHING_CONTROL + 5.0 + 30.0 * np.exp(-BLEACHING_SAMPLE_INDEX / 500.0)


def _fit_with_bleaching(control=None, signal=None, method="OLS", sample_index=None):
    control = BLEACHING_CONTROL if control is None else control
    signal = BLEACHING_SIGNAL if signal is None else signal
    sample_index = BLEACHING_SAMPLE_INDEX if sample_index is None else sample_index
    model = estimate_control_fit(control, signal, sample_index=sample_index, method=method, photobleaching_detrend=True)
    return model, apply_control_fit(model, control, sample_index)


@pytest.mark.parametrize("method", ["OLS", "IRWLS"])
def test_bleaching_fit_recovers_the_generating_coefficients(method):
    model, fitted_values = _fit_with_bleaching(method=method)

    np.testing.assert_allclose(model.slope, 2.0, atol=1e-3)
    np.testing.assert_allclose(model.intercept, 5.0, atol=0.1)
    np.testing.assert_allclose(model.bleaching_amplitude, 30.0, atol=0.1)
    np.testing.assert_allclose(model.decay_constant, 500.0, rtol=0.02)
    np.testing.assert_allclose(fitted_values, BLEACHING_SIGNAL, atol=0.05)


def test_bleaching_fit_beats_a_control_only_fit():
    """The point of the joint fit: fitting the control alone lets its slope absorb the decay."""
    _, joint = _fit_with_bleaching()
    control_only = controlFit(BLEACHING_CONTROL, BLEACHING_SIGNAL, method="OLS")

    assert np.abs(joint - BLEACHING_SIGNAL).max() < 0.05
    assert np.abs(control_only - BLEACHING_SIGNAL).max() > 5.0


def test_bleaching_fit_invents_no_trend_when_there_is_none():
    signal_without_bleaching = 2.0 * BLEACHING_CONTROL + 5.0

    model, fitted_values = _fit_with_bleaching(signal=signal_without_bleaching)

    np.testing.assert_allclose(fitted_values, signal_without_bleaching, atol=1e-4)
    assert abs(model.bleaching_amplitude) < 1e-3


def test_bleaching_fit_stays_bounded_on_a_ramp():
    """A decay slower than the segment is barely identifiable, but must not run away.

    The exponential flattens toward a straight line, which it could otherwise reach by growing
    the amplitude and offset without limit in opposite directions.
    """
    ramp = 2.0 * BLEACHING_CONTROL + 5.0 + BLEACHING_SAMPLE_INDEX * 0.01

    model, fitted_values = _fit_with_bleaching(signal=ramp)

    assert np.abs(fitted_values).max() < 10.0 * np.abs(ramp).max()
    assert abs(model.bleaching_amplitude) < 1e4


def test_control_fit_option_matches_the_estimate_and_apply_pair():
    """The parameter is a passthrough on controlFit, not a second implementation."""
    _, direct = _fit_with_bleaching()
    through_option = controlFit(BLEACHING_CONTROL, BLEACHING_SIGNAL, method="OLS", photobleaching_detrend=True)

    np.testing.assert_allclose(through_option, direct, atol=1e-12)


# ── applying a model estimated elsewhere ──────────────────────────────────────


def test_a_model_estimated_from_a_window_extrapolates_on_the_recording_axis():
    """Baseline-epoch fitting estimates on a window and applies beyond it.

    The sample index carries the position of each sample in the recording, so the exponential
    keeps a single origin and the fit continues correctly outside the window it came from.
    """
    window = slice(0, 1500)
    model = estimate_control_fit(
        BLEACHING_CONTROL[window],
        BLEACHING_SIGNAL[window],
        sample_index=BLEACHING_SAMPLE_INDEX[window],
        method="OLS",
        photobleaching_detrend=True,
    )

    fitted_values = apply_control_fit(model, BLEACHING_CONTROL, BLEACHING_SAMPLE_INDEX)

    # Estimated on the first 1500 samples, still tracking the signal at sample 4999.
    np.testing.assert_allclose(fitted_values, BLEACHING_SIGNAL, atol=0.1)


def test_apply_control_fit_without_a_bleaching_term_is_purely_linear():
    control = np.array([1.0, 2.0, 3.0])
    model = ControlFitModel(slope=2.0, intercept=10.0)

    result = apply_control_fit(model, control, np.arange(3.0))

    np.testing.assert_allclose(result, np.array([12.0, 14.0, 16.0]), atol=1e-12)


def test_apply_control_fit_adds_the_bleaching_term_when_present():
    control = np.array([1.0, 2.0, 3.0])
    # 10 + 2*control + 4*exp(-i/2) for i = 0, 1, 2
    model = ControlFitModel(slope=2.0, intercept=10.0, bleaching_amplitude=4.0, decay_constant=2.0)

    result = apply_control_fit(model, control, np.arange(3.0))

    np.testing.assert_allclose(result, np.array([16.0, 16.42612262, 17.47151776]), atol=1e-6)
