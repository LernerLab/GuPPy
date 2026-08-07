import numpy as np
import pytest

from guppy.analysis import control_fit
from guppy.analysis.artifact_removal import retained_chunk_indices
from guppy.analysis.control_fit import ControlFitModel, select_fit_window_indices


def _fit_and_predict(control, signal, method="OLS", photobleaching_detrend=False):
    """Fit on a whole trace and predict over it, the way a per-segment fit does."""
    model = control_fit.fit(control, signal, method=method, photobleaching_detrend=photobleaching_detrend)
    return model, control_fit.predict(model, control)


# ── the linear fit ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["IRWLS", "OLS"])
def test_fit_output_is_linear_transform_of_control(method):
    rng = np.random.default_rng(seed=0)
    control = rng.standard_normal(500)
    signal = 2.5 * control + 1.0 + 0.01 * rng.standard_normal(500)
    _, fitted_values = _fit_and_predict(control, signal, method=method)
    # result should be a linear function of control; residuals should be small
    residuals = signal - fitted_values
    assert residuals.std() < 0.1


def test_fit_ols_known_signal_returns_exact_fit():
    # signal = 3.0 * control + 0.5; ordinary least squares should recover this exactly
    control = np.array([0.0, 1.0, 2.0])
    signal = np.array([0.5, 3.5, 6.5])
    model, fitted_values = _fit_and_predict(control, signal)
    assert model.slope == pytest.approx(3.0)
    assert model.intercept == pytest.approx(0.5)
    assert model.decay_constant is None
    np.testing.assert_allclose(fitted_values, np.array([0.5, 3.5, 6.5]), atol=1e-10)


def test_fit_unknown_method_raises():
    control = np.array([0.0, 1.0, 2.0])
    signal = np.array([0.5, 3.5, 6.5])
    with pytest.raises(ValueError) as exception_info:
        _fit_and_predict(control, signal, method="quadratic")
    message = str(exception_info.value)
    assert "quadratic" in message
    assert "is not recognized" in message
    assert "IRWLS" in message


def test_fit_irwls_downweights_outliers_while_ols_is_pulled_off():
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

    irwls_model, _ = _fit_and_predict(control, signal, method="IRWLS")
    ols_model, _ = _fit_and_predict(control, signal, method="OLS")

    # IRWLS stays close to the true clean line.
    np.testing.assert_allclose(irwls_model.slope, 2.0, atol=0.05)
    np.testing.assert_allclose(irwls_model.intercept, 1.0, atol=1.0)
    # Ordinary least squares is visibly pulled off the true slope by the outliers,
    # and IRWLS is markedly closer to the truth than OLS.
    assert abs(ols_model.slope - 2.0) > abs(irwls_model.slope - 2.0)
    assert abs(ols_model.slope - 2.0) > 0.2


# ── the photobleaching term ───────────────────────────────────────────────────

# A control that varies enough for the regression to be well determined, plus bleaching the control
# does not carry: signal = 2*control + 5 + 30*exp(-i/500) over 5000 samples.
BLEACHING_SAMPLE_INDEX = np.arange(5000.0)
BLEACHING_CONTROL = 100.0 + 10.0 * np.sin(2.0 * np.pi * BLEACHING_SAMPLE_INDEX / 700.0)
BLEACHING_SIGNAL = 2.0 * BLEACHING_CONTROL + 5.0 + 30.0 * np.exp(-BLEACHING_SAMPLE_INDEX / 500.0)


def _fit_with_bleaching(control=None, signal=None, method="OLS"):
    control = BLEACHING_CONTROL if control is None else control
    signal = BLEACHING_SIGNAL if signal is None else signal
    return _fit_and_predict(control, signal, method=method, photobleaching_detrend=True)


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
    _, control_only = _fit_and_predict(BLEACHING_CONTROL, BLEACHING_SIGNAL, photobleaching_detrend=False)

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


# ── applying a model estimated elsewhere ──────────────────────────────────────


def test_a_model_estimated_from_a_window_extrapolates_on_the_recording_axis():
    """Baseline-epoch fitting estimates on a window and applies beyond it.

    The sample index carries the position of each sample in the recording, so the exponential
    keeps a single origin and the fit continues correctly outside the window it came from.
    """
    model = control_fit.fit(
        BLEACHING_CONTROL,
        BLEACHING_SIGNAL,
        indices=np.arange(1500),
        method="OLS",
        photobleaching_detrend=True,
    )

    fitted_values = control_fit.predict(model, BLEACHING_CONTROL)

    # Estimated on the first 1500 samples, still tracking the signal at sample 4999.
    np.testing.assert_allclose(fitted_values, BLEACHING_SIGNAL, atol=0.1)


def test_predict_without_a_bleaching_term_is_purely_linear():
    control = np.array([1.0, 2.0, 3.0])
    model = ControlFitModel(slope=2.0, intercept=10.0)

    result = control_fit.predict(model, control)

    np.testing.assert_allclose(result, np.array([12.0, 14.0, 16.0]), atol=1e-12)


def test_predict_adds_the_bleaching_term_when_present():
    control = np.array([1.0, 2.0, 3.0])
    # 10 + 2*control + 4*exp(-i/2) for i = 0, 1, 2
    model = ControlFitModel(slope=2.0, intercept=10.0, bleaching_amplitude=4.0, decay_constant=2.0)

    result = control_fit.predict(model, control)

    np.testing.assert_allclose(result, np.array([16.0, 16.42612262, 17.47151776]), atol=1e-6)


# ── selecting the baseline-epoch samples ──────────────────────────────────────


def test_select_window_indices_intersects_the_window_with_retained_data():
    ts = np.arange(10.0)
    # Good chunks keep {0,1,2} and {4,...,9}; t=3 falls in the removed gap.
    coords = np.array([[-0.5, 2.5], [3.5, 9.5]])

    indices = select_fit_window_indices(ts, retained_chunk_indices(ts, coords), 0, 5)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 4, 5]))


def test_select_window_indices_raises_when_the_window_holds_no_retained_data():
    ts = np.arange(10.0)
    # Retain only {4,...,9}; a fit window of [0, 3] intersects no retained data.
    coords = np.array([[3.5, 9.5]])
    with pytest.raises(ValueError, match="no data after artifact removal"):
        select_fit_window_indices(ts, retained_chunk_indices(ts, coords), 0, 3)


def test_select_window_indices_raises_on_a_window_outside_the_recording():
    ts = np.arange(10.0)
    coords = np.array([[-0.5, 9.5]])
    with pytest.raises(ValueError, match="controlFitWindowEnd=120 exceeds signal duration"):
        select_fit_window_indices(ts, retained_chunk_indices(ts, coords), 0, 120)
