import numpy as np
import pytest

from guppy.analysis.z_score import (
    controlFit,
    deltaFF,
    execute_controlFit_dff,
    filterSignal,
    z_score_computation,
)


def test_filter_signal_window_zero_returns_original(uniform_signal):
    result = filterSignal(0, uniform_signal)
    np.testing.assert_array_equal(result, uniform_signal)


def test_filter_signal_window_greater_than_one_smooths(uniform_signal):
    result = filterSignal(11, uniform_signal)
    assert result.var() < uniform_signal.var()


def test_filter_signal_window_one_raises():
    signal = np.ones(100)
    with pytest.raises(Exception):
        filterSignal(1, signal)


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


def test_control_fit_output_is_linear_transform_of_control():
    rng = np.random.default_rng(seed=0)
    control = rng.standard_normal(500)
    signal = 2.5 * control + 1.0 + 0.01 * rng.standard_normal(500)
    result = controlFit(control, signal)
    # result should be a linear function of control; residuals should be small
    residuals = signal - result
    assert residuals.std() < 0.1


def test_control_fit_known_linear_signal_returns_exact_fit():
    # signal = 3.0 * control + 0.5; polyfit should recover this exactly
    control = np.array([0.0, 1.0, 2.0])
    signal = np.array([0.5, 3.5, 6.5])
    result = controlFit(control, signal)
    np.testing.assert_allclose(result, np.array([0.5, 3.5, 6.5]), atol=1e-10)


def test_z_score_computation_standard_has_zero_mean_unit_std():
    rng = np.random.default_rng(seed=1)
    dff = rng.standard_normal(1000) * 5.0 + 3.0
    timestamps = np.linspace(0, 10, 1000)
    result = z_score_computation(dff, timestamps, "standard z-score", 0, 0)
    np.testing.assert_allclose(np.nanmean(result), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.nanstd(result), 1.0, atol=1e-10)


def test_z_score_computation_baseline_mean_near_zero_in_window():
    rng = np.random.default_rng(seed=2)
    dff = rng.standard_normal(1000) * 5.0 + 3.0
    timestamps = np.linspace(0, 10, 1000)
    result = z_score_computation(dff, timestamps, "baseline z-score", 0.0, 3.0)
    baseline_indices = np.where((timestamps > 0.0) & (timestamps < 3.0))[0]
    np.testing.assert_allclose(np.nanmean(result[baseline_indices]), 0.0, atol=1e-10)


def test_z_score_computation_baseline_raises_when_window_outside_timestamps():
    dff = np.ones(100)
    timestamps = np.linspace(5.0, 10.0, 100)
    with pytest.raises(Exception):
        z_score_computation(dff, timestamps, "baseline z-score", 0.0, 4.0)


def test_z_score_computation_mad_median_near_zero():
    rng = np.random.default_rng(seed=3)
    dff = rng.standard_normal(1000)
    timestamps = np.linspace(0, 10, 1000)
    result = z_score_computation(dff, timestamps, "mad", 0, 0)
    np.testing.assert_allclose(np.median(result), 0.0, atol=1e-10)


def test_z_score_computation_unknown_method_falls_through_to_mad():
    rng = np.random.default_rng(seed=4)
    dff = rng.standard_normal(500)
    timestamps = np.linspace(0, 5, 500)
    result = z_score_computation(dff, timestamps, "unknown_method", 0, 0)
    # The else-branch uses MAD — median of output should be near zero
    np.testing.assert_allclose(np.median(result), 0.0, atol=1e-10)


def test_execute_control_fit_dff_isosbestic_true_signal_proportional_to_control():
    # signal = 1.5 * control → perfect linear fit → control_fit == signal → norm_data == 0
    control = np.array([1.0, 2.0, 3.0])
    signal = 1.5 * control
    norm_data, control_fit = execute_controlFit_dff(control, signal, isosbestic_control=True, filter_window=0)
    np.testing.assert_allclose(norm_data, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(control_fit, np.array([1.5, 3.0, 4.5]), atol=1e-10)


def test_execute_control_fit_dff_isosbestic_false_signal_offset_from_control():
    # signal = control + 1.0 → perfect linear fit → control_fit == signal → norm_data == 0
    control = np.array([1.0, 2.0, 3.0])
    signal = control + 1.0
    norm_data, control_fit = execute_controlFit_dff(control, signal, isosbestic_control=False, filter_window=0)
    np.testing.assert_allclose(norm_data, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(control_fit, np.array([2.0, 3.0, 4.0]), atol=1e-10)
