import numpy as np
import pytest

from guppy.analysis.z_score import (
    compute_z_score,
    controlFit,
    deltaFF,
    execute_controlFit_dff,
    filterSignal,
    validate_chunk_lengths_for_filtering,
    z_score_computation,
)


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
    # both start (0.0) and end (4.0) are before the signal starts (5.0),
    # so start < ts_min is flagged first
    with pytest.raises(ValueError, match="baselineWindowStart=0.0 is before the signal start"):
        z_score_computation(dff, timestamps, "baseline z-score", 0.0, 4.0)


def test_z_score_computation_baseline_raises_when_end_exceeds_signal_duration():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 90.5, 100)
    with pytest.raises(ValueError, match="baselineWindowEnd=120 exceeds signal duration"):
        z_score_computation(dff, timestamps, "baseline z-score", 0.0, 120)


def test_z_score_computation_baseline_raises_non_numeric_start():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowStart=.* is not a valid number"):
        z_score_computation(dff, timestamps, "baseline z-score", "bad", 5.0)


def test_z_score_computation_baseline_raises_non_numeric_end():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowEnd=.* is not a valid number"):
        z_score_computation(dff, timestamps, "baseline z-score", 0.0, None)


def test_z_score_computation_baseline_raises_when_start_equals_end():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowStart=5 must be strictly less than baselineWindowEnd=5"):
        z_score_computation(dff, timestamps, "baseline z-score", 5, 5)


def test_z_score_computation_baseline_raises_when_start_greater_than_end():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowStart=7 must be strictly less than baselineWindowEnd=3"):
        z_score_computation(dff, timestamps, "baseline z-score", 7, 3)


def test_z_score_computation_baseline_raises_start_before_signal():
    dff = np.ones(100)
    timestamps = np.linspace(5.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowStart=-1 is before the signal start"):
        z_score_computation(dff, timestamps, "baseline z-score", -1, 8.0)


def test_z_score_computation_baseline_error_message_includes_timespan():
    dff = np.ones(100)
    timestamps = np.linspace(5.0, 10.0, 100)
    with pytest.raises(ValueError, match=r"signal timespan is \[5, 10\]s"):
        z_score_computation(dff, timestamps, "baseline z-score", 0.0, 4.0)


def test_z_score_computation_baseline_raises_nan_start():
    dff = np.ones(100)
    timestamps = np.linspace(0.0, 10.0, 100)
    with pytest.raises(ValueError, match="baselineWindowStart=.* is not a valid number"):
        z_score_computation(dff, timestamps, "baseline z-score", float("nan"), 5.0)


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


# ── compute_z_score ───────────────────────────────────────────────────────────


def test_compute_z_score_isosbestic_returns_standard_normalized_array():
    # signal = 2 * control → perfect fit → norm_data = 0 everywhere → standard z-score = NaN (0/0)
    # Use slightly noisy signal to get a real z-score
    rng = np.random.default_rng(seed=30)
    n = 200
    tsNew = np.linspace(1.0, 11.0, n)
    # coords cover strictly inside the tsNew range
    coords = np.array([[0.5, 11.5]])
    control = 2.0 + rng.standard_normal(n) * 0.1
    signal = 1.5 * control + rng.standard_normal(n) * 0.05

    z_score_arr, norm_data_arr, control_fit_arr, temp_control_arr = compute_z_score(
        control=control,
        signal=signal,
        tsNew=tsNew,
        coords=coords,
        artifactsRemovalMethod="replace with NaN",
        filter_window=0,
        isosbestic_control=True,
        zscore_method="standard z-score",
        baseline_start=0.0,
        baseline_end=0.0,
    )

    # With isosbestic_control=True, temp_control_arr is None
    assert temp_control_arr is None
    # z_score has same length as tsNew
    assert z_score_arr.shape[0] == n
    # Standard z-score has ~zero mean and ~unit std
    np.testing.assert_allclose(np.nanmean(z_score_arr), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.nanstd(z_score_arr), 1.0, atol=1e-6)
    # norm_data and control_fit are filled inside the coords window
    inside = (tsNew > 0.5) & (tsNew < 11.5)
    assert not np.any(np.isnan(norm_data_arr[inside]))
    assert not np.any(np.isnan(control_fit_arr[inside]))
