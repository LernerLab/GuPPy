import numpy as np
import pytest

from guppy.analysis.z_score import compute_z_score, z_score_computation

# ── z_score_computation ───────────────────────────────────────────────────────


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


def _bleaching_pair(n=2000):
    """A control/signal pair carrying bleaching the control cannot account for.

    The control oscillates so the control-to-signal regression is well determined; the signal
    adds an exponential decay on top of that linear relationship, which no rescaling of the
    control can remove and which therefore shows up as drift in the dF/F.
    """
    tsNew = np.linspace(0.0, 100.0, n)
    control = 2.0 + 0.5 * np.sin(tsNew)
    signal = 3.0 * control + 1.0 + 4.0 * np.exp(-np.arange(n) / 200.0)
    return tsNew, control, signal


def _run_compute_z_score(tsNew, control, signal, photobleaching_detrend):
    return compute_z_score(
        control=control,
        signal=signal,
        tsNew=tsNew,
        coords=np.array([[-1.0, 101.0]]),
        artifactsRemovalMethod="replace with NaN",
        filter_window=0,
        isosbestic_control=True,
        zscore_method="standard z-score",
        baseline_start=0.0,
        baseline_end=0.0,
        control_fit_method="OLS",
        photobleaching_detrend=photobleaching_detrend,
    )


def _drift(dff, tsNew):
    """Start-to-end change in the dF/F, over the first and last second of the trace."""
    return float(np.mean(dff[tsNew <= 1.0]) - np.mean(dff[tsNew >= 99.0]))


def test_compute_z_score_detrending_flattens_the_dff_drift():
    tsNew, control, signal = _bleaching_pair()

    _, dff_without_detrend, _, _ = _run_compute_z_score(tsNew, control, signal, False)
    _, dff_with_detrend, _, _ = _run_compute_z_score(tsNew, control, signal, True)

    # The decay dominates the trace when the fit cannot represent it, and is gone once it can.
    assert _drift(dff_without_detrend, tsNew) > 20.0
    assert abs(_drift(dff_with_detrend, tsNew)) < 0.5


def test_compute_z_score_detrending_folds_the_decay_into_the_fitted_control():
    """The bleaching is part of the fit, not something removed from the dF/F afterwards."""
    tsNew, control, signal = _bleaching_pair()

    _, _, fit_without_detrend, _ = _run_compute_z_score(tsNew, control, signal, False)
    _, _, fit_with_detrend, _ = _run_compute_z_score(tsNew, control, signal, True)

    # Only the joint fit tracks the signal, because only it can represent the decay.
    assert np.abs(fit_with_detrend - signal).max() < 0.05
    assert np.abs(fit_without_detrend - signal).max() > 1.0


def test_compute_z_score_detrending_leaves_a_bleaching_free_pair_alone():
    """A signal that is already a linear function of its control has nothing to detrend."""
    tsNew = np.linspace(0.0, 100.0, 2000)
    control = 2.0 + 0.5 * np.sin(tsNew)
    signal = 3.0 * control + 1.0

    _, dff_without_detrend, _, _ = _run_compute_z_score(tsNew, control, signal, False)
    _, dff_with_detrend, _, _ = _run_compute_z_score(tsNew, control, signal, True)

    np.testing.assert_allclose(dff_with_detrend, dff_without_detrend, atol=1e-3)
