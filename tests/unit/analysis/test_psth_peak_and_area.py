import numpy as np
import pytest

from guppy.analysis.psth_peak_and_area import compute_psth_peak_and_area


def _make_psth_mean(n_timepoints=101, n_trials=1):
    """Flat psth_mean array shaped (n_timepoints, n_trials)."""
    return np.zeros((n_timepoints, n_trials))


def test_known_positive_peak_detected_at_correct_timestamp():
    # Build a psth_mean with a clear peak at timestamp 2.0
    timestamps = np.linspace(-5, 5, 101)
    psth_mean = np.zeros((101, 1))
    peak_index = np.argmin(np.abs(timestamps - 2.0))
    psth_mean[peak_index, 0] = 10.0
    result = compute_psth_peak_and_area(
        psth_mean, timestamps, sampling_rate=10.0, peak_startPoint=[0.0], peak_endPoint=[4.0]
    )
    np.testing.assert_allclose(result["peak_pos_1"], np.array([10.0]), atol=1e-6)


def test_triangular_peak_area_matches_trapezoid():
    # Triangle: linearly ramps from 0 at t=0 to 1 at t=0.5, back to 0 at t=1
    n = 101
    timestamps = np.linspace(0, 1, n)
    psth_mean = np.zeros((n, 1))
    mid = n // 2
    psth_mean[:mid, 0] = np.linspace(0, 1, mid)
    psth_mean[mid:, 0] = np.linspace(1, 0, n - mid)
    # compute_psth_peak_and_area calls np.trapezoid without dx or x,
    # so it uses unit spacing (dx=1 sample) rather than the actual time axis
    start_index = np.where(timestamps >= 0.0)[0][0]
    end_index = np.where(timestamps >= 1.0)[0][0]
    expected_area = np.trapezoid(psth_mean[start_index:end_index, 0])
    result = compute_psth_peak_and_area(
        psth_mean, timestamps, sampling_rate=100.0, peak_startPoint=[0.0], peak_endPoint=[1.0]
    )
    np.testing.assert_allclose(result["area_1"], np.array([expected_area]), rtol=1e-5)


def test_end_point_less_than_start_point_raises():
    psth_mean = _make_psth_mean()
    timestamps = np.linspace(-5, 5, 101)
    with pytest.raises(Exception):
        compute_psth_peak_and_area(
            psth_mean, timestamps, sampling_rate=10.0, peak_startPoint=[2.0], peak_endPoint=[0.0]
        )


def test_end_point_equal_to_start_point_raises():
    psth_mean = _make_psth_mean()
    timestamps = np.linspace(-5, 5, 101)
    with pytest.raises(Exception):
        compute_psth_peak_and_area(
            psth_mean, timestamps, sampling_rate=10.0, peak_startPoint=[1.0], peak_endPoint=[1.0]
        )


def test_unequal_start_and_end_point_counts_raises():
    psth_mean = _make_psth_mean()
    timestamps = np.linspace(-5, 5, 101)
    with pytest.raises(Exception):
        compute_psth_peak_and_area(
            psth_mean, timestamps, sampling_rate=10.0, peak_startPoint=[0.0, 1.0], peak_endPoint=[2.0]
        )


def test_all_nan_start_points_returns_nan_peak_and_area():
    psth_mean = _make_psth_mean()
    timestamps = np.linspace(-5, 5, 101)
    result = compute_psth_peak_and_area(
        psth_mean, timestamps, sampling_rate=10.0, peak_startPoint=[np.nan], peak_endPoint=[np.nan]
    )
    assert np.isnan(result["peak"])
    assert np.isnan(result["area"])


def test_multiple_windows_produce_correctly_numbered_keys():
    psth_mean = _make_psth_mean()
    timestamps = np.linspace(-5, 5, 101)
    result = compute_psth_peak_and_area(
        psth_mean,
        timestamps,
        sampling_rate=10.0,
        peak_startPoint=[-4.0, 1.0],
        peak_endPoint=[-1.0, 4.0],
    )
    assert "peak_pos_1" in result
    assert "peak_neg_1" in result
    assert "area_1" in result
    assert "peak_pos_2" in result
    assert "peak_neg_2" in result
    assert "area_2" in result
