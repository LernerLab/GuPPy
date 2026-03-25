import numpy as np

from guppy.analysis.cross_correlation import compute_cross_correlation


def test_identical_arrays_max_correlation_at_lag_zero():
    signal = np.sin(np.linspace(0, 4 * np.pi, 200))
    sample_rate = 20.0
    result = compute_cross_correlation([signal], [signal], sample_rate)
    # Last row is the lag axis (in units of 1/sample_rate)
    lag_row = result[-1, :]
    cross_corr_row = result[0, :]
    zero_lag_index = np.argmin(np.abs(lag_row))
    assert np.argmax(cross_corr_row) == zero_lag_index


def test_all_values_normalized_within_minus_one_to_one():
    rng = np.random.default_rng(seed=11)
    a = rng.standard_normal(300)
    b = rng.standard_normal(300)
    result = compute_cross_correlation([a], [b], sample_rate=100.0)
    # Exclude last row (lag axis)
    corr_rows = result[:-1, :]
    assert np.all(corr_rows >= -1.0 - 1e-6)
    assert np.all(corr_rows <= 1.0 + 1e-6)


def test_output_lag_axis_has_correct_values():
    # n=5, sample_rate=10.0 → lags [-4,-3,-2,-1,0,1,2,3,4] / 10.0
    n = 5
    a = np.ones(n)
    b = np.ones(n)
    result = compute_cross_correlation([a], [b], sample_rate=10.0)
    expected_lags = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4], dtype="float32")
    np.testing.assert_allclose(result[-1], expected_lags, atol=1e-5)


def test_single_pair_auto_correlation_peaks_at_one(uniform_signal):
    # For identical inputs, normalized correlation at zero lag equals 1.0
    result = compute_cross_correlation([uniform_signal], [uniform_signal], sample_rate=100.0)
    zero_lag_index = len(uniform_signal) - 1
    np.testing.assert_allclose(result[0, zero_lag_index], 1.0, atol=1e-5)


def test_multiple_pairs_lag_axis_center_is_zero():
    rng = np.random.default_rng(seed=12)
    n = 100
    signals = [rng.standard_normal(n) for _ in range(3)]
    result = compute_cross_correlation(signals, signals, sample_rate=50.0)
    # Center of lag axis (zero-lag position at index n-1) should be 0.0
    np.testing.assert_allclose(result[-1, n - 1], 0.0, atol=1e-6)


def test_nan_containing_arrays_lag_center_is_zero():
    a = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
    b = np.array([2.0, 1.0, np.nan, 1.0, 2.0])
    result = compute_cross_correlation([a], [b], sample_rate=100.0)
    # Zero-lag position is at index len(a)-1 = 4; lag value should be 0.0
    np.testing.assert_allclose(result[-1, 4], 0.0, atol=1e-6)
