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


def test_output_lag_axis_length_is_two_n_minus_one():
    n = 150
    a = np.ones(n)
    b = np.ones(n)
    result = compute_cross_correlation([a], [b], sample_rate=100.0)
    # Last row is the lag axis; its length should be 2*n - 1
    assert result.shape[1] == 2 * n - 1


def test_single_pair_produces_two_rows(uniform_signal):
    # One correlation pair + one lag row = 2 rows total
    result = compute_cross_correlation([uniform_signal], [uniform_signal], sample_rate=100.0)
    assert result.shape[0] == 2


def test_multiple_pairs_produce_correct_number_of_rows():
    rng = np.random.default_rng(seed=12)
    signals = [rng.standard_normal(100) for _ in range(3)]
    result = compute_cross_correlation(signals, signals, sample_rate=50.0)
    # 3 correlation rows + 1 lag row = 4 rows
    assert result.shape[0] == 4


def test_nan_containing_arrays_do_not_crash():
    a = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
    b = np.array([2.0, 1.0, np.nan, 1.0, 2.0])
    # Should not raise — uses "direct" method path
    result = compute_cross_correlation([a], [b], sample_rate=100.0)
    assert result.shape[1] == 2 * len(a) - 1
