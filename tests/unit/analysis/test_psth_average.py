import numpy as np

from guppy.analysis.psth_average import psth_shape_check


def test_psth_shape_check_all_same_length_returns_unchanged():
    arrays = [np.ones(10), np.ones(10) * 2, np.ones(10) * 3]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(10))
    np.testing.assert_array_equal(result[1], np.ones(10) * 2)
    np.testing.assert_array_equal(result[2], np.ones(10) * 3)


def test_psth_shape_check_shorter_arrays_padded_with_nan():
    # First array (length 8) padded to match last element (length 10);
    # original 8 values are preserved; padding slots are NaN
    arrays = [np.ones(8), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0][:8], np.ones(8))
    assert np.all(np.isnan(result[0][8:]))
    np.testing.assert_array_equal(result[1], np.ones(10))


def test_psth_shape_check_longer_arrays_truncated():
    # First array (length 15) truncated to match last element (length 10)
    arrays = [np.ones(15), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(10))
    np.testing.assert_array_equal(result[1], np.ones(10))


def test_psth_shape_check_uses_last_element_length_not_maximum():
    # Lengths [10, 8, 12] → canonical length is 12 (the last element, not the max)
    # First array padded by 2, second padded by 4, third unchanged
    arrays = [np.ones(10), np.ones(8), np.ones(12)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0][:10], np.ones(10))
    assert np.all(np.isnan(result[0][10:]))
    np.testing.assert_array_equal(result[1][:8], np.ones(8))
    assert np.all(np.isnan(result[1][8:]))
    np.testing.assert_array_equal(result[2], np.ones(12))


def test_psth_shape_check_last_element_shorter_truncates_longer_earlier_arrays():
    # Lengths [12, 10, 8] → canonical length is 8 (the last element)
    arrays = [np.ones(12), np.ones(10), np.ones(8)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.ones(8))
    np.testing.assert_array_equal(result[1], np.ones(8))
    np.testing.assert_array_equal(result[2], np.ones(8))


def test_psth_shape_check_single_element_returned_unchanged():
    arrays = [np.array([1.0, 2.0, 3.0])]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0]))


def test_psth_shape_check_preserves_values_of_unchanged_arrays():
    data = np.arange(10, dtype=float)
    arrays = [data.copy(), np.ones(10)]
    result = psth_shape_check(arrays)
    np.testing.assert_array_equal(result[0], data)
