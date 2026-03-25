import numpy as np

from guppy.analysis.artifact_removal import (
    addingNaNValues,
    eliminateData,
    eliminateTs,
    removeTTLs,
)


def test_eliminate_data_single_window_output_length():
    # Window [1.0, 4.0] spans indices where ts is between 1.0 and 4.0 exclusively
    ts = np.linspace(0, 5, 501)  # 0.01 spacing
    data = np.arange(501, dtype=float)
    coords = np.array([[1.0, 4.0]])
    result_data, result_ts = eliminateData(
        data=data, ts=ts, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0
    )
    expected_count = np.sum((ts > 1.0) & (ts < 4.0))
    assert result_data.shape[0] == expected_count
    assert result_ts.shape[0] == expected_count


def test_eliminate_data_single_window_timestamps_start_at_light_turn_on():
    ts = np.linspace(0, 5, 501)
    data = np.ones(501)
    coords = np.array([[1.0, 4.0]])
    time_for_lights_turn_on = 0.5
    _, result_ts = eliminateData(
        data=data, ts=ts, coords=coords, timeForLightsTurnOn=time_for_lights_turn_on, sampling_rate=100.0
    )
    np.testing.assert_allclose(result_ts[0], time_for_lights_turn_on, atol=1e-6)


def test_eliminate_data_two_windows_output_length_is_sum_of_windows():
    ts = np.linspace(0, 10, 1001)
    data = np.arange(1001, dtype=float)
    coords = np.array([[1.0, 4.0], [6.0, 9.0]])
    result_data, result_ts = eliminateData(
        data=data, ts=ts, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0
    )
    count_window_1 = np.sum((ts > 1.0) & (ts < 4.0))
    count_window_2 = np.sum((ts > 6.0) & (ts < 9.0))
    assert result_data.shape[0] == count_window_1 + count_window_2
    assert result_ts.shape[0] == count_window_1 + count_window_2


def test_eliminate_data_two_windows_timestamps_are_continuous():
    ts = np.linspace(0, 10, 1001)
    data = np.ones(1001)
    coords = np.array([[1.0, 4.0], [6.0, 9.0]])
    _, result_ts = eliminateData(data=data, ts=ts, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    # timestamps should be monotonically increasing with no large gaps
    diffs = np.diff(result_ts)
    assert (diffs > 0).all()


def test_eliminate_data_all_zeros_returns_zero_array():
    ts = np.linspace(0, 5, 501)
    data = np.zeros(501)
    coords = np.array([[1.0, 4.0]])
    result_data, _ = eliminateData(data=data, ts=ts, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    assert (result_data == 0).all()


def test_eliminate_ts_all_ttls_inside_window_are_preserved():
    tsNew = np.linspace(0, 5, 501)
    ttl_ts = np.array([1.5, 2.0, 3.5])
    coords = np.array([[1.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, tsNew=tsNew, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    assert result.shape[0] == 3


def test_eliminate_ts_ttls_outside_window_are_dropped():
    tsNew = np.linspace(0, 5, 501)
    ttl_ts = np.array([0.5, 1.5, 2.0, 4.5])
    coords = np.array([[1.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, tsNew=tsNew, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    # 0.5 and 4.5 are outside [1.0, 4.0]; 1.5 and 2.0 are inside
    assert result.shape[0] == 2


def test_adding_nan_values_out_of_window_indices_are_nan():
    ts = np.linspace(0, 5, 501)
    data = np.ones(501)
    coords = np.array([[1.0, 4.0]])
    result = addingNaNValues(data=data.copy(), ts=ts, coords=coords)
    outside_indices = np.where((ts <= 1.0) | (ts >= 4.0))[0]
    assert np.all(np.isnan(result[outside_indices]))


def test_adding_nan_values_in_window_values_unchanged():
    ts = np.linspace(0, 5, 501)
    data = np.arange(501, dtype=float)
    coords = np.array([[1.0, 4.0]])
    result = addingNaNValues(data=data.copy(), ts=ts, coords=coords)
    inside_indices = np.where((ts > 1.0) & (ts < 4.0))[0]
    np.testing.assert_array_equal(result[inside_indices], data[inside_indices])


def test_adding_nan_values_touching_windows_no_double_counting():
    # Touching (not overlapping) windows — strict inequalities mean boundary points fall in neither window
    ts = np.linspace(0, 10, 1001)
    data = np.ones(1001)
    coords = np.array([[1.0, 5.0], [5.0, 9.0]])
    result = addingNaNValues(data=data.copy(), ts=ts, coords=coords)
    # Points strictly inside either window should not be NaN
    inside = np.where(((ts > 1.0) & (ts < 5.0)) | ((ts > 5.0) & (ts < 9.0)))[0]
    assert not np.any(np.isnan(result[inside]))


def test_remove_ttls_returns_only_timestamps_inside_windows():
    ts = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    coords = np.array([[1.0, 3.0]])
    result = removeTTLs(ts=ts, coords=coords)
    expected = ts[(ts > 1.0) & (ts < 3.0)]
    np.testing.assert_array_equal(result, expected)


def test_remove_ttls_empty_result_when_no_timestamps_in_windows():
    ts = np.array([0.1, 0.2, 4.5, 4.8])
    coords = np.array([[1.0, 4.0]])
    result = removeTTLs(ts=ts, coords=coords)
    assert result.shape[0] == 0


def test_remove_ttls_two_windows_returns_timestamps_from_both():
    ts = np.array([0.5, 1.5, 2.5, 5.5, 6.5, 9.5])
    coords = np.array([[1.0, 4.0], [5.0, 8.0]])
    result = removeTTLs(ts=ts, coords=coords)
    # 1.5, 2.5 inside [1,4]; 5.5, 6.5 inside [5,8]
    assert result.shape[0] == 4
