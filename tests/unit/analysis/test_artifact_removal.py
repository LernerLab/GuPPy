import numpy as np

from guppy.analysis.artifact_removal import (
    addingNaNtoChunksWithArtifacts,
    addingNaNValues,
    eliminateData,
    eliminateTs,
    processTimestampsForArtifacts,
    remove_artifacts,
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


def test_eliminate_data_all_zeros_returns_zero_array():
    ts = np.linspace(0, 5, 501)
    data = np.zeros(501)
    coords = np.array([[1.0, 4.0]])
    result_data, _ = eliminateData(data=data, ts=ts, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    assert (result_data == 0).all()


def test_eliminate_ts_all_ttls_inside_window_are_preserved():
    # tsNew integers: first value strictly inside (0.0, 4.0) is 1.0, so shift = 1.0 - 0.0 = 1.0
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ttl_ts = np.array([1.5, 2.0, 3.5])
    coords = np.array([[0.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, tsNew=tsNew, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    np.testing.assert_allclose(result, np.array([0.5, 1.0, 2.5]), atol=1e-6)


def test_eliminate_ts_ttls_outside_window_are_dropped():
    # tsNew integers: first value strictly inside (0.0, 4.0) is 1.0, so shift = 1.0 - 0.0 = 1.0
    # 0.0 equals the window start so fails the strict >, 4.5 exceeds the window end
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ttl_ts = np.array([0.0, 1.5, 2.0, 4.5])
    coords = np.array([[0.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, tsNew=tsNew, coords=coords, timeForLightsTurnOn=0.0, sampling_rate=100.0)
    np.testing.assert_allclose(result, np.array([0.5, 1.0]), atol=1e-6)


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
    # 0.5, 3.5, 4.5 outside [1.0, 3.0]; 1.5 and 2.5 inside
    np.testing.assert_array_equal(result, np.array([1.5, 2.5]))


def test_remove_ttls_empty_result_when_no_timestamps_in_windows():
    ts = np.array([0.1, 0.2, 4.5, 4.8])
    coords = np.array([[1.0, 4.0]])
    result = removeTTLs(ts=ts, coords=coords)
    assert result.shape[0] == 0


def test_remove_ttls_two_windows_returns_timestamps_from_both():
    ts = np.array([0.5, 1.5, 2.5, 5.5, 6.5, 9.5])
    coords = np.array([[1.0, 4.0], [5.0, 8.0]])
    result = removeTTLs(ts=ts, coords=coords)
    # 1.5, 2.5 inside [1,4]; 5.5, 6.5 inside [5,8]; 0.5 and 9.5 are outside both windows
    expected = np.array([1.5, 2.5, 5.5, 6.5])
    np.testing.assert_array_equal(result, expected)


# ── addingNaNtoChunksWithArtifacts ────────────────────────────────────────────


def test_adding_nan_to_chunks_sets_nan_outside_coords_for_signal_and_control():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coords = np.array([[1.0, 4.0]])
    name_to_data = {"control_dms": np.ones(6), "signal_dms": np.ones(6) * 2.0}
    compound_name_to_ttl_timestamps = {"TTL1_dms": np.array([1.5, 2.5, 4.5])}

    result_data, _ = addingNaNtoChunksWithArtifacts(
        storesList, {"dms": tsNew}, {"dms": coords}, name_to_data, compound_name_to_ttl_timestamps
    )

    # ts=[0,1,2,3,4,5]: strictly inside (1,4) → indices 2,3; outside → 0,1,4,5 must be NaN
    assert np.isnan(result_data["control_dms"][0])
    assert np.isnan(result_data["control_dms"][1])
    assert result_data["control_dms"][2] == 1.0
    assert result_data["control_dms"][3] == 1.0
    assert np.isnan(result_data["control_dms"][4])


def test_adding_nan_to_chunks_drops_ttls_outside_coords():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coords = np.array([[1.0, 4.0]])
    name_to_data = {"control_dms": np.ones(6), "signal_dms": np.ones(6)}
    compound_name_to_ttl_timestamps = {"TTL1_dms": np.array([1.5, 2.5, 4.5])}

    _, result_ttl = addingNaNtoChunksWithArtifacts(
        storesList, {"dms": tsNew}, {"dms": coords}, name_to_data, compound_name_to_ttl_timestamps
    )

    # 4.5 is outside (1, 4); only 1.5 and 2.5 are kept
    np.testing.assert_array_equal(result_ttl["TTL1_dms"], np.array([1.5, 2.5]))


# ── processTimestampsForArtifacts ─────────────────────────────────────────────


def test_process_timestamps_for_artifacts_concatenates_data_inside_coords():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coords = np.array([[1.0, 4.0]])
    # ts strictly inside (1, 4): indices 2 and 3 (ts=2.0, ts=3.0)
    name_to_data = {
        "control_dms": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        "signal_dms": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    compound_name_to_ttl_timestamps = {"TTL1_dms": np.array([1.5, 2.5, 4.5])}

    result_data, result_ts, result_ttl = processTimestampsForArtifacts(
        0.5, storesList, {"dms": tsNew}, {"dms": 100.0}, {"dms": coords}, name_to_data, compound_name_to_ttl_timestamps
    )

    # Only 2 samples inside the window
    assert result_data["control_dms"].shape[0] == 2
    np.testing.assert_array_equal(result_data["control_dms"], np.array([30.0, 40.0]))
    # First corrected timestamp = timeForLightsTurnOn = 0.5
    np.testing.assert_allclose(result_ts["dms"][0], 0.5, atol=1e-6)
    # TTLs: 1.5 and 2.5 inside; 4.5 outside
    assert result_ttl["TTL1_dms"].shape[0] == 2


# ── remove_artifacts ──────────────────────────────────────────────────────────


def test_remove_artifacts_concatenate_method_returns_correct_data():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coords = np.array([[1.0, 4.0]])
    name_to_data = {"control_dms": np.ones(6), "signal_dms": np.ones(6) * 2.0}
    compound_name_to_ttl_timestamps = {"TTL1_dms": np.array([1.5, 2.5, 4.5])}

    result_data, result_ts, result_ttl = remove_artifacts(
        0.5,
        storesList,
        {"dms": tsNew},
        {"dms": 100.0},
        {"dms": coords},
        name_to_data,
        compound_name_to_ttl_timestamps,
        method="concatenate",
    )

    assert result_data["control_dms"].shape[0] == 2
    assert result_ts is not None
    assert result_ttl["TTL1_dms"].shape[0] == 2


def test_remove_artifacts_replace_with_nan_method_returns_nan_outside_coords():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    tsNew = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coords = np.array([[1.0, 4.0]])
    name_to_data = {"control_dms": np.ones(6), "signal_dms": np.ones(6)}
    compound_name_to_ttl_timestamps = {"TTL1_dms": np.array([1.5, 2.5, 4.5])}

    result_data, result_ts, result_ttl = remove_artifacts(
        0.0,
        storesList,
        {"dms": tsNew},
        {"dms": 100.0},
        {"dms": coords},
        name_to_data,
        compound_name_to_ttl_timestamps,
        method="replace with NaN",
    )

    # NaN method: pair_name_to_corrected_timestamps is None
    assert result_ts is None
    assert np.isnan(result_data["control_dms"][0])
    assert result_data["control_dms"][2] == 1.0
