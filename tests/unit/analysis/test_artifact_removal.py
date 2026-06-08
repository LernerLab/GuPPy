import numpy as np
import pytest

from guppy.analysis.artifact_removal import (
    addingNaNtoChunksWithArtifacts,
    addingNaNValues,
    eliminateData,
    eliminateTs,
    processTimestampsForArtifacts,
    remove_artifacts,
    removeTTLs,
)


def test_remove_artifacts_raises_for_invalid_method():
    with pytest.raises(ValueError, match=r"Invalid artifact removal method 'bogus'"):
        remove_artifacts(
            timeForLightsTurnOn=0.0,
            storesList=np.array([["c"], ["control_dms"]]),
            pair_name_to_tsNew={},
            pair_name_to_sampling_rate={},
            pair_name_to_coords={},
            name_to_data={},
            compound_name_to_ttl_timestamps={},
            method="bogus",
        )


def test_remove_artifacts_invalid_method_message_lists_allowed_methods():
    with pytest.raises(ValueError) as exception_info:
        remove_artifacts(
            timeForLightsTurnOn=0.0,
            storesList=np.array([["c"], ["control_dms"]]),
            pair_name_to_tsNew={},
            pair_name_to_sampling_rate={},
            pair_name_to_coords={},
            name_to_data={},
            compound_name_to_ttl_timestamps={},
            method="bogus",
        )
    message = str(exception_info.value)
    assert "concatenate" in message
    assert "replace with NaN" in message


def test_eliminate_data_single_window_output_length():
    # Window [1.0, 4.0] spans indices where ts is between 1.0 and 4.0 exclusively
    ts = np.linspace(0, 5, 501)  # 0.01 spacing
    data = np.arange(501, dtype=float)
    coords = np.array([[1.0, 4.0]])
    result_data, result_ts = eliminateData(data=data, ts=ts, coords=coords)
    expected_count = np.sum((ts > 1.0) & (ts < 4.0))
    assert result_data.shape[0] == expected_count
    assert result_ts.shape[0] == expected_count


def test_eliminate_data_single_window_preserves_original_recording_time():
    ts = np.linspace(0, 5, 501)
    data = np.ones(501)
    coords = np.array([[1.0, 4.0]])
    _, result_ts = eliminateData(data=data, ts=ts, coords=coords)
    # Timestamps are preserved (not re-stamped): the kept samples keep their original times,
    # so the result equals the in-window subset of ts verbatim.
    expected_ts = ts[(ts > 1.0) & (ts < 4.0)]
    np.testing.assert_allclose(result_ts, expected_ts, atol=1e-12)


def test_eliminate_data_two_windows_preserve_gapped_original_timestamps():
    ts = np.linspace(0, 10, 1001)
    data = np.arange(1001, dtype=float)
    coords = np.array([[1.0, 4.0], [6.0, 9.0]])
    result_data, result_ts = eliminateData(data=data, ts=ts, coords=coords)
    count_window_1 = np.sum((ts > 1.0) & (ts < 4.0))
    count_window_2 = np.sum((ts > 6.0) & (ts < 9.0))
    assert result_data.shape[0] == count_window_1 + count_window_2
    # The two windows are concatenated but each keeps its original time, leaving a gap from
    # ~3.99 to ~6.01 (samples 4.0-6.0 are dropped). The result is the gapped original series.
    expected_ts = np.concatenate((ts[(ts > 1.0) & (ts < 4.0)], ts[(ts > 6.0) & (ts < 9.0)]))
    np.testing.assert_allclose(result_ts, expected_ts, atol=1e-12)
    # Discontinuity at the window boundary is preserved (spacing >> the 0.01 sample spacing).
    boundary_gap = result_ts[count_window_1] - result_ts[count_window_1 - 1]
    assert boundary_gap > 2.0


def test_eliminate_data_all_zeros_returns_zero_array():
    ts = np.linspace(0, 5, 501)
    data = np.zeros(501)
    coords = np.array([[1.0, 4.0]])
    result_data, _ = eliminateData(data=data, ts=ts, coords=coords)
    assert (result_data == 0).all()


def test_eliminate_ts_all_ttls_inside_window_are_preserved():
    # Timestamps are preserved (not re-stamped): in-window TTLs are returned verbatim.
    ttl_ts = np.array([1.5, 2.0, 3.5])
    coords = np.array([[0.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, coords=coords)
    np.testing.assert_allclose(result, np.array([1.5, 2.0, 3.5]), atol=1e-12)


def test_eliminate_ts_ttls_outside_window_are_dropped():
    # 0.0 equals the window start so fails the strict >, 4.5 exceeds the window end.
    # The kept TTLs retain their original times (no re-stamping).
    ttl_ts = np.array([0.0, 1.5, 2.0, 4.5])
    coords = np.array([[0.0, 4.0]])
    result = eliminateTs(ts=ttl_ts, coords=coords)
    np.testing.assert_allclose(result, np.array([1.5, 2.0]), atol=1e-12)


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
    # Timestamps are preserved in original recording time (not re-stamped): the in-window
    # samples are ts=2.0 and ts=3.0.
    np.testing.assert_allclose(result_ts["dms"], np.array([2.0, 3.0]), atol=1e-12)
    # TTLs: 1.5 and 2.5 inside (preserved verbatim); 4.5 outside
    np.testing.assert_allclose(result_ttl["TTL1_dms"], np.array([1.5, 2.5]), atol=1e-12)


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
