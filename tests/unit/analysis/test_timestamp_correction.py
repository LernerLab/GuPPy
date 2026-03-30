import numpy as np

from guppy.analysis.timestamp_correction import (
    applyCorrection_ttl,
    check_cntrl_sig_length,
    correct_timestamps,
    decide_naming_and_applyCorrection_ttl,
    timestampCorrection,
)


def test_apply_correction_ttl_tdt_mode_all_above_rec_start_subtracts_both_offsets():
    # All timestamps >= timeRecStart → subtract both timeRecStart and timeForLightsTurnOn
    # [10.5-10.0-2.0, 11.0-10.0-2.0, 12.0-10.0-2.0] = [-1.5, -1.0, 0.0]
    ttl_timestamps = np.array([10.5, 11.0, 12.0])
    result = applyCorrection_ttl(2.0, 10.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.array([-1.5, -1.0, 0.0]))


def test_apply_correction_ttl_tdt_mode_some_below_rec_start_subtracts_only_lights_turn_on():
    # One timestamp is below timeRecStart → subtract only timeForLightsTurnOn
    # [9.5-2.0, 11.0-2.0, 12.0-2.0] = [7.5, 9.0, 10.0]
    ttl_timestamps = np.array([9.5, 11.0, 12.0])
    result = applyCorrection_ttl(2.0, 10.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.array([7.5, 9.0, 10.0]))


def test_apply_correction_ttl_csv_mode_subtracts_only_lights_turn_on():
    # CSV mode always subtracts only timeForLightsTurnOn regardless of timeRecStart
    # [5.0-3.0, 8.0-3.0, 12.0-3.0] = [2.0, 5.0, 9.0]
    ttl_timestamps = np.array([5.0, 8.0, 12.0])
    result = applyCorrection_ttl(3.0, 0.0, ttl_timestamps, "csv")
    np.testing.assert_allclose(result, np.array([2.0, 5.0, 9.0]))


def test_apply_correction_ttl_tdt_mode_all_at_rec_start_subtracts_both_offsets():
    # All timestamps >= timeRecStart=100 → [100-100-1, ..., 109-100-1] = [-1, 0, ..., 8]
    ttl_timestamps = np.arange(10, dtype=float) + 100.0
    result = applyCorrection_ttl(1.0, 100.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.arange(-1, 9, dtype=float))


def test_check_cntrl_sig_length_control_shorter_returns_control_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    name_to_data = {
        "control_DMS": np.ones(50),
        "signal_DMS": np.ones(100),
    }
    result = check_cntrl_sig_length(channels_arr, name_to_data)
    assert result == ["control_DMS"]


def test_check_cntrl_sig_length_signal_shorter_returns_signal_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    name_to_data = {
        "control_DMS": np.ones(100),
        "signal_DMS": np.ones(50),
    }
    result = check_cntrl_sig_length(channels_arr, name_to_data)
    assert result == ["signal_DMS"]


def test_check_cntrl_sig_length_equal_length_returns_signal_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    name_to_data = {
        "control_DMS": np.ones(100),
        "signal_DMS": np.ones(100),
    }
    result = check_cntrl_sig_length(channels_arr, name_to_data)
    assert result == ["signal_DMS"]


def test_check_cntrl_sig_length_multiple_pairs():
    channels_arr = np.array([["control_DMS", "control_NAc"], ["signal_DMS", "signal_NAc"]])
    name_to_data = {
        "control_DMS": np.ones(80),
        "signal_DMS": np.ones(100),
        "control_NAc": np.ones(100),
        "signal_NAc": np.ones(70),
    }
    result = check_cntrl_sig_length(channels_arr, name_to_data)
    assert result == ["control_DMS", "signal_NAc"]


# ── timestampCorrection ───────────────────────────────────────────────────────


def test_timestamp_correction_csv_mode_slices_at_lights_turn_on():
    # CSV mode: correctionIndex = where timestamp >= timeForLightsTurnOn
    # timestamps = [0, 1, 2, 3, 4, 5]; timeForLightsTurnOn = 2.0
    # → correctionIndex = [2, 3, 4, 5]; timestampNew = [2.0, 3.0, 4.0, 5.0]
    storesList = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    name_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    name_to_data = {"control_dms": data.copy(), "signal_dms": data.copy() * 2}
    name_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    name_to_npoints = {"control_dms": None, "signal_dms": None}

    result_ts, result_idx, result_data = timestampCorrection(
        2.0, storesList, name_to_timestamps, name_to_data, name_to_sampling_rate, name_to_npoints, mode="csv"
    )

    np.testing.assert_array_equal(result_ts["control_dms"], np.array([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(result_data["control_dms"], np.array([30.0, 40.0, 50.0, 60.0]))
    np.testing.assert_array_equal(result_data["signal_dms"], np.array([60.0, 80.0, 100.0, 120.0]))


# ── decide_naming_and_applyCorrection_ttl ─────────────────────────────────────


def test_decide_naming_applies_csv_correction_to_ttl_and_forms_compound_name():
    # CSV mode: corrected = TTL - timeForLightsTurnOn
    # compound_name = "TTL1_dms"; [3-1, 5-1, 7-1] = [2.0, 4.0, 6.0]
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    name_to_timestamps_ttl = {"TTL1": np.array([3.0, 5.0, 7.0])}
    name_to_timestamps = {
        "control_dms": np.array([1.0, 2.0, 3.0, 4.0]),
        "signal_dms": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    name_to_data = {"control_dms": np.ones(4), "signal_dms": np.ones(4)}

    result = decide_naming_and_applyCorrection_ttl(
        1.0, storesList, name_to_timestamps_ttl, name_to_timestamps, name_to_data, mode="csv"
    )

    assert "TTL1_dms" in result
    np.testing.assert_array_equal(result["TTL1_dms"], np.array([2.0, 4.0, 6.0]))


# ── correct_timestamps ────────────────────────────────────────────────────────


def test_correct_timestamps_returns_all_four_outputs_consistent():
    storesList = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    data = np.arange(5, dtype=float)
    name_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    name_to_data = {"control_dms": data.copy(), "signal_dms": data.copy()}
    name_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    name_to_npoints = {"control_dms": None, "signal_dms": None}
    name_to_timestamps_ttl = {"TTL1": np.array([2.5, 3.5])}

    result_ts, result_idx, result_data, result_ttl = correct_timestamps(
        1.0,
        storesList,
        name_to_timestamps,
        name_to_data,
        name_to_sampling_rate,
        name_to_npoints,
        name_to_timestamps_ttl,
        mode="csv",
    )

    # CSV mode: correctionIndex = where timestamp >= 1.0 → indices [1, 2, 3, 4]
    assert "control_dms" in result_ts
    assert result_ts["control_dms"].shape[0] == 4
    assert "TTL1_dms" in result_ttl
    # CSV TTL: [2.5-1.0, 3.5-1.0] = [1.5, 2.5]
    np.testing.assert_array_equal(result_ttl["TTL1_dms"], np.array([1.5, 2.5]))
