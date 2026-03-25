import numpy as np

from guppy.analysis.timestamp_correction import (
    applyCorrection_ttl,
    check_cntrl_sig_length,
)


def test_apply_correction_ttl_tdt_mode_all_above_rec_start_subtracts_both_offsets():
    time_rec_start = 10.0
    time_for_lights_turn_on = 2.0
    ttl_timestamps = np.array([10.5, 11.0, 12.0])
    result = applyCorrection_ttl(time_for_lights_turn_on, time_rec_start, ttl_timestamps, "tdt")
    expected = ttl_timestamps - time_rec_start - time_for_lights_turn_on
    np.testing.assert_allclose(result, expected)


def test_apply_correction_ttl_tdt_mode_some_below_rec_start_subtracts_only_lights_turn_on():
    time_rec_start = 10.0
    time_for_lights_turn_on = 2.0
    # One timestamp is below timeRecStart
    ttl_timestamps = np.array([9.5, 11.0, 12.0])
    result = applyCorrection_ttl(time_for_lights_turn_on, time_rec_start, ttl_timestamps, "tdt")
    expected = ttl_timestamps - time_for_lights_turn_on
    np.testing.assert_allclose(result, expected)


def test_apply_correction_ttl_csv_mode_subtracts_only_lights_turn_on():
    time_rec_start = 0.0  # not used in csv mode
    time_for_lights_turn_on = 3.0
    ttl_timestamps = np.array([5.0, 8.0, 12.0])
    result = applyCorrection_ttl(time_for_lights_turn_on, time_rec_start, ttl_timestamps, "csv")
    expected = ttl_timestamps - time_for_lights_turn_on
    np.testing.assert_allclose(result, expected)


def test_apply_correction_ttl_output_same_shape_as_input():
    ttl_timestamps = np.arange(10, dtype=float) + 100.0
    result = applyCorrection_ttl(1.0, 100.0, ttl_timestamps, "tdt")
    assert result.shape == ttl_timestamps.shape


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
