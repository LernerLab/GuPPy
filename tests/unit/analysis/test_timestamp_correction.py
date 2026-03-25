import numpy as np

from guppy.analysis.timestamp_correction import (
    applyCorrection_ttl,
    check_cntrl_sig_length,
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
