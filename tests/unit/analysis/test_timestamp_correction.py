import numpy as np
import pytest

from guppy.analysis.timestamp_correction import (
    applyCorrection_ttl,
    check_cntrl_sig_length,
    correct_timestamps,
    decide_naming_and_applyCorrection_ttl,
    timestampCorrection,
)


def test_apply_correction_ttl_tdt_mode_all_above_rec_start_subtracts_only_rec_start():
    # All timestamps >= timeRecStart → subtract only timeRecStart (recording-start basis;
    # timeForLightsTurnOn is no longer subtracted from events)
    # [10.5-10.0, 11.0-10.0, 12.0-10.0] = [0.5, 1.0, 2.0]
    ttl_timestamps = np.array([10.5, 11.0, 12.0])
    result = applyCorrection_ttl(2.0, 10.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.array([0.5, 1.0, 2.0]))


def test_apply_correction_ttl_tdt_mode_some_below_rec_start_leaves_unchanged():
    # One timestamp is below timeRecStart (not on the recording clock) → leave unchanged
    ttl_timestamps = np.array([9.5, 11.0, 12.0])
    result = applyCorrection_ttl(2.0, 10.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.array([9.5, 11.0, 12.0]))


def test_apply_correction_ttl_csv_mode_leaves_timestamps_unchanged():
    # CSV timestamps are already recording-relative → returned unchanged (no lights-on shift)
    ttl_timestamps = np.array([5.0, 8.0, 12.0])
    result = applyCorrection_ttl(3.0, 0.0, ttl_timestamps, "csv")
    np.testing.assert_allclose(result, np.array([5.0, 8.0, 12.0]))


def test_apply_correction_ttl_tdt_mode_all_at_rec_start_subtracts_only_rec_start():
    # All timestamps >= timeRecStart=100 → [100-100, ..., 109-100] = [0, 1, ..., 9]
    ttl_timestamps = np.arange(10, dtype=float) + 100.0
    result = applyCorrection_ttl(1.0, 100.0, ttl_timestamps, "tdt")
    np.testing.assert_allclose(result, np.arange(0, 10, dtype=float))


def test_check_cntrl_sig_length_control_shorter_returns_control_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    store_label_to_data = {
        "control_DMS": np.ones(50),
        "signal_DMS": np.ones(100),
    }
    result = check_cntrl_sig_length(channels_arr, store_label_to_data)
    assert result == ["control_DMS"]


def test_check_cntrl_sig_length_signal_shorter_returns_signal_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    store_label_to_data = {
        "control_DMS": np.ones(100),
        "signal_DMS": np.ones(50),
    }
    result = check_cntrl_sig_length(channels_arr, store_label_to_data)
    assert result == ["signal_DMS"]


def test_check_cntrl_sig_length_equal_length_returns_signal_name():
    channels_arr = np.array([["control_DMS"], ["signal_DMS"]])
    store_label_to_data = {
        "control_DMS": np.ones(100),
        "signal_DMS": np.ones(100),
    }
    result = check_cntrl_sig_length(channels_arr, store_label_to_data)
    assert result == ["signal_DMS"]


def test_check_cntrl_sig_length_multiple_pairs():
    channels_arr = np.array([["control_DMS", "control_NAc"], ["signal_DMS", "signal_NAc"]])
    store_label_to_data = {
        "control_DMS": np.ones(80),
        "signal_DMS": np.ones(100),
        "control_NAc": np.ones(100),
        "signal_NAc": np.ones(70),
    }
    result = check_cntrl_sig_length(channels_arr, store_label_to_data)
    assert result == ["control_DMS", "signal_NAc"]


# ── timestampCorrection ───────────────────────────────────────────────────────


def test_timestamp_correction_csv_mode_slices_at_lights_turn_on():
    # CSV mode: correctionIndex = where timestamp >= timeForLightsTurnOn
    # timestamps = [0, 1, 2, 3, 4, 5]; timeForLightsTurnOn = 2.0
    # → correctionIndex = [2, 3, 4, 5]; timestampNew = [2.0, 3.0, 4.0, 5.0]
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy() * 2}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}

    result_ts, result_idx, result_data = timestampCorrection(
        2.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        mode="csv",
    )

    np.testing.assert_array_equal(result_ts["control_dms"], np.array([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(result_data["control_dms"], np.array([30.0, 40.0, 50.0, 60.0]))
    np.testing.assert_array_equal(result_data["signal_dms"], np.array([60.0, 80.0, 100.0, 120.0]))


# ── decide_naming_and_applyCorrection_ttl ─────────────────────────────────────


def test_decide_naming_applies_csv_correction_to_ttl_and_forms_compound_name():
    # CSV mode: events stay on the recording-start basis (unchanged); only the
    # compound name "TTL1_dms" is formed.
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    store_label_to_timestamps_ttl = {"TTL1": np.array([3.0, 5.0, 7.0])}
    store_label_to_timestamps = {
        "control_dms": np.array([1.0, 2.0, 3.0, 4.0]),
        "signal_dms": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    store_label_to_data = {"control_dms": np.ones(4), "signal_dms": np.ones(4)}

    result = decide_naming_and_applyCorrection_ttl(
        1.0, store_array, store_label_to_timestamps_ttl, store_label_to_timestamps, store_label_to_data, mode="csv"
    )

    assert "TTL1_dms" in result
    np.testing.assert_array_equal(result["TTL1_dms"], np.array([3.0, 5.0, 7.0]))


# ── correct_timestamps ────────────────────────────────────────────────────────


# ── timestampCorrection error paths ────────────────────────────────────────────


def test_timestamp_correction_raises_for_invalid_mode():
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    store_label_to_timestamps = {"control_dms": np.zeros(3), "signal_dms": np.zeros(3)}
    store_label_to_data = {"control_dms": np.zeros(3), "signal_dms": np.zeros(3)}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}
    with pytest.raises(ValueError, match=r"Mode 'invalid' is not supported"):
        timestampCorrection(
            0.0,
            store_array,
            store_label_to_timestamps,
            store_label_to_data,
            store_label_to_sampling_rate,
            store_label_to_npoints,
            mode="invalid",
        )


def test_timestamp_correction_surfaces_mismatched_recording_site_pairs_via_storeslist():
    """control_dms paired with signal_vms — surfaces from get_control_and_signal_channel_names."""
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_vms"]])
    store_label_to_timestamps = {"control_dms": np.zeros(3), "signal_vms": np.zeros(3)}
    store_label_to_data = {"control_dms": np.zeros(3), "signal_vms": np.zeros(3)}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_vms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_vms": None}
    with pytest.raises(ValueError, match="Mismatched signal/control recording-site pairs"):
        timestampCorrection(
            0.0,
            store_array,
            store_label_to_timestamps,
            store_label_to_data,
            store_label_to_sampling_rate,
            store_label_to_npoints,
            mode="csv",
        )


def test_decide_naming_surfaces_mismatched_recording_site_pairs_via_storeslist():
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_vms", "TTL1"]])
    store_label_to_timestamps_ttl = {"TTL1": np.array([1.0, 2.0])}
    store_label_to_timestamps = {"control_dms": np.zeros(3), "signal_vms": np.zeros(3)}
    store_label_to_data = {"control_dms": np.zeros(3), "signal_vms": np.zeros(3)}
    with pytest.raises(ValueError, match="Mismatched signal/control recording-site pairs"):
        decide_naming_and_applyCorrection_ttl(
            0.0,
            store_array,
            store_label_to_timestamps_ttl,
            store_label_to_timestamps,
            store_label_to_data,
            mode="csv",
        )


def test_correct_timestamps_returns_all_four_outputs_consistent():
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    data = np.arange(5, dtype=float)
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy()}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}
    store_label_to_timestamps_ttl = {"TTL1": np.array([2.5, 3.5])}

    result_ts, result_idx, result_data, result_ttl = correct_timestamps(
        1.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        store_label_to_timestamps_ttl,
        mode="csv",
    )

    # CSV mode: correctionIndex = where timestamp >= 1.0 → indices [1, 2, 3, 4]
    assert "control_dms" in result_ts
    assert result_ts["control_dms"].shape[0] == 4
    assert "TTL1_dms" in result_ttl
    # CSV TTL stays on the recording-start basis (unchanged): [2.5, 3.5]
    np.testing.assert_array_equal(result_ttl["TTL1_dms"], np.array([2.5, 3.5]))


def test_events_and_continuous_share_one_recording_start_basis():
    # Issue #355: continuous timestampNew and event ts must share a single time basis.
    # The continuous stream is sliced at >= timeForLightsTurnOn (recording basis) and events
    # are NOT re-zeroed, so an event at recording time T equals the continuous timestamp at
    # its matching sample index round((T - timestampNew[0]) * sampling_rate).
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    timestamps = np.arange(0.0, 5.0, 0.1)
    data = np.arange(timestamps.shape[0], dtype=float)
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy()}
    store_label_to_sampling_rate = {"control_dms": np.array([10.0]), "signal_dms": np.array([10.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}
    store_label_to_timestamps_ttl = {"TTL1": np.array([2.4])}

    corrected_ts, _, _, corrected_ttl = correct_timestamps(
        1.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        store_label_to_timestamps_ttl,
        mode="csv",
    )

    timestampNew = corrected_ts["signal_dms"]
    event = corrected_ttl["TTL1_dms"][0]
    # Event is on the recording-start basis (NOT re-zeroed to lights-on at 0.0).
    np.testing.assert_allclose(event, 2.4)
    # It falls within the continuous timespan and lands on the matching continuous sample.
    assert timestampNew[0] <= event <= timestampNew[-1]
    index = int(round((event - timestampNew[0]) * 10.0))
    np.testing.assert_allclose(timestampNew[index], event)
