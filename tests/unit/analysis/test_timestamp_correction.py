import numpy as np
import pytest

from guppy.analysis.timestamp_correction import (
    applyCorrection_ttl,
    correct_timestamps,
    decide_naming_and_applyCorrection_ttl,
    pair_timeline_label,
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


class TestPairTimelineLabel:
    @pytest.fixture
    def control_signal_names(self):
        return np.array([["control_DMS", "control_NAc"], ["signal_DMS", "signal_NAc"]])

    def test_signal_names_the_signal(self, control_signal_names):
        assert pair_timeline_label(control_signal_names, pair_index=1, pair_timestamps_channel="signal") == "signal_NAc"

    def test_control_names_the_control(self, control_signal_names):
        assert (
            pair_timeline_label(control_signal_names, pair_index=0, pair_timestamps_channel="control") == "control_DMS"
        )

    def test_an_unrecognized_channel_raises(self, control_signal_names):
        with pytest.raises(KeyError):
            pair_timeline_label(control_signal_names, pair_index=0, pair_timestamps_channel="reference")


# ── timestampCorrection ───────────────────────────────────────────────────────


def test_timestamp_correction_csv_mode_slices_at_lights_turn_on():
    # CSV mode: the cut is measured from the recording's own start.
    # timestamps = [0, 1, 2, 3, 4, 5] start at 0, so with timeForLightsTurnOn = 2.0 the cut
    # is at 0 + 2.0 → correctionIndex = [2, 3, 4, 5]; timestampNew = [2.0, 3.0, 4.0, 5.0]
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy() * 2}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}

    result_ts, result_idx, result_data, _ = timestampCorrection(
        2.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        mode="csv",
        pair_timestamps_channel="signal",
    )

    np.testing.assert_array_equal(result_ts["control_dms"], np.array([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(result_data["control_dms"], np.array([30.0, 40.0, 50.0, 60.0]))
    np.testing.assert_array_equal(result_data["signal_dms"], np.array([60.0, 80.0, 100.0, 120.0]))


def test_timestamp_correction_csv_mode_measures_lights_turn_on_from_recording_start():
    # A recording whose clock does not start at 0 (e.g. an NWB file with starting_time=100.0).
    # timestamps = [100..105]; timeForLightsTurnOn = 2.0 → cut at 100.0 + 2.0 = 102.0
    # → correctionIndex = [2, 3, 4, 5]; timestampNew = [102.0, 103.0, 104.0, 105.0].
    # Comparing against timeForLightsTurnOn alone would keep every sample.
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])
    timestamps = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy() * 2}
    store_label_to_sampling_rate = {"control_dms": np.array([1.0]), "signal_dms": np.array([1.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}

    result_ts, result_idx, result_data, _ = timestampCorrection(
        2.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        mode="csv",
        pair_timestamps_channel="signal",
    )

    np.testing.assert_array_equal(result_ts["control_dms"], np.array([102.0, 103.0, 104.0, 105.0]))
    np.testing.assert_array_equal(result_idx["control_dms"], np.array([2, 3, 4, 5]))
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
    result = decide_naming_and_applyCorrection_ttl(
        1.0,
        store_array,
        store_label_to_timestamps_ttl,
        store_label_to_timestamps,
        mode="csv",
        pair_timestamps_channel="signal",
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
            pair_timestamps_channel="signal",
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
            pair_timestamps_channel="signal",
        )


def test_decide_naming_surfaces_mismatched_recording_site_pairs_via_storeslist():
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_vms", "TTL1"]])
    store_label_to_timestamps_ttl = {"TTL1": np.array([1.0, 2.0])}
    store_label_to_timestamps = {"control_dms": np.zeros(3), "signal_vms": np.zeros(3)}
    with pytest.raises(ValueError, match="Mismatched signal/control recording-site pairs"):
        decide_naming_and_applyCorrection_ttl(
            0.0,
            store_array,
            store_label_to_timestamps_ttl,
            store_label_to_timestamps,
            mode="csv",
            pair_timestamps_channel="signal",
        )


def test_correct_timestamps_returns_all_five_outputs_consistent():
    store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    data = np.arange(5, dtype=float)
    store_label_to_timestamps = {"control_dms": timestamps.copy(), "signal_dms": timestamps.copy()}
    store_label_to_data = {"control_dms": data.copy(), "signal_dms": data.copy()}
    store_label_to_sampling_rate = {"control_dms": np.array([100.0]), "signal_dms": np.array([100.0])}
    store_label_to_npoints = {"control_dms": None, "signal_dms": None}
    store_label_to_timestamps_ttl = {"TTL1": np.array([2.5, 3.5])}

    result_ts, result_idx, result_data, result_timeline_labels, result_ttl = correct_timestamps(
        1.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        store_label_to_timestamps_ttl,
        mode="csv",
        pair_timestamps_channel="signal",
    )

    # CSV mode: correctionIndex = where timestamp >= 1.0 → indices [1, 2, 3, 4]
    assert "control_dms" in result_ts
    assert result_ts["control_dms"].shape[0] == 4
    assert result_timeline_labels == {"control_dms": "signal_dms", "signal_dms": "signal_dms"}
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

    corrected_ts, _, _, _, corrected_ttl = correct_timestamps(
        1.0,
        store_array,
        store_label_to_timestamps,
        store_label_to_data,
        store_label_to_sampling_rate,
        store_label_to_npoints,
        store_label_to_timestamps_ttl,
        mode="csv",
        pair_timestamps_channel="signal",
    )

    timestampNew = corrected_ts["signal_dms"]
    event = corrected_ttl["TTL1_dms"][0]
    # Event is on the recording-start basis (NOT re-zeroed to lights-on at 0.0).
    np.testing.assert_allclose(event, 2.4)
    # It falls within the continuous timespan and lands on the matching continuous sample.
    assert timestampNew[0] <= event <= timestampNew[-1]
    index = int(round((event - timestampNew[0]) * 10.0))
    np.testing.assert_allclose(timestampNew[index], event)


class TestPairTimestampsChannel:
    """Each pair is timed by the chosen channel, and both channels are cut to the shorter one."""

    @pytest.fixture
    def store_array(self):
        return np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])

    @pytest.fixture
    def offset_channels(self):
        """A control sampled on whole seconds and a signal sampled half a second later."""
        return {
            "timestamps": {
                "control_dms": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                "signal_dms": np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            },
            "data": {"control_dms": np.arange(6, dtype=float), "signal_dms": np.arange(6, dtype=float) + 10},
            "sampling_rate": {"control_dms": np.array([1.0]), "signal_dms": np.array([1.0])},
            "npoints": {"control_dms": None, "signal_dms": None},
        }

    @pytest.mark.parametrize(
        "pair_timestamps_channel, expected_timestamps",
        [("signal", [2.5, 3.5, 4.5, 5.5]), ("control", [2.0, 3.0, 4.0, 5.0])],
    )
    def test_the_pair_is_timed_by_the_chosen_channel(
        self, store_array, offset_channels, pair_timestamps_channel, expected_timestamps
    ):
        result_ts, result_idx, _, result_timeline_labels = timestampCorrection(
            2.0,
            store_array,
            offset_channels["timestamps"],
            offset_channels["data"],
            offset_channels["sampling_rate"],
            offset_channels["npoints"],
            mode="csv",
            pair_timestamps_channel=pair_timestamps_channel,
        )

        # The warm-up cut is measured from the chosen channel's own start, so both keep rows 2-5.
        np.testing.assert_array_equal(result_ts["control_dms"], expected_timestamps)
        np.testing.assert_array_equal(result_ts["signal_dms"], expected_timestamps)
        np.testing.assert_array_equal(result_idx["signal_dms"], [2, 3, 4, 5])
        chosen_label = f"{pair_timestamps_channel}_dms"
        assert result_timeline_labels == {"control_dms": chosen_label, "signal_dms": chosen_label}

    def test_a_longer_chosen_channel_is_cut_to_the_shorter(self, store_array, offset_channels):
        offset_channels["data"]["control_dms"] = np.arange(4, dtype=float)

        result_ts, _, result_data, _ = timestampCorrection(
            0.0,
            store_array,
            offset_channels["timestamps"],
            offset_channels["data"],
            offset_channels["sampling_rate"],
            offset_channels["npoints"],
            mode="csv",
            pair_timestamps_channel="signal",
        )

        np.testing.assert_array_equal(result_ts["signal_dms"], [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result_data["signal_dms"], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_array_equal(result_data["control_dms"], [0.0, 1.0, 2.0, 3.0])

    def test_a_shorter_chosen_channel_cuts_the_other(self, store_array, offset_channels):
        offset_channels["data"]["control_dms"] = np.arange(4, dtype=float)
        offset_channels["timestamps"]["control_dms"] = np.array([0.0, 1.0, 2.0, 3.0])

        result_ts, _, result_data, _ = timestampCorrection(
            0.0,
            store_array,
            offset_channels["timestamps"],
            offset_channels["data"],
            offset_channels["sampling_rate"],
            offset_channels["npoints"],
            mode="csv",
            pair_timestamps_channel="control",
        )

        np.testing.assert_array_equal(result_ts["signal_dms"], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result_data["signal_dms"], [10.0, 11.0, 12.0, 13.0])

    @pytest.mark.parametrize("pair_timestamps_channel, expected_ttl", [("signal", [1.5]), ("control", [2.0])])
    def test_tdt_events_are_rebased_on_the_chosen_channel(self, pair_timestamps_channel, expected_ttl):
        store_array = np.array([["ctrl0", "sig0", "ttl0"], ["control_dms", "signal_dms", "TTL1"]])
        store_label_to_timestamps = {"control_dms": np.array([10.0, 11.0]), "signal_dms": np.array([10.5, 11.5])}

        result = decide_naming_and_applyCorrection_ttl(
            0.0,
            store_array,
            {"TTL1": np.array([12.0])},
            store_label_to_timestamps,
            mode="tdt",
            pair_timestamps_channel=pair_timestamps_channel,
        )

        np.testing.assert_allclose(result["TTL1_dms"], expected_ttl)
