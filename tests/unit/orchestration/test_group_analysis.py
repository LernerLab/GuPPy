import numpy as np
import pytest

from guppy.orchestration.group_analysis import (
    _create_group_folder,
    _filter_stores_list_to_averaged_events,
    _group_event_labels,
    _merge_group_stores_list,
    _validate_fiber_recording_sites_consistent_for_group,
)
from guppy.utils.utils import GROUP_MEMBERS_FILENAME


@pytest.fixture
def write_stores_list():
    """Return a helper writing a minimal storesList.csv into a run folder."""

    def _write(run_folder, store_ids):
        run_folder.mkdir(parents=True, exist_ok=True)
        raw_labels = [f"raw{i}" for i in range(len(store_ids))]
        (run_folder / "storesList.csv").write_text(",".join(raw_labels) + "\n" + ",".join(store_ids) + "\n")
        return str(run_folder)

    return _write


class TestValidateFiberRecordingSitesConsistentForGroup:
    def test_passes_when_all_match(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(
            tmp_path / "session1" / "session1_output_1", ["control_DMS", "signal_DMS", "port_entries"]
        )
        member_2 = write_stores_list(
            tmp_path / "session2" / "session2_output_1", ["control_DMS", "signal_DMS", "port_entries"]
        )

        _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

    def test_allows_reordered_stores(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(
            tmp_path / "session1" / "session1_output_1", ["control_DMS", "signal_DMS", "port_entries"]
        )
        member_2 = write_stores_list(
            tmp_path / "session2" / "session2_output_1", ["signal_DMS", "port_entries", "control_DMS"]
        )

        _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

    def test_allows_same_recording_site_with_different_events(self, tmp_path, write_stores_list):
        # Issue #368: runs from the same fiber recording site (DMS) but under different
        # behavioral conditions must be allowed to average together.
        member_1 = write_stores_list(
            tmp_path / "session1" / "session1_output_1", ["control_DMS", "signal_DMS", "novelobject"]
        )
        member_2 = write_stores_list(
            tmp_path / "session2" / "session2_output_1", ["control_DMS", "signal_DMS", "novelfemale1"]
        )

        _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

    def test_raises_for_non_overlapping_fibers(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(
            tmp_path / "session1" / "session1_output_1", ["control_DMS_A", "signal_DMS_A", "port_entries_A"]
        )
        member_2 = write_stores_list(
            tmp_path / "session2" / "session2_output_1", ["control_DMS_B", "signal_DMS_B", "port_entries_B"]
        )

        with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
            _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

    def test_raises_for_mismatched_recording_site_labels(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(
            tmp_path / "session1" / "session1_output_1", ["control_region1", "signal_region1", "port_entries1"]
        )
        member_2 = write_stores_list(
            tmp_path / "session2" / "session2_output_1", ["control_region2", "signal_region2", "port_entries2"]
        )

        with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
            _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

    def test_error_message_lists_session_name_and_store_ids(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(tmp_path / "session1" / "session1_output_1", ["control_region1", "signal_region1"])
        member_2 = write_stores_list(tmp_path / "session2" / "session2_output_1", ["control_region2", "signal_region2"])

        with pytest.raises(ValueError) as exception_info:
            _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member_1, member_2])

        message = str(exception_info.value)
        for expected in [
            "session1",
            "session2",
            "control_region1",
            "signal_region1",
            "control_region2",
            "signal_region2",
        ]:
            assert expected in message

    def test_single_member_does_not_raise(self, tmp_path, write_stores_list):
        member = write_stores_list(tmp_path / "session1" / "session1_output_1", ["control_DMS", "signal_DMS"])

        _validate_fiber_recording_sites_consistent_for_group(member_run_folders=[member])


class TestMergeGroupStoresList:
    def test_unions_and_deduplicates_member_stores(self, tmp_path, write_stores_list):
        member_1 = write_stores_list(tmp_path / "s1" / "s1_output_1", ["control_dms", "signal_dms", "rewarded"])
        member_2 = write_stores_list(tmp_path / "s2" / "s2_output_1", ["control_dms", "signal_dms", "unrewarded"])

        store_array = _merge_group_stores_list(member_run_folders=[member_1, member_2])

        assert sorted(store_array[1, :].tolist()) == [
            "control_dms",
            "rewarded",
            "signal_dms",
            "unrewarded",
        ]


class TestGroupEventLabels:
    def test_drops_continuous_streams(self):
        store_array = np.array(
            [["raw0", "raw1", "raw2"], ["control_dms", "signal_dms", "rewarded"]],
        )
        parameters = {"useTransientsAsEvents": False, "selectForTransientsComputation": "z_score"}

        assert _group_event_labels(store_array=store_array, inputParameters=parameters) == ["rewarded"]


class TestCreateGroupFolder:
    def test_creates_the_directory(self, tmp_path):
        group_folder = tmp_path / "saline_group"

        _create_group_folder(group_folder=str(group_folder))

        assert group_folder.is_dir()

    def test_rebuilds_an_existing_group_from_scratch(self, tmp_path):
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / GROUP_MEMBERS_FILENAME).write_text('{"member_run_folders": []}')
        (group_folder / "dropped_member_leftover.h5").touch()

        _create_group_folder(group_folder=str(group_folder))

        assert group_folder.is_dir()
        assert list(group_folder.iterdir()) == []

    def test_refuses_a_directory_it_did_not_create(self, tmp_path):
        existing = tmp_path / "important_group"
        existing.mkdir()
        (existing / "someones_data.csv").touch()

        with pytest.raises(ValueError, match="was not created by the Group Analysis step"):
            _create_group_folder(group_folder=str(existing))

        assert (existing / "someones_data.csv").exists()


class TestFilterStoresListToAveragedEvents:
    def test_keeps_continuous_streams_and_averaged_events_only(self):
        store_array = np.array(
            [
                ["raw0", "raw1", "raw2", "raw3"],
                ["control_dms", "signal_dms", "rewarded", "unrewarded"],
            ]
        )

        result = _filter_stores_list_to_averaged_events(store_array=store_array, averaged_events=["rewarded"])

        np.testing.assert_array_equal(
            result, np.array([["raw0", "raw1", "raw2"], ["control_dms", "signal_dms", "rewarded"]])
        )

    def test_drops_every_event_when_none_were_averaged(self):
        store_array = np.array([["raw0", "raw1"], ["signal_dms", "rewarded"]])

        result = _filter_stores_list_to_averaged_events(store_array=store_array, averaged_events=[])

        np.testing.assert_array_equal(result, np.array([["raw0"], ["signal_dms"]]))
