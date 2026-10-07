import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.analysis.standard_io import (
    write_covariate_correlations_to_hdf5,
    write_tonic_to_hdf5,
)
from guppy.orchestration.group_analysis import (
    _clear_group_results,
    _filter_stores_list_to_averaged_events,
    _group_event_labels,
    _merge_group_stores_list,
    _recording_sites_with_results,
    _validate_covariate_correlations_consistent_for_group,
    _validate_fiber_recording_sites_consistent_for_group,
    _validate_tonic_epochs_consistent_for_group,
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


class TestClearGroupResults:
    def test_removes_results_but_keeps_the_definition(self, tmp_path):
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / GROUP_MEMBERS_FILENAME).write_text('{"member_run_folders": []}')
        (group_folder / "dropped_member_leftover.h5").touch()
        (group_folder / "storesList.csv").touch()
        nested = group_folder / "cross_correlation_output"
        nested.mkdir()
        (nested / "corr_event_z_score_a_b.h5").touch()

        _clear_group_results(group_folder=str(group_folder))

        assert [entry.name for entry in group_folder.iterdir()] == [GROUP_MEMBERS_FILENAME]

    def test_is_a_no_op_for_a_definition_only_group(self, tmp_path):
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / GROUP_MEMBERS_FILENAME).write_text('{"member_run_folders": []}')

        _clear_group_results(group_folder=str(group_folder))

        assert [entry.name for entry in group_folder.iterdir()] == [GROUP_MEMBERS_FILENAME]


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


@pytest.fixture
def member_run_folders(tmp_path):
    """Two empty member run folders."""
    folders = [tmp_path / "session1" / "session1_output_1", tmp_path / "session2" / "session2_output_1"]
    for folder in folders:
        folder.mkdir(parents=True)
    return [str(folder) for folder in folders]


def write_tonic(run_folder, site, epochs):
    tonic = pd.DataFrame(
        {"mean_zscore": np.zeros(len(epochs)), "mean_dff": np.zeros(len(epochs))},
        index=pd.Index(epochs, name="epoch"),
    )
    write_tonic_to_hdf5(run_folder, tonic, site)


def write_covariate_correlations(run_folder, site, covariates, bin_width):
    correlations = pd.DataFrame(
        {
            "metric": ["mean_zscore"] * len(covariates),
            "covariate": covariates,
            "pearson_r": np.zeros(len(covariates)),
            "spearman_rho": np.zeros(len(covariates)),
            "n_bins": np.full(len(covariates), 10),
        }
    )
    write_covariate_correlations_to_hdf5(filepath=run_folder, correlations=correlations, recording_site=site)
    (Path(run_folder) / "GuPPyParamtersUsed.json").write_text(json.dumps({"binnedMetricsWidth": bin_width}))


class TestRecordingSitesWithResults:
    def test_no_member_holds_the_result(self, member_run_folders):
        assert _recording_sites_with_results(member_run_folders=member_run_folders, prefix="tonic_") == []

    def test_every_member_holds_the_result(self, member_run_folders):
        for run_folder in member_run_folders:
            write_tonic(run_folder, "DMS", ["baseline"])
            write_tonic(run_folder, "NAc", ["baseline"])

        sites = _recording_sites_with_results(member_run_folders=member_run_folders, prefix="tonic_")

        assert sites == ["DMS", "NAc"]

    def test_raises_when_only_some_members_hold_the_result(self, member_run_folders):
        write_tonic(member_run_folders[0], "DMS", ["baseline"])

        with pytest.raises(ValueError, match=r"tonic_DMS.h5 in some member runs but not in:\n  - session2"):
            _recording_sites_with_results(member_run_folders=member_run_folders, prefix="tonic_")

    def test_raises_when_members_hold_different_sites(self, member_run_folders):
        write_tonic(member_run_folders[0], "DMS", ["baseline"])
        write_tonic(member_run_folders[1], "NAc", ["baseline"])

        with pytest.raises(ValueError, match="tonic_DMS.h5 in some member runs"):
            _recording_sites_with_results(member_run_folders=member_run_folders, prefix="tonic_")

    def test_tonic_epoch_definitions_are_not_results(self, member_run_folders):
        (Path(member_run_folders[0]) / "tonic_epochs_DMS.csv").write_text("label,start,end\n")

        assert _recording_sites_with_results(member_run_folders=member_run_folders, prefix="tonic_") == []


class TestValidateTonicEpochsConsistentForGroup:
    def test_passes_for_reordered_epochs(self, member_run_folders):
        write_tonic(member_run_folders[0], "DMS", ["baseline", "drug"])
        write_tonic(member_run_folders[1], "DMS", ["drug", "baseline"])

        _validate_tonic_epochs_consistent_for_group(member_run_folders=member_run_folders, sites=["DMS"])

    def test_raises_for_mismatched_epochs(self, member_run_folders):
        write_tonic(member_run_folders[0], "DMS", ["baseline", "drug"])
        write_tonic(member_run_folders[1], "DMS", ["baseline", "saline"])

        with pytest.raises(ValueError, match="disagree for recording site 'DMS'"):
            _validate_tonic_epochs_consistent_for_group(member_run_folders=member_run_folders, sites=["DMS"])


class TestValidateCovariateCorrelationsConsistentForGroup:
    def test_passes_for_matching_members(self, member_run_folders):
        for run_folder in member_run_folders:
            write_covariate_correlations(run_folder, "DMS", ["akinesia", "grooming"], 50)

        _validate_covariate_correlations_consistent_for_group(member_run_folders=member_run_folders, sites=["DMS"])

    def test_raises_for_mismatched_bin_width(self, member_run_folders):
        write_covariate_correlations(member_run_folders[0], "DMS", ["akinesia"], 50)
        write_covariate_correlations(member_run_folders[1], "DMS", ["akinesia"], 60)

        with pytest.raises(ValueError, match="binnedMetricsWidth, but the members differ"):
            _validate_covariate_correlations_consistent_for_group(member_run_folders=member_run_folders, sites=["DMS"])

    def test_raises_for_mismatched_covariates(self, member_run_folders):
        write_covariate_correlations(member_run_folders[0], "DMS", ["akinesia", "grooming"], 50)
        write_covariate_correlations(member_run_folders[1], "DMS", ["akinesia"], 50)

        with pytest.raises(ValueError, match="covariates akinesia, grooming"):
            _validate_covariate_correlations_consistent_for_group(member_run_folders=member_run_folders, sites=["DMS"])
