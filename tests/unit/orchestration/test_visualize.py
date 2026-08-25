"""
Unit tests for guppy.orchestration.visualize._validate_metric_against_step4_outputs.
"""

import re

import pytest

from guppy.orchestration.visualize import _validate_metric_against_step4_outputs


@pytest.fixture
def make_session():
    """Return a factory building a session directory that contains one output run folder."""

    def _make(tmp_path, name="session1"):
        session_dir = tmp_path / name
        session_dir.mkdir(parents=True, exist_ok=True)
        run_folder = session_dir / f"{name}_output_1"
        run_folder.mkdir(parents=True, exist_ok=True)
        # select_run_folders validates that picked outputs have a storesList.csv.
        (run_folder / "storesList.csv").write_text("")
        return session_dir, run_folder

    return _make


@pytest.fixture
def make_parameters():
    """Return a factory for the minimal inputParameters the metric validator reads."""

    def _make(session_dir, *, visualize_zscore_or_dff="z_score", selected_runs=("1",), selected_group_folders=()):
        return {
            "session_folders": [str(session_dir)],
            "combine_data": False,
            "visualize_zscore_or_dff": visualize_zscore_or_dff,
            "selected_runs": {str(session_dir): list(selected_runs)},
            "selected_group_folders": list(selected_group_folders),
        }

    return _make


class TestZScoreMetric:
    def test_passes_when_z_score_psth_files_present(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")

        _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))

    def test_raises_when_z_score_psth_files_missing(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match="z_score"):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))

    def test_raises_names_missing_output_directory_in_message(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match=re.escape(str(run_folder))):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))

    def test_raises_suggests_alternative_metric(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match="dff"):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))

    def test_raises_suggests_rerun_step4(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match="Re-run step 4"):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))


class TestDffMetric:
    def test_passes_when_dff_psth_files_present(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="dff"))

    def test_raises_when_dff_psth_files_missing(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match="dff"):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="dff"))


class TestBothMetricsComputed:
    def test_passes_for_z_score(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))

    def test_passes_for_dff(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="dff"))


class TestStep3FilesAreNotPsthOutputs:
    def test_step3_hdf5_does_not_satisfy_the_check(self, tmp_path, make_session, make_parameters):
        """z_score_region.hdf5 (step-3 output) must not be mistaken for a PSTH file."""
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "z_score_region.hdf5").write_bytes(b"")

        with pytest.raises(ValueError, match="z_score"):
            _validate_metric_against_step4_outputs(make_parameters(session_dir, visualize_zscore_or_dff="z_score"))


class TestPartialFailureAcrossSessions:
    def test_only_missing_output_directories_are_reported(self, tmp_path, make_session):
        session1_dir, run_folder_1 = make_session(tmp_path, "session1")
        session2_dir, run_folder_2 = make_session(tmp_path, "session2")
        # session1 has z_score PSTH; session2 does not.
        (run_folder_1 / "ttl_region_z_score_region.h5").write_bytes(b"")
        (run_folder_2 / "ttl_region_dff_region.h5").write_bytes(b"")

        parameters = {
            "session_folders": [str(session1_dir), str(session2_dir)],
            "combine_data": False,
            "visualize_zscore_or_dff": "z_score",
            "selected_runs": {str(session1_dir): ["1"], str(session2_dir): ["1"]},
            "selected_group_folders": [],
        }

        with pytest.raises(ValueError) as exception_info:
            _validate_metric_against_step4_outputs(parameters)

        message = str(exception_info.value)
        assert str(run_folder_2) in message
        assert str(run_folder_1) not in message
        assert "1 output director" in message


class TestNoOutputDirectories:
    def test_no_op_when_no_output_directories(self, tmp_path, make_parameters):
        """When no *_output_* dirs exist the function returns silently.

        The homepage gate `validate_selected_runs_for_consumers` skips sessions with
        no output dirs on disk, so they reach orchestration with no entry in
        selected_runs. The validator must skip such sessions instead of erroring.
        """
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()

        _validate_metric_against_step4_outputs(
            make_parameters(session_dir, visualize_zscore_or_dff="z_score", selected_runs=())
        )


class TestGroupFolders:
    def test_group_folder_with_the_metric_passes(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / "ttl_region_z_score_region.h5").write_bytes(b"")

        _validate_metric_against_step4_outputs(make_parameters(session_dir, selected_group_folders=[str(group_folder)]))

    def test_group_folder_missing_the_metric_is_reported(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match=re.escape(str(group_folder))):
            _validate_metric_against_step4_outputs(
                make_parameters(session_dir, selected_group_folders=[str(group_folder)])
            )
