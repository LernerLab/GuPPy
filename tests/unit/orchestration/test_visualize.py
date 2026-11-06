"""
Unit tests for guppy.orchestration.visualize.
"""

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from guppy.orchestration.visualize import (
    _validate_results_exist,
    helper_plots,
    visualizeResults,
)


@pytest.fixture
def make_session():
    """Return a factory building a session directory that contains one output run folder."""

    def _make(tmp_path, name="session1"):
        session_dir = tmp_path / name
        session_dir.mkdir(parents=True, exist_ok=True)
        run_folder = session_dir / "output_1"
        run_folder.mkdir(parents=True, exist_ok=True)
        # select_run_folders validates that picked outputs have a storesList.csv.
        (run_folder / "storesList.csv").write_text("")
        return session_dir, run_folder

    return _make


@pytest.fixture
def make_parameters():
    """Return a factory for the minimal inputParameters the validator reads."""

    def _make(session_dir, *, selected_runs=("1",), selected_group_folders=()):
        parent = str(Path(session_dir).parent)
        return {
            "session_folders": [str(session_dir)],
            "combine_data": False,
            "selected_runs": {str(session_dir): list(selected_runs)},
            "selected_group_folders": list(selected_group_folders),
            "input_root_folder": parent,
            "output_root_folder": parent,
        }

    return _make


class TestValidateResultsExist:
    def test_passes_when_z_score_psth_files_present(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")

        _validate_results_exist(make_parameters(session_dir))

    def test_passes_when_dff_psth_files_present(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")

        _validate_results_exist(make_parameters(session_dir))

    def test_raises_when_the_run_folder_holds_no_psth_files(self, tmp_path, make_session, make_parameters):
        session_dir, _ = make_session(tmp_path)

        with pytest.raises(ValueError, match="Nothing to visualize was found"):
            _validate_results_exist(make_parameters(session_dir))

    def test_step3_hdf5_does_not_satisfy_the_check(self, tmp_path, make_session, make_parameters):
        """z_score_region.hdf5 (step-3 output) must not be mistaken for a PSTH file."""
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "z_score_region.hdf5").write_bytes(b"")

        with pytest.raises(ValueError, match="Nothing to visualize was found"):
            _validate_results_exist(make_parameters(session_dir))

    def test_transient_frequency_table_does_not_satisfy_the_check(self, tmp_path, make_session, make_parameters):
        """Step 4 writes freqAndAmp_z_score_region.h5 even for a run without events."""
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "freqAndAmp_z_score_region.h5").write_bytes(b"")

        with pytest.raises(ValueError, match="Nothing to visualize was found"):
            _validate_results_exist(make_parameters(session_dir))

    def test_raises_names_the_empty_output_directory(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)

        with pytest.raises(ValueError, match=re.escape(str(run_folder))):
            _validate_results_exist(make_parameters(session_dir))

    def test_raises_names_what_step_5_can_show(self, tmp_path, make_session, make_parameters):
        session_dir, _ = make_session(tmp_path)

        with pytest.raises(ValueError) as error:
            _validate_results_exist(make_parameters(session_dir))

        assert str(error.value).endswith(
            "Step 5 shows PSTH results, which need behavioral events and come from step 4 (or, for a "
            "'_group' directory, from the Group Analysis step); binned metrics and covariate correlations "
            "from step 4; and results from the Tonic Analysis step."
        )

    def test_passes_when_only_tonic_results_are_present(self, tmp_path, make_session, make_parameters):
        """A session without behavioral events can still hold Tonic Analysis results."""
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "tonic_region.h5").write_bytes(b"")

        _validate_results_exist(make_parameters(session_dir))

    def test_passes_when_only_binned_metrics_are_present(self, tmp_path, make_session, make_parameters):
        session_dir, run_folder = make_session(tmp_path)
        (run_folder / "binned_metrics_region.h5").write_bytes(b"")

        _validate_results_exist(make_parameters(session_dir))

    def test_one_populated_directory_is_enough(self, tmp_path, make_session):
        """A directory without results is skipped downstream, not fatal here."""
        session1_dir, run_folder_1 = make_session(tmp_path, "session1")
        session2_dir, _ = make_session(tmp_path, "session2")
        (run_folder_1 / "ttl_region_z_score_region.h5").write_bytes(b"")

        _validate_results_exist(
            {
                "session_folders": [str(session1_dir), str(session2_dir)],
                "combine_data": False,
                "selected_runs": {str(session1_dir): ["1"], str(session2_dir): ["1"]},
                "input_root_folder": str(tmp_path),
                "output_root_folder": str(tmp_path),
                "selected_group_folders": [],
            }
        )

    def test_no_op_when_no_output_directories(self, tmp_path, make_parameters):
        """When no *_output_* dirs exist the function returns silently.

        The homepage gate `validate_selected_runs_for_consumers` skips sessions with
        no output dirs on disk, so they reach orchestration with no entry in
        selected_runs. The validator must skip such sessions instead of erroring.
        """
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()

        _validate_results_exist(make_parameters(session_dir, selected_runs=()))

    def test_group_folder_with_results_passes(self, tmp_path, make_session, make_parameters):
        session_dir, _ = make_session(tmp_path)
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / "ttl_region_z_score_region.h5").write_bytes(b"")

        _validate_results_exist(make_parameters(session_dir, selected_group_folders=[str(group_folder)]))

    def test_group_folder_without_results_is_named(self, tmp_path, make_session, make_parameters):
        session_dir, _ = make_session(tmp_path)
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()

        with pytest.raises(ValueError, match=re.escape(str(group_folder))):
            _validate_results_exist(make_parameters(session_dir, selected_group_folders=[str(group_folder)]))


class TestVisualizeResultsSelectionSources:
    """Step 5 runs on whatever is selected: session runs, groups, or both."""

    @pytest.fixture
    def group_with_results(self, tmp_path):
        group_folder = tmp_path / "saline_group"
        group_folder.mkdir()
        (group_folder / "group_members.json").write_text('{"member_run_folders": []}')
        (group_folder / "storesList.csv").write_text("LNRW\nport_entries\n")
        (group_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
        return group_folder

    @pytest.fixture
    def parameters(self, tmp_path):
        return {
            "session_folders": [],
            "combine_data": False,
            "selected_runs": {},
            "selected_group_folders": [],
            "useTransientsAsEvents": False,
            "selectForTransientsComputation": "z_score",
        }

    def test_a_group_alone_is_visualized_without_any_selected_runs(
        self, monkeypatch, parameters, group_with_results, tmp_path, make_session
    ):
        """Sessions with no selected run are skipped, not fatal."""
        session_dir, _ = make_session(tmp_path)
        plotted = []
        monkeypatch.setattr(
            "guppy.orchestration.visualize.createPlots",
            lambda filepath, event, inputParameters, *, label: plotted.append(filepath),
        )
        parameters["session_folders"] = [str(session_dir)]
        parameters["selected_group_folders"] = [str(group_with_results)]

        visualizeResults(parameters)

        assert plotted == [str(group_with_results)]

    def test_saving_parameters_is_skipped_when_only_groups_are_selected(
        self, monkeypatch, parameters, group_with_results, tmp_path, make_session
    ):
        """A group-only run has no individual output dir to snapshot into."""
        session_dir, _ = make_session(tmp_path)
        saved = []
        monkeypatch.setattr("guppy.orchestration.visualize.createPlots", lambda *a, **k: None)
        monkeypatch.setattr(
            "guppy.orchestration.visualize.save_parameters",
            lambda *, inputParameters: saved.append(inputParameters),
        )
        parameters["session_folders"] = [str(session_dir)]
        parameters["selected_group_folders"] = [str(group_with_results)]

        visualizeResults(parameters)

        assert saved == []

    def test_raises_when_nothing_at_all_is_selected(self, parameters, tmp_path, make_session):
        session_dir, _ = make_session(tmp_path)
        parameters["session_folders"] = [str(session_dir)]

        with pytest.raises(ValueError, match="Nothing is selected to visualize"):
            visualizeResults(parameters)


class TestHelperPlots:
    """A directory the visualizer cannot plot is skipped, so the rest still open."""

    @pytest.fixture
    def run_folder_without_psth(self, tmp_path):
        run_folder = tmp_path / "session1_output_1"
        run_folder.mkdir()
        (run_folder / "storesList.csv").write_text("Dv1A,Dv2A,PrtN\ncontrol_region,signal_region,ttl\n")
        return run_folder

    def test_no_dashboard_is_opened_without_any_results(self, run_folder_without_psth):
        with patch("guppy.orchestration.visualize.VisualizationDashboard") as dashboard_class:
            helper_plots(
                str(run_folder_without_psth), ["ttl_region"], {"nSecPrev": -10, "nSecPost": 20}, label="output_1"
            )

        dashboard_class.assert_not_called()

    def test_the_skipped_directory_is_named_in_a_warning(self, run_folder_without_psth, caplog):
        with caplog.at_level(logging.WARNING, logger="guppy.orchestration.visualize"):
            helper_plots(
                str(run_folder_without_psth), ["ttl_region"], {"nSecPrev": -10, "nSecPost": 20}, label="output_1"
            )

        assert str(run_folder_without_psth) in caplog.text
        assert "Nothing to visualize was found" in caplog.text

    def test_no_dashboard_is_opened_without_behavior_events_or_whole_session_results(self, run_folder_without_psth):
        with patch("guppy.orchestration.visualize.VisualizationDashboard") as dashboard_class:
            helper_plots(str(run_folder_without_psth), [], {"nSecPrev": -10, "nSecPost": 20}, label="output_1")

        dashboard_class.assert_not_called()

    def test_a_dashboard_without_a_plotter_is_opened_for_tonic_results_alone(self, run_folder_without_psth):
        (run_folder_without_psth / "tonic_region.h5").write_bytes(b"")

        with patch("guppy.orchestration.visualize.VisualizationDashboard") as dashboard_class:
            dashboard_class.return_value.build_template.return_value.main.objects = []
            helper_plots(str(run_folder_without_psth), [], {"nSecPrev": -10, "nSecPost": 20}, label="output_1")

        dashboard_class.assert_called_once_with(
            filepath=str(run_folder_without_psth),
            plotter=None,
            basename="output_1",
            events=[],
            metric=None,
            available_metrics=[],
        )
