import json
import math
import os

import numpy as np
import panel as pn
import pytest

from guppy.frontend.frontend_utils import default_root_path
from guppy.frontend.input_parameters import ParameterForm, checkSameLocation, getAbsPath
from guppy.utils.utils import run_folder_for_run


@pytest.fixture(scope="session")
def frontend_base_dir(tmp_path_factory):
    """Create a real temp directory and point GUPPY_BASE_DIR at it.

    Ensures FileSelector resolves to a real path and prevents Tk dialogs.
    Restores the original value on teardown.
    """
    base_dir = tmp_path_factory.mktemp("frontend_base")
    original = os.environ.get("GUPPY_BASE_DIR")
    os.environ["GUPPY_BASE_DIR"] = str(base_dir)
    yield base_dir
    if original is None:
        del os.environ["GUPPY_BASE_DIR"]
    else:
        os.environ["GUPPY_BASE_DIR"] = original


@pytest.fixture
def bare_parameter_form(panel_extension, frontend_base_dir):
    """Build a BootstrapTemplate + ParameterForm with no files set."""
    template = pn.template.BootstrapTemplate(title="Test")
    return ParameterForm(template=template)


@pytest.fixture
def parameter_form(panel_extension, frontend_base_dir, tmp_path):
    """Build a BootstrapTemplate + ParameterForm backed by a real temp directory.

    Sets files_1.value to a list of session paths under tmp_path so that
    getInputParameters() can be called without raising.
    """
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    template = pn.template.BootstrapTemplate(title="Test")
    form = ParameterForm(template=template)
    form.files_1.value = [str(session_dir)]
    return form


# ── checkSameLocation ─────────────────────────────────────────────────────────


def test_check_same_location_same_parent_returns_single_element_array(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    paths = [str(parent / "a"), str(parent / "b")]
    result = checkSameLocation(paths, [])
    assert len(result) == 1
    assert result[0] == str(parent)


def test_check_same_location_different_parents_raises(tmp_path):
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    paths = [str(dir_a / "x"), str(dir_b / "y")]
    with pytest.raises(Exception, match="same location"):
        checkSameLocation(paths, [])


# ── getAbsPath ────────────────────────────────────────────────────────────────


def test_get_abs_path_both_empty_raises(bare_parameter_form):
    with pytest.raises(Exception, match="No folder"):
        getAbsPath(bare_parameter_form.files_1, bare_parameter_form.files_2)


def test_get_abs_path_files_1_populated_returns_parent(tmp_path, bare_parameter_form):
    parent = tmp_path / "data"
    parent.mkdir()
    session = parent / "session1"
    session.mkdir()
    bare_parameter_form.files_1.value = [str(session)]
    result = getAbsPath(bare_parameter_form.files_1, bare_parameter_form.files_2)
    assert len(result) == 1
    assert result[0] == str(parent)


def test_get_abs_path_files_2_used_when_files_1_empty(tmp_path, bare_parameter_form):
    parent = tmp_path / "data"
    parent.mkdir()
    session = parent / "session1"
    session.mkdir()
    bare_parameter_form.files_2.value = [str(session)]
    result = getAbsPath(bare_parameter_form.files_1, bare_parameter_form.files_2)
    assert len(result) == 1
    assert result[0] == str(parent)


# ── ParameterForm ─────────────────────────────────────────────────────────────


class TestParameterForm:
    def test_time_for_lights_turn_on_default(self, parameter_form):
        assert parameter_form.timeForLightsTurnOn.value == 1

    def test_number_of_cores_default(self, parameter_form):
        assert parameter_form.numberOfCores.value == 2

    def test_moving_avg_filter_default(self, parameter_form):
        assert parameter_form.moving_avg_filter.value == 100

    def test_moving_wd_default(self, parameter_form):
        assert parameter_form.moving_wd.value == 15

    def test_high_amp_filt_default(self, parameter_form):
        assert parameter_form.highAmpFilt.value == 2

    def test_transients_thresh_default(self, parameter_form):
        assert parameter_form.transientsThresh.value == 3

    def test_no_channels_np_default(self, parameter_form):
        assert parameter_form.no_channels_np.value == 2

    def test_n_sec_prev_default(self, parameter_form):
        assert parameter_form.nSecPrev.value == -10

    def test_n_sec_post_default(self, parameter_form):
        assert parameter_form.nSecPost.value == 20

    def test_time_interval_default(self, parameter_form):
        assert parameter_form.timeInterval.value == 2

    def test_bin_psth_trials_default(self, parameter_form):
        assert parameter_form.bin_psth_trials.value == 0

    def test_baseline_wd_strt_default(self, parameter_form):
        assert parameter_form.baseline_wd_strt.value == 0

    def test_baseline_wd_end_default(self, parameter_form):
        assert parameter_form.baseline_wd_end.value == 0

    def test_baseline_correction_start_default(self, parameter_form):
        assert parameter_form.baselineCorrectionStart.value == -5

    def test_baseline_correction_end_default(self, parameter_form):
        assert parameter_form.baselineCorrectionEnd.value == 0

    def test_isosbestic_control_default(self, parameter_form):
        assert parameter_form.isosbestic_control.value is True
        assert parameter_form.isosbestic_control.options == [True, False]

    def test_combine_data_default(self, parameter_form):
        assert parameter_form.combine_data.value is False
        assert parameter_form.combine_data.options == [True, False]

    def test_compute_psth_default(self, parameter_form):
        assert parameter_form.computePsth.value == "z_score"
        assert "z_score" in parameter_form.computePsth.options
        assert "dff" in parameter_form.computePsth.options
        assert "Both" in parameter_form.computePsth.options

    def test_transients_default(self, parameter_form):
        assert parameter_form.transients.value == "z_score"

    def test_plot_z_score_dff_default(self, parameter_form):
        assert parameter_form.plot_zScore_dff.value == "None"

    def test_remove_artifacts_default(self, parameter_form):
        assert parameter_form.removeArtifacts.value is False
        assert parameter_form.removeArtifacts.options == [True, False]

    def test_artifacts_removal_method_default(self, parameter_form):
        assert parameter_form.artifactsRemovalMethod.value == "replace with NaN"
        assert "concatenate" in parameter_form.artifactsRemovalMethod.options
        assert "replace with NaN" in parameter_form.artifactsRemovalMethod.options

    def test_z_score_computation_default(self, parameter_form):
        assert parameter_form.z_score_computation.value == "standard z-score"
        assert "standard z-score" in parameter_form.z_score_computation.options
        assert "baseline z-score" in parameter_form.z_score_computation.options
        assert "modified z-score" in parameter_form.z_score_computation.options

    def test_compute_corr_default(self, parameter_form):
        assert parameter_form.computeCorr.value is False
        assert parameter_form.computeCorr.options == [True, False]

    def test_use_time_or_trials_default(self, parameter_form):
        assert parameter_form.use_time_or_trials.value == "Time (min)"
        assert "Time (min)" in parameter_form.use_time_or_trials.options
        assert "# of trials" in parameter_form.use_time_or_trials.options

    def test_average_for_group_default(self, parameter_form):
        assert parameter_form.averageForGroup.value is False

    def test_visualize_average_results_default(self, parameter_form):
        assert parameter_form.visualizeAverageResults.value is False

    def test_visualize_zscore_or_dff_default(self, parameter_form):
        assert parameter_form.visualize_zscore_or_dff.value == "z_score"
        assert "z_score" in parameter_form.visualize_zscore_or_dff.options
        assert "dff" in parameter_form.visualize_zscore_or_dff.options

    def test_df_widget_initial_peak_start_values(self, parameter_form):
        df = parameter_form.df_widget.value
        assert df["Peak Start time"].iloc[0] == -5
        assert df["Peak Start time"].iloc[1] == 0
        assert df["Peak Start time"].iloc[2] == 5

    def test_df_widget_initial_peak_end_values(self, parameter_form):
        df = parameter_form.df_widget.value
        assert df["Peak End time"].iloc[0] == 0
        assert df["Peak End time"].iloc[1] == 3
        assert df["Peak End time"].iloc[2] == 10

    def test_df_widget_remaining_rows_are_nan(self, parameter_form):
        df = parameter_form.df_widget.value
        for row_index in range(3, len(df)):
            assert math.isnan(df["Peak Start time"].iloc[row_index])
            assert math.isnan(df["Peak End time"].iloc[row_index])

    def test_time_for_lights_turn_on_mutation(self, parameter_form):
        parameter_form.timeForLightsTurnOn.value = 5
        assert parameter_form.timeForLightsTurnOn.value == 5

    def test_isosbestic_control_mutation(self, parameter_form):
        parameter_form.isosbestic_control.value = False
        assert parameter_form.isosbestic_control.value is False

    def test_get_input_parameters_returns_expected_keys(self, parameter_form):
        result = parameter_form.getInputParameters()
        for key in (
            "abspath",
            "session_folders",
            "numberOfCores",
            "timeForLightsTurnOn",
            "nSecPrev",
            "nSecPost",
            "zscore_method",
        ):
            assert key in result

    def test_get_input_parameters_default_scalar_values(self, parameter_form):
        result = parameter_form.getInputParameters()
        assert result["numberOfCores"] == 2
        assert result["timeForLightsTurnOn"] == 1
        assert result["nSecPrev"] == -10
        assert result["nSecPost"] == 20
        assert result["zscore_method"] == "standard z-score"

    def test_source_mode_default_is_local(self, parameter_form):
        assert parameter_form.source_mode.value == "local"
        assert parameter_form.files_1.visible is True
        assert parameter_form.dandi_selector.panel.visible is False

    def test_source_mode_toggle_to_dandi_shows_dandi_panel(self, parameter_form):
        parameter_form.source_mode.value = "dandi"
        assert parameter_form.files_1.visible is False
        assert parameter_form.dandi_selector.panel.visible is True

    def test_source_mode_toggle_back_to_local_restores(self, parameter_form):
        parameter_form.source_mode.value = "dandi"
        parameter_form.source_mode.value = "local"
        assert parameter_form.files_1.visible is True
        assert parameter_form.dandi_selector.panel.visible is False

    def test_get_input_parameters_local_mode_sets_mode_and_no_dandi_map(self, parameter_form):
        result = parameter_form.getInputParameters()
        assert result["mode"] == "local"
        assert result["dandi_uri_map"] is None


class TestNumericParameterValidation:
    def test_defaults_pass(self, parameter_form):
        # The default form values are all valid, so config-time validation must not raise.
        parameter_form.getInputParameters()

    def test_zero_cores_raises(self, parameter_form):
        parameter_form.numberOfCores.value = 0
        with pytest.raises(ValueError, match="numberOfCores=0 must be greater than 0"):
            parameter_form.getInputParameters()

    def test_cores_exceeding_host_raises(self, parameter_form):
        parameter_form.numberOfCores.value = 10_000_000
        with pytest.raises(ValueError, match=r"exceeds the \d+ core\(s\) available"):
            parameter_form.getInputParameters()

    def test_negative_filter_window_raises(self, parameter_form):
        parameter_form.moving_avg_filter.value = -1
        with pytest.raises(ValueError, match="filter_window=-1 must be 0 or greater"):
            parameter_form.getInputParameters()

    def test_zero_filter_window_allowed(self, parameter_form):
        parameter_form.moving_avg_filter.value = 0
        parameter_form.getInputParameters()

    def test_negative_time_for_lights_turn_on_raises(self, parameter_form):
        parameter_form.timeForLightsTurnOn.value = -1
        with pytest.raises(ValueError, match="timeForLightsTurnOn=-1 must be 0 or greater"):
            parameter_form.getInputParameters()

    def test_zero_time_for_lights_turn_on_allowed(self, parameter_form):
        parameter_form.timeForLightsTurnOn.value = 0
        parameter_form.getInputParameters()

    def test_zero_moving_window_raises(self, parameter_form):
        parameter_form.moving_wd.value = 0
        with pytest.raises(ValueError, match="moving_window=0 must be greater than 0"):
            parameter_form.getInputParameters()

    def test_zero_high_amp_filt_raises(self, parameter_form):
        parameter_form.highAmpFilt.value = 0
        with pytest.raises(ValueError, match="highAmpFilt=0 must be greater than 0"):
            parameter_form.getInputParameters()

    def test_negative_transients_thresh_raises(self, parameter_form):
        parameter_form.transientsThresh.value = -3
        with pytest.raises(ValueError, match="transientsThresh=-3 must be greater than 0"):
            parameter_form.getInputParameters()

    def test_nsecprev_equal_to_nsecpost_raises(self, parameter_form):
        parameter_form.nSecPrev.value = 5
        parameter_form.nSecPost.value = 5
        with pytest.raises(ValueError, match="nSecPrev=5 must be strictly less than nSecPost=5"):
            parameter_form.getInputParameters()

    def test_nsecprev_greater_than_nsecpost_raises(self, parameter_form):
        parameter_form.nSecPrev.value = 30
        parameter_form.nSecPost.value = 20
        with pytest.raises(ValueError, match="nSecPrev=30 must be strictly less than nSecPost=20"):
            parameter_form.getInputParameters()


class _FakeAsset:
    def __init__(self, path):
        self.path = path


class _FakeDandiset:
    def __init__(self, asset_paths):
        self._asset_paths = asset_paths

    def get_assets(self):
        return [_FakeAsset(path) for path in self._asset_paths]


class _FakeDandiAPIClient:
    dandisets_by_id = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_dandiset(self, dandiset_id, version=None):
        return self.dandisets_by_id[dandiset_id]


@pytest.fixture
def patched_dandi_client(monkeypatch, tmp_path):
    from guppy.frontend import dandi_selector as dandi_selector_module

    monkeypatch.setattr(dandi_selector_module, "DandiAPIClient", _FakeDandiAPIClient)
    # Point the mirror parent at tmp_path so tests don't pollute the real temp dir.
    monkeypatch.setattr(dandi_selector_module, "_MIRROR_ROOT", str(tmp_path / "dandi_mirror"))
    return _FakeDandiAPIClient


class TestParameterFormDandiMode:
    def test_dandi_mode_builds_uri_map_and_session_dirs(self, bare_parameter_form, tmp_path, patched_dandi_client):
        output_root = tmp_path / "dandi_output"
        output_root.mkdir()
        patched_dandi_client.dandisets_by_id = {
            "000971": _FakeDandiset(["sub-01/session_a.nwb", "sub-02/session_b.nwb"]),
        }
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.dandiset_input.value = "000971"
        mirror_root = form.dandi_selector._current_mirror_root
        form.dandi_selector.asset_file_selector.value = [
            os.path.join(mirror_root, "sub-01", "session_a.nwb"),
            os.path.join(mirror_root, "sub-02", "session_b.nwb"),
        ]
        form.dandi_selector.output_root_selector.value = [str(output_root)]

        result = form.getInputParameters()

        assert result["mode"] == "dandi"
        assert result["abspath"] == str(output_root)
        session_a = str(output_root / "session_a")
        session_b = str(output_root / "session_b")
        assert sorted(result["session_folders"]) == sorted([session_a, session_b])
        assert result["dandi_uri_map"] == {
            session_a: "dandi://000971/sub-01/session_a.nwb",
            session_b: "dandi://000971/sub-02/session_b.nwb",
        }
        for session_dir in (session_a, session_b):
            assert os.path.isdir(session_dir)

    def test_dandi_mode_no_asset_raises(self, bare_parameter_form, tmp_path):
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.output_root_selector.value = [str(tmp_path)]
        with pytest.raises(Exception, match="select at least one NWB asset"):
            form.getInputParameters()

    def test_dandi_mode_no_output_root_raises(self, bare_parameter_form, patched_dandi_client):
        patched_dandi_client.dandisets_by_id = {"000971": _FakeDandiset(["sub-01/data.nwb"])}
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.dandiset_input.value = "000971"
        mirror_root = form.dandi_selector._current_mirror_root
        form.dandi_selector.asset_file_selector.value = [os.path.join(mirror_root, "sub-01", "data.nwb")]
        with pytest.raises(Exception, match="local output directory"):
            form.getInputParameters()


class TestOutputsSelector:
    def test_outputs_selector_exists_and_is_filtered(self, parameter_form):
        assert isinstance(parameter_form.outputs_selector, pn.widgets.FileSelector)
        assert parameter_form.outputs_selector.file_pattern == "*_output_*"

    def test_retarget_uses_first_session_as_directory(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        # root_directory must equal directory so Panel's startswith() validation in _dir_change
        # can't silently revert (especially on Windows where root_directory="/" resolves to a
        # potentially different drive than tmp_path).
        assert bare_parameter_form.outputs_selector.root_directory == str(session)
        assert bare_parameter_form.outputs_selector.directory == str(session)
        assert bare_parameter_form.outputs_selector.value == []

    def test_retarget_falls_back_to_default_root_when_files_1_cleared(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.files_1.value = []
        assert bare_parameter_form.outputs_selector.directory == default_root_path()

    def test_retarget_multiple_sessions_uses_common_parent_as_root(self, bare_parameter_form, tmp_path):
        # Multi-session: root must be the common parent so the user can navigate between
        # sessions to reach each one's _output_* dirs. Directory starts at sessions[0] so
        # the FileSelector lands on one session's outputs immediately.
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        bare_parameter_form.files_1.value = [str(session_a), str(session_b)]
        assert bare_parameter_form.outputs_selector.root_directory == str(tmp_path)
        assert bare_parameter_form.outputs_selector.directory == str(session_a)

    def test_retarget_clears_stale_outputs_selector_value(self, bare_parameter_form, tmp_path):
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        bare_parameter_form.files_1.value = [str(session_a)]
        bare_parameter_form.outputs_selector.value = [str(session_a / "stale_output_x")]

        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        bare_parameter_form.files_1.value = [str(session_b)]
        assert bare_parameter_form.outputs_selector.value == []

    def test_collect_selected_outputs_groups_by_session(self, bare_parameter_form, tmp_path):
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        run_a1 = run_folder_for_run(str(session_a), "run1")
        run_a2 = run_folder_for_run(str(session_a), "run2")
        run_b1 = run_folder_for_run(str(session_b), "run1")
        for path in (run_a1, run_a2, run_b1):
            os.mkdir(path)

        bare_parameter_form.outputs_selector.value = [run_a1, run_a2, run_b1]
        result = bare_parameter_form._collect_selected_runs()
        assert result == {
            str(session_a): ["run1", "run2"],
            str(session_b): ["run1"],
        }

    def test_collect_selected_outputs_empty_returns_empty_dict(self, bare_parameter_form):
        bare_parameter_form.outputs_selector.value = []
        assert bare_parameter_form._collect_selected_runs() == {}

    def test_validate_selected_outputs_raises_when_session_has_dirs_but_none_selected(
        self, bare_parameter_form, tmp_path
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        os.mkdir(run_folder_for_run(str(session), "baseline"))

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = []

        with pytest.raises(ValueError, match="No output directory selected"):
            bare_parameter_form.validate_selected_runs_for_consumers()

    def test_validate_selected_outputs_skips_sessions_without_output_dirs(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = []

        bare_parameter_form.validate_selected_runs_for_consumers()

    def test_get_input_parameters_omits_run_name_keys(self, parameter_form):
        result = parameter_form.getInputParameters()
        assert "run_name" not in result
        assert "run_name_policy" not in result

    def test_get_input_parameters_selected_outputs_reflects_selector(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_dir = run_folder_for_run(str(session), "baseline")
        os.mkdir(run_dir)

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        result = bare_parameter_form.getInputParameters()
        assert result["selected_runs"] == {str(session): ["baseline"]}


class TestRebuildPerSessionWidgets:
    def test_preserves_existing_widget_value_across_rebuilds(self, bare_parameter_form, tmp_path):
        """When files_2 fires twice and the prior selection still exists, preserve it."""
        session = tmp_path / "sessionA"
        session.mkdir()
        os.mkdir(run_folder_for_run(str(session), "run1"))
        os.mkdir(run_folder_for_run(str(session), "run2"))

        bare_parameter_form.files_2.value = [str(session)]
        widget = bare_parameter_form.group_selected_outputs_widgets[str(session)]
        widget.value = "run2"

        # Rebuild with the same session — existing widget is reused, "run2" preserved.
        bare_parameter_form.files_2.param.trigger("value")
        reused_widget = bare_parameter_form.group_selected_outputs_widgets[str(session)]
        assert reused_widget is widget
        assert reused_widget.value == "run2"

    def test_resets_existing_widget_value_when_prior_selection_invalid(self, bare_parameter_form, tmp_path):
        """When the prior selection no longer exists in run_names, fall back to the first option."""
        session = tmp_path / "sessionA"
        session.mkdir()
        run1 = run_folder_for_run(str(session), "run1")
        run2 = run_folder_for_run(str(session), "run2")
        os.mkdir(run1)
        os.mkdir(run2)

        bare_parameter_form.files_2.value = [str(session)]
        widget = bare_parameter_form.group_selected_outputs_widgets[str(session)]
        widget.value = "run2"

        # Remove run2 so the prior selection becomes invalid; rebuild.
        os.rmdir(run2)
        bare_parameter_form.files_2.param.trigger("value")

        reused_widget = bare_parameter_form.group_selected_outputs_widgets[str(session)]
        assert reused_widget is widget
        assert reused_widget.value == "run1"
        assert reused_widget.options == ["run1"]


class TestFolderSelectionCards:
    def test_input_folder_selection_card_exists_and_is_open(self, parameter_form):
        assert isinstance(parameter_form.input_folder_selection, pn.Card)
        assert parameter_form.input_folder_selection.title == "Input Folder Selection"
        assert parameter_form.input_folder_selection.collapsed is False

    def test_output_folder_selection_card_exists_and_is_collapsed(self, parameter_form):
        assert isinstance(parameter_form.output_folder_selection, pn.Card)
        assert parameter_form.output_folder_selection.title == "Output Folder Selection"
        assert parameter_form.output_folder_selection.collapsed is True

    def test_individual_card_starts_collapsed(self, parameter_form):
        assert parameter_form.individual.collapsed is True

    def test_add_to_template_appends_input_then_output_first(self, parameter_form):
        main = parameter_form.template.main
        assert main[0] is parameter_form.input_folder_selection
        assert main[1] is parameter_form.output_folder_selection
        assert main[2] is parameter_form.individual
        assert main[3] is parameter_form.group
        assert main[4] is parameter_form.visualize


# Distinctive non-default snapshot so a successful load is unambiguous. peak_*Point
# carry NaN tail entries exactly as save_parameters serializes them.
SAVED_PARAMETERS = {
    "guppy_version": "test-version",
    "combine_data": True,
    "isosbestic_control": False,
    "control_fit_method": "OLS",
    "controlFitWindowMode": "baseline epoch",
    "controlFitWindowStart": 3,
    "controlFitWindowEnd": 8,
    "timeForLightsTurnOn": 7,
    "filter_window": 42,
    "removeArtifacts": True,
    "artifactsRemovalMethod": "replace with NaN",
    "computeTonic": True,
    "noChannels": 3,
    "zscore_method": "modified z-score",
    "baselineWindowStart": 2,
    "baselineWindowEnd": 9,
    "nSecPrev": -3,
    "nSecPost": 8,
    "computeCorr": True,
    "timeInterval": 5,
    "bin_psth_trials": 4,
    "use_time_or_trials": "# of trials",
    "baselineCorrectionStart": -2,
    "baselineCorrectionEnd": 1,
    "peak_startPoint": [-4.0, 1.0, 6.0] + [float("nan")] * 7,
    "peak_endPoint": [1.0, 4.0, 11.0] + [float("nan")] * 7,
    "selectForComputePsth": "dff",
    "selectForTransientsComputation": "Both",
    "moving_window": 12,
    "highAmpFilt": 5,
    "transientsThresh": 6,
    "plot_zScore_dff": "Both",
    "visualize_zscore_or_dff": "dff",
    "averageForGroup": True,
}


def _write_run_with_parameters(session_dir, run_name, parameters):
    """Create an ``_output_<run>`` dir under session_dir holding a GuPPyParamtersUsed.json."""
    run_dir = run_folder_for_run(str(session_dir), run_name)
    os.mkdir(run_dir)
    with open(os.path.join(run_dir, "GuPPyParamtersUsed.json"), "w") as parameters_file:
        json.dump(parameters, parameters_file)
    return run_dir


class TestParameterAutoPopulate:
    def test_set_input_parameters_round_trips_get_input_parameters(self, parameter_form):
        parameter_form.setInputParameters(SAVED_PARAMETERS)
        result = parameter_form.getInputParameters()
        for key, widget in parameter_form._scalar_parameter_widgets().items():
            assert result[key] == SAVED_PARAMETERS[key], f"{key} did not round-trip"
        # NaN tail entries compare equal only via isnan.
        np.testing.assert_array_equal(result["peak_startPoint"], SAVED_PARAMETERS["peak_startPoint"])
        np.testing.assert_array_equal(result["peak_endPoint"], SAVED_PARAMETERS["peak_endPoint"])

    def test_set_input_parameters_ignores_unknown_keys(self, parameter_form):
        # guppy_version has no backing widget and must be ignored without error.
        parameter_form.setInputParameters({"guppy_version": "x", "nSecPost": 99})
        assert parameter_form.nSecPost.value == 99

    def test_selecting_output_run_populates_widgets(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_dir = _write_run_with_parameters(session, "baseline", SAVED_PARAMETERS)

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        assert bare_parameter_form.timeForLightsTurnOn.value == 7
        assert bare_parameter_form.z_score_computation.value == "modified z-score"
        assert bare_parameter_form.combine_data.value is True
        assert bare_parameter_form.nSecPrev.value == -3
        assert list(bare_parameter_form.df_widget.value["Peak Start time"])[:3] == [-4.0, 1.0, 6.0]

    def test_agreeing_runs_populate_widgets(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_a = _write_run_with_parameters(session, "run_a", SAVED_PARAMETERS)
        run_b = _write_run_with_parameters(session, "run_b", SAVED_PARAMETERS)

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_a, run_b]

        assert bare_parameter_form.timeForLightsTurnOn.value == 7

    def test_conflicting_runs_leave_form_unchanged(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_a = _write_run_with_parameters(session, "run_a", SAVED_PARAMETERS)
        conflicting = {**SAVED_PARAMETERS, "timeForLightsTurnOn": 99}
        run_b = _write_run_with_parameters(session, "run_b", conflicting)

        default_time = bare_parameter_form.timeForLightsTurnOn.value
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_a, run_b]

        # Conflicting snapshots: form must be left untouched for the user to reconcile.
        assert bare_parameter_form.timeForLightsTurnOn.value == default_time

    def test_selecting_run_without_json_is_noop(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_dir = run_folder_for_run(str(session), "fresh")
        os.mkdir(run_dir)

        default_time = bare_parameter_form.timeForLightsTurnOn.value
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        assert bare_parameter_form.timeForLightsTurnOn.value == default_time
