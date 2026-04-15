import math
import os

import panel as pn
import pytest

from guppy.frontend.input_parameters import ParameterForm, checkSameLocation, getAbsPath


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
        assert parameter_form.artifactsRemovalMethod.value == "concatenate"
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
            "folderNames",
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


class TestParameterFormDandiMode:
    def test_dandi_mode_builds_uri_map_and_session_dirs(self, bare_parameter_form, tmp_path):
        output_root = tmp_path / "dandi_output"
        output_root.mkdir()
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.dandiset_input.options = ["000971"]
        form.dandi_selector.dandiset_input.value = "000971"
        form.dandi_selector.asset_select.options = ["sub-01/ses-1.nwb", "sub-02/ses-1.nwb"]
        form.dandi_selector.asset_select.value = ["sub-01/ses-1.nwb", "sub-02/ses-1.nwb"]
        form.dandi_selector.output_root_selector.value = [str(output_root)]

        result = form.getInputParameters()

        assert result["mode"] == "dandi"
        assert result["abspath"] == str(output_root)
        expected_session_dirs = [
            str(output_root / "ses-1"),  # Both assets share basename 'ses-1'; dict key dedupes
        ]
        # Two assets with the same basename collide intentionally — exercise distinct paths.
        form.dandi_selector.asset_select.value = ["sub-01/session_a.nwb", "sub-02/session_b.nwb"]
        result = form.getInputParameters()
        session_a = str(output_root / "session_a")
        session_b = str(output_root / "session_b")
        assert sorted(result["folderNames"]) == sorted([session_a, session_b])
        assert result["dandi_uri_map"] == {
            session_a: "dandi://000971/sub-01/session_a.nwb",
            session_b: "dandi://000971/sub-02/session_b.nwb",
        }
        # Session directories were created on disk
        import os as _os

        for session_dir in (session_a, session_b):
            assert _os.path.isdir(session_dir)

    def test_dandi_mode_no_asset_raises(self, bare_parameter_form, tmp_path):
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.output_root_selector.value = [str(tmp_path)]
        with pytest.raises(Exception, match="select at least one NWB asset"):
            form.getInputParameters()

    def test_dandi_mode_no_output_root_raises(self, bare_parameter_form):
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        form.dandi_selector.dandiset_input.options = ["000971"]
        form.dandi_selector.dandiset_input.value = "000971"
        form.dandi_selector.asset_select.options = ["sub-01/data.nwb"]
        form.dandi_selector.asset_select.value = ["sub-01/data.nwb"]
        with pytest.raises(Exception, match="local output directory"):
            form.getInputParameters()
