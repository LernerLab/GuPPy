import math

import pytest

from guppy.frontend.input_parameters import checkSameLocation, getAbsPath

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


def test_get_abs_path_both_empty_raises(panel_extension, frontend_base_dir):
    import panel as pn

    from guppy.frontend.input_parameters import ParameterForm

    template = pn.template.BootstrapTemplate(title="Test")
    form = ParameterForm(template=template)
    form.files_1.value = []
    form.files_2.value = []
    with pytest.raises(Exception, match="No folder"):
        getAbsPath(form.files_1, form.files_2)


def test_get_abs_path_files_1_populated_returns_parent(tmp_path, panel_extension, frontend_base_dir):
    import panel as pn

    from guppy.frontend.input_parameters import ParameterForm

    parent = tmp_path / "data"
    parent.mkdir()
    session = parent / "session1"
    session.mkdir()
    template = pn.template.BootstrapTemplate(title="Test")
    form = ParameterForm(template=template)
    form.files_1.value = [str(session)]
    form.files_2.value = []
    result = getAbsPath(form.files_1, form.files_2)
    assert len(result) == 1
    assert result[0] == str(parent)


def test_get_abs_path_files_2_used_when_files_1_empty(tmp_path, panel_extension, frontend_base_dir):
    import panel as pn

    from guppy.frontend.input_parameters import ParameterForm

    parent = tmp_path / "data"
    parent.mkdir()
    session = parent / "session1"
    session.mkdir()
    template = pn.template.BootstrapTemplate(title="Test")
    form = ParameterForm(template=template)
    form.files_1.value = []
    form.files_2.value = [str(session)]
    result = getAbsPath(form.files_1, form.files_2)
    assert len(result) == 1
    assert result[0] == str(parent)


# ── ParameterForm default widget values ───────────────────────────────────────


def test_parameter_form_time_for_lights_turn_on_default(parameter_form):
    assert parameter_form.timeForLightsTurnOn.value == 1


def test_parameter_form_number_of_cores_default(parameter_form):
    assert parameter_form.numberOfCores.value == 2


def test_parameter_form_moving_avg_filter_default(parameter_form):
    assert parameter_form.moving_avg_filter.value == 100


def test_parameter_form_moving_wd_default(parameter_form):
    assert parameter_form.moving_wd.value == 15


def test_parameter_form_high_amp_filt_default(parameter_form):
    assert parameter_form.highAmpFilt.value == 2


def test_parameter_form_transients_thresh_default(parameter_form):
    assert parameter_form.transientsThresh.value == 3


def test_parameter_form_no_channels_np_default(parameter_form):
    assert parameter_form.no_channels_np.value == 2


def test_parameter_form_n_sec_prev_default(parameter_form):
    assert parameter_form.nSecPrev.value == -10


def test_parameter_form_n_sec_post_default(parameter_form):
    assert parameter_form.nSecPost.value == 20


def test_parameter_form_time_interval_default(parameter_form):
    assert parameter_form.timeInterval.value == 2


def test_parameter_form_bin_psth_trials_default(parameter_form):
    assert parameter_form.bin_psth_trials.value == 0


def test_parameter_form_baseline_wd_strt_default(parameter_form):
    assert parameter_form.baseline_wd_strt.value == 0


def test_parameter_form_baseline_wd_end_default(parameter_form):
    assert parameter_form.baseline_wd_end.value == 0


def test_parameter_form_baseline_correction_start_default(parameter_form):
    assert parameter_form.baselineCorrectionStart.value == -5


def test_parameter_form_baseline_correction_end_default(parameter_form):
    assert parameter_form.baselineCorrectionEnd.value == 0


def test_parameter_form_isosbestic_control_default(parameter_form):
    assert parameter_form.isosbestic_control.value is True
    assert parameter_form.isosbestic_control.options == [True, False]


def test_parameter_form_combine_data_default(parameter_form):
    assert parameter_form.combine_data.value is False
    assert parameter_form.combine_data.options == [True, False]


def test_parameter_form_compute_psth_default(parameter_form):
    assert parameter_form.computePsth.value == "z_score"
    assert "z_score" in parameter_form.computePsth.options
    assert "dff" in parameter_form.computePsth.options
    assert "Both" in parameter_form.computePsth.options


def test_parameter_form_transients_default(parameter_form):
    assert parameter_form.transients.value == "z_score"


def test_parameter_form_plot_z_score_dff_default(parameter_form):
    assert parameter_form.plot_zScore_dff.value == "None"


def test_parameter_form_remove_artifacts_default(parameter_form):
    assert parameter_form.removeArtifacts.value is False
    assert parameter_form.removeArtifacts.options == [True, False]


def test_parameter_form_artifacts_removal_method_default(parameter_form):
    assert parameter_form.artifactsRemovalMethod.value == "concatenate"
    assert "concatenate" in parameter_form.artifactsRemovalMethod.options
    assert "replace with NaN" in parameter_form.artifactsRemovalMethod.options


def test_parameter_form_z_score_computation_default(parameter_form):
    assert parameter_form.z_score_computation.value == "standard z-score"
    assert "standard z-score" in parameter_form.z_score_computation.options
    assert "baseline z-score" in parameter_form.z_score_computation.options
    assert "modified z-score" in parameter_form.z_score_computation.options


def test_parameter_form_compute_corr_default(parameter_form):
    assert parameter_form.computeCorr.value is False
    assert parameter_form.computeCorr.options == [True, False]


def test_parameter_form_use_time_or_trials_default(parameter_form):
    assert parameter_form.use_time_or_trials.value == "Time (min)"
    assert "Time (min)" in parameter_form.use_time_or_trials.options
    assert "# of trials" in parameter_form.use_time_or_trials.options


def test_parameter_form_average_for_group_default(parameter_form):
    assert parameter_form.averageForGroup.value is False


def test_parameter_form_visualize_average_results_default(parameter_form):
    assert parameter_form.visualizeAverageResults.value is False


def test_parameter_form_visualize_zscore_or_dff_default(parameter_form):
    assert parameter_form.visualize_zscore_or_dff.value == "z_score"
    assert "z_score" in parameter_form.visualize_zscore_or_dff.options
    assert "dff" in parameter_form.visualize_zscore_or_dff.options


def test_parameter_form_df_widget_initial_peak_start_values(parameter_form):
    df = parameter_form.df_widget.value
    assert df["Peak Start time"].iloc[0] == -5
    assert df["Peak Start time"].iloc[1] == 0
    assert df["Peak Start time"].iloc[2] == 5


def test_parameter_form_df_widget_initial_peak_end_values(parameter_form):
    df = parameter_form.df_widget.value
    assert df["Peak End time"].iloc[0] == 0
    assert df["Peak End time"].iloc[1] == 3
    assert df["Peak End time"].iloc[2] == 10


def test_parameter_form_df_widget_remaining_rows_are_nan(parameter_form):
    df = parameter_form.df_widget.value
    for row_index in range(3, len(df)):
        assert math.isnan(df["Peak Start time"].iloc[row_index])
        assert math.isnan(df["Peak End time"].iloc[row_index])


# ── ParameterForm value mutation ──────────────────────────────────────────────


def test_parameter_form_time_for_lights_turn_on_mutation(parameter_form):
    parameter_form.timeForLightsTurnOn.value = 5
    assert parameter_form.timeForLightsTurnOn.value == 5


def test_parameter_form_isosbestic_control_mutation(parameter_form):
    parameter_form.isosbestic_control.value = False
    assert parameter_form.isosbestic_control.value is False


# ── getInputParameters ────────────────────────────────────────────────────────


def test_get_input_parameters_returns_expected_keys(parameter_form):
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


def test_get_input_parameters_default_scalar_values(parameter_form):
    result = parameter_form.getInputParameters()
    assert result["numberOfCores"] == 2
    assert result["timeForLightsTurnOn"] == 1
    assert result["nSecPrev"] == -10
    assert result["nSecPost"] == 20
    assert result["zscore_method"] == "standard z-score"
