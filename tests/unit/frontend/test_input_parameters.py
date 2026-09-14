import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.frontend.input_parameters import (
    ParameterForm,
    _table_heading,
    _titled_box,
)
from guppy.utils.utils import (
    run_directory_root,
    run_folder_for_run,
)


@pytest.fixture(scope="session")
def frontend_base_dir(tmp_path_factory):
    """Create a real temp directory for the form's file selectors to start in."""
    return tmp_path_factory.mktemp("frontend_base")


@pytest.fixture
def notifications(monkeypatch):
    """Capture Panel notifications, which are None outside a served session."""
    captured = []

    class RecordingNotifications:
        def __getattr__(self, level):
            def record(message, **kwargs):
                captured.append((level, message, kwargs))

            return record

    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: RecordingNotifications()))
    return captured


@pytest.fixture
def output_root_folder(tmp_path):
    """The directory the form mirrors ``tmp_path`` into."""
    base = tmp_path / "derivatives"
    base.mkdir()
    return base


@pytest.fixture
def unconfigured_parameter_form(panel_extension, frontend_base_dir):
    """Build a BootstrapTemplate + ParameterForm with nothing chosen at all."""
    template = pn.template.BootstrapTemplate(title="Test")
    return ParameterForm(template=template, start_path=str(frontend_base_dir))


@pytest.fixture
def bare_parameter_form(unconfigured_parameter_form, tmp_path, output_root_folder):
    """Build a ParameterForm pointed at ``tmp_path`` as its data root, with no files set."""
    form = unconfigured_parameter_form
    form.input_root_selector.value = [str(tmp_path)]
    form.output_root_selector.value = [str(output_root_folder)]
    return form


@pytest.fixture
def parameter_form(bare_parameter_form, tmp_path):
    """Build a ParameterForm with one session selected, so getInputParameters() can run."""
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    bare_parameter_form.files_1.value = [str(session_dir)]
    return bare_parameter_form


# The cards ParameterForm appends to the template, in display order.
_CARD_ATTRIBUTES = (
    "root_folder_selection",
    "input_folder_selection",
    "output_folder_selection",
    "individual",
    "group",
)

# The titled sections stacked inside the Parameter Selection card, in display order.
_SECTION_ATTRIBUTES = (
    "execution_param_wd",
    "control_fit_param_wd",
    "filtering_param_wd",
    "zscore_param_wd",
    "psth_param_wd",
    "peak_param_wd",
    "transients_param_wd",
    "binned_metrics_param_wd",
    "significance_param_wd",
)

# Panel puts a default 5px margin either side of every object, so each member of a
# row of siblings occupies its own width plus 10px.
_SIBLING_MARGIN = 10


def _occupied_width(node) -> int:
    """Return the horizontal extent ``node`` takes up inside its parent, in pixels."""
    declared = getattr(node, "width", None)
    if declared:
        return declared
    return _content_width(node)


def _content_width(node) -> int:
    """Return the horizontal extent a layout's children need, ignoring its own declared width."""
    # pn.WidgetBox is not a pn.Column subclass; ListPanel is the common base that
    # also covers Column, Row and Card.
    if not isinstance(node, pn.layout.ListPanel):
        return 0
    extents = [extent for extent in (_occupied_width(child) for child in node.objects) if extent]
    if not extents:
        return 0
    if isinstance(node, pn.Row):
        return sum(extents) + _SIBLING_MARGIN * len(extents)
    return max(extents)


def _label_width(name: str) -> int:
    """Approximate the rendered width of a widget label plus its help icon, in pixels."""
    return int(len(name) * 7.1) + 26


def _width_bearing_containers(node) -> list:
    """Return every layout at or below ``node`` that pins its own width."""
    if not isinstance(node, pn.layout.ListPanel):
        return []
    found = [node] if getattr(node, "width", None) else []
    for child in node.objects:
        found.extend(_width_bearing_containers(child))
    return found


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

    def test_compute_binned_metrics_default(self, parameter_form):
        assert parameter_form.computeBinnedMetrics.value is False

    def test_binned_metrics_width_default(self, parameter_form):
        assert parameter_form.binnedMetricsWidth.value == 120

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

    def test_photobleaching_detrend_default(self, parameter_form):
        assert parameter_form.photobleaching_detrend.value is False
        assert parameter_form.photobleaching_detrend.options == [True, False]

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

    def test_z_score_computation_default(self, parameter_form):
        assert parameter_form.z_score_computation.value == "standard z-score"
        assert "standard z-score" in parameter_form.z_score_computation.options
        assert "baseline z-score" in parameter_form.z_score_computation.options
        assert "modified z-score" in parameter_form.z_score_computation.options

    def test_compute_corr_default(self, parameter_form):
        assert parameter_form.computeCorr.value is False
        assert parameter_form.computeCorr.options == [True, False]

    def test_use_transients_as_events_default(self, parameter_form):
        assert parameter_form.useTransientsAsEvents.value is False
        assert parameter_form.useTransientsAsEvents.options == [True, False]

    def test_auc_units_default(self, parameter_form):
        assert parameter_form.auc_units.value == "samples"
        assert parameter_form.auc_units.options == ["samples", "seconds"]

    def test_use_time_or_trials_default(self, parameter_form):
        assert parameter_form.use_time_or_trials.value == "Time (min)"
        assert "Time (min)" in parameter_form.use_time_or_trials.options
        assert "# of trials" in parameter_form.use_time_or_trials.options

    def test_comparison_table_starts_with_a_single_blank_row(self, parameter_form):
        # A fixed block of slots is mostly blank rows for anyone running two comparisons.
        assert parameter_form.comparison_df_widget.value.shape == (1, 2)
        assert list(parameter_form.comparison_df_widget.value.columns) == ["Event A", "Event B"]

    def test_add_button_grows_the_comparison_table_without_limit(self, parameter_form):
        # The number of worthwhile pairs grows with the square of the event count, so six
        # events already allow fifteen -- more than any fixed table would hold.
        for _ in range(14):
            parameter_form._add_comparison_row()

        assert parameter_form.comparison_df_widget.value.shape == (15, 2)

    def test_removing_a_row_drops_that_comparison(self, parameter_form):
        parameter_form.comparison_df_widget.value = pd.DataFrame(
            {"Event A": ["a", "b", "c"], "Event B": ["x", "y", "z"]}
        )

        parameter_form._remove_comparison_row(SimpleNamespace(row=1))

        assert list(parameter_form.comparison_df_widget.value["Event A"]) == ["a", "c"]
        assert list(parameter_form.comparison_df_widget.value["Event B"]) == ["x", "z"]

    def test_removing_the_last_row_leaves_one_blank_row(self, parameter_form):
        parameter_form.comparison_df_widget.value = pd.DataFrame({"Event A": ["only"], "Event B": ["pair"]})

        parameter_form._remove_comparison_row(SimpleNamespace(row=0))

        assert parameter_form.comparison_df_widget.value.shape == (1, 2)
        assert list(parameter_form.comparison_df_widget.value["Event A"]) == [""]

    def test_loads_a_saved_run_holding_more_comparisons_than_the_table_shows(self, parameter_form):
        # Assigning a longer list into the table's existing index used to raise, so a run
        # driven through the API with many comparisons could not be reopened in the form.
        saved_a = [f"a{index}" for index in range(12)]
        saved_b = [f"b{index}" for index in range(12)]

        parameter_form.setInputParameters({"psthComparisonsA": saved_a, "psthComparisonsB": saved_b})

        assert list(parameter_form.comparison_df_widget.value["Event A"]) == saved_a
        assert list(parameter_form.comparison_df_widget.value["Event B"]) == saved_b

    def test_no_layout_overflows_its_declared_width(self, parameter_form):
        # Contents wider than their container overflow the panel visually, which no other
        # assertion here would catch. Sweeping every card rather than a hardcoded list of
        # boxes means a newly added parameter cannot slip past the check.
        for card_name in _CARD_ATTRIBUTES:
            for container in _width_bearing_containers(getattr(parameter_form, card_name)):
                occupied = _content_width(container)
                assert occupied <= container.width, (
                    f"{card_name}: {type(container).__name__} contents occupy {occupied}px "
                    f"inside a {container.width}px container"
                )

    def test_width_sweep_reaches_every_parameter_box(self, parameter_form):
        # Guards the sweep itself: a refactor that drops a box out of the card tree would
        # otherwise leave the overflow test passing because it found nothing to check.
        swept = {
            id(container)
            for card_name in _CARD_ATTRIBUTES
            for container in _width_bearing_containers(getattr(parameter_form, card_name))
        }
        for box_name in _SECTION_ATTRIBUTES:
            assert id(getattr(parameter_form, box_name)) in swept, f"{box_name} was not reached by the sweep"

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

    def test_get_input_parameters_abspath_is_the_shared_parent(self, parameter_form, tmp_path):
        session_a = tmp_path / "sessions" / "session_a"
        session_b = tmp_path / "sessions" / "session_b"
        session_a.mkdir(parents=True)
        session_b.mkdir(parents=True)
        parameter_form.files_1.value = [str(session_a), str(session_b)]

        result = parameter_form.getInputParameters()

        assert result["abspath"] == str(tmp_path / "sessions")
        assert result["session_folders"] == [str(session_a), str(session_b)]

    def test_get_input_parameters_accepts_sessions_from_different_parents(self, parameter_form, tmp_path):
        tdt_session = tmp_path / "tdt_data" / "session_a"
        csv_session = tmp_path / "csv_data" / "session_b"
        tdt_session.mkdir(parents=True)
        csv_session.mkdir(parents=True)
        parameter_form.files_1.value = [str(tdt_session), str(csv_session)]

        result = parameter_form.getInputParameters()

        assert result["abspath"] == str(tmp_path)
        assert result["session_folders"] == [str(tdt_session), str(csv_session)]

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

    def test_zero_binned_metrics_width_raises(self, parameter_form):
        parameter_form.binnedMetricsWidth.value = 0
        with pytest.raises(ValueError, match="binnedMetricsWidth=0 must be greater than 0"):
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


def _dandi_form_with_existing_runs(
    *, form, patched_dandi_client, output_root, asset_paths, run_names, output_root_folder
):
    """Drive a form into DANDI mode with assets whose mirrored session dirs already hold runs."""
    patched_dandi_client.dandisets_by_id = {"000971": _FakeDandiset(asset_paths)}
    output_root.mkdir()
    for asset_path in asset_paths:
        session = output_root / Path(asset_path).name.removesuffix(".nwb")
        session.mkdir()
        for run_name in run_names:
            Path(
                run_folder_for_run(
                    str(session),
                    run_name,
                    output_root_folder=str(output_root_folder),
                    input_root_folder=str(output_root),
                )
            ).mkdir(parents=True, exist_ok=True)

    form.source_mode.value = "dandi"
    form.dandi_selector.dandiset_input.value = "000971"
    mirror_root = form.dandi_selector._current_mirror_root
    form.dandi_selector.asset_file_selector.value = [
        str(Path(mirror_root).joinpath(*asset_path.split("/"))) for asset_path in asset_paths
    ]
    form.input_root_selector.value = [str(output_root)]
    return form


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
            str(Path(mirror_root) / "sub-01" / "session_a.nwb"),
            str(Path(mirror_root) / "sub-02" / "session_b.nwb"),
        ]
        form.input_root_selector.value = [str(output_root)]

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
            assert Path(session_dir).is_dir()

    def test_dandi_mode_no_asset_raises(self, bare_parameter_form, tmp_path):
        form = bare_parameter_form
        form.source_mode.value = "dandi"
        with pytest.raises(Exception, match="select at least one NWB asset"):
            form.getInputParameters()

    def test_dandi_mode_without_an_input_root_folder_raises(
        self, unconfigured_parameter_form, patched_dandi_client, output_root_folder
    ):
        patched_dandi_client.dandisets_by_id = {"000971": _FakeDandiset(["sub-01/data.nwb"])}
        form = unconfigured_parameter_form
        form.output_root_selector.value = [str(output_root_folder)]
        form.source_mode.value = "dandi"
        form.dandi_selector.dandiset_input.value = "000971"
        mirror_root = form.dandi_selector._current_mirror_root
        form.dandi_selector.asset_file_selector.value = [str(Path(mirror_root) / "sub-01" / "data.nwb")]
        with pytest.raises(Exception, match="no data root selected|input root folder"):
            form.getInputParameters()

    def test_dandi_asset_selection_offers_that_session_run_names(
        self, bare_parameter_form, tmp_path, patched_dandi_client, output_root_folder
    ):
        form = _dandi_form_with_existing_runs(
            form=bare_parameter_form,
            patched_dandi_client=patched_dandi_client,
            output_root=tmp_path / "dandi_output",
            asset_paths=["sub-01/session_a.nwb"],
            run_names=["1", "baseline"],
            output_root_folder=output_root_folder,
        )
        assert form.run_names_for_all_sessions.options == ["1", "baseline"]

    def test_dandi_run_name_choice_selects_the_mirrored_session_run(
        self, bare_parameter_form, tmp_path, patched_dandi_client, output_root_folder
    ):
        output_root = tmp_path / "dandi_output"
        form = _dandi_form_with_existing_runs(
            form=bare_parameter_form,
            patched_dandi_client=patched_dandi_client,
            output_root=output_root,
            asset_paths=["sub-01/session_a.nwb", "sub-02/session_b.nwb"],
            run_names=["1"],
            output_root_folder=output_root_folder,
        )

        form.run_names_for_all_sessions.value = ["1"]

        assert form._collect_selected_runs() == {
            str(output_root / "session_a"): ["1"],
            str(output_root / "session_b"): ["1"],
        }

    def test_dandi_outputs_selector_is_rooted_at_the_output_root(
        self, bare_parameter_form, tmp_path, patched_dandi_client, output_root_folder
    ):
        output_root = tmp_path / "dandi_output"
        form = _dandi_form_with_existing_runs(
            form=bare_parameter_form,
            patched_dandi_client=patched_dandi_client,
            output_root=output_root,
            asset_paths=["sub-01/session_a.nwb"],
            run_names=["1"],
            output_root_folder=output_root_folder,
        )
        assert form.outputs_selector.root_directory == str(output_root_folder)

    def test_switching_dandisets_drops_the_previous_run_selection(
        self, bare_parameter_form, tmp_path, patched_dandi_client, output_root_folder
    ):
        # The asset FileSelector is rebuilt on a dandiset change, which empties the asset
        # selection without firing a value event of its own.
        form = _dandi_form_with_existing_runs(
            form=bare_parameter_form,
            patched_dandi_client=patched_dandi_client,
            output_root=tmp_path / "dandi_output",
            asset_paths=["sub-01/session_a.nwb"],
            run_names=["1"],
            output_root_folder=output_root_folder,
        )
        form.run_names_for_all_sessions.value = ["1"]
        patched_dandi_client.dandisets_by_id["000972"] = _FakeDandiset(["sub-09/other.nwb"])

        form.dandi_selector.dandiset_input.value = "000972"

        assert form.run_names_for_all_sessions.options == []
        assert form._collect_selected_runs() == {}


@pytest.fixture
def sessions_with_runs(tmp_path, output_root_folder):
    """Build three sessions on disk: A has runs 1/baseline, B has 1/2, C has none.

    ``1`` is shared by A and B while ``baseline`` and ``2`` belong to one session each,
    so the union the run-name picker offers is distinguishable from an intersection.
    C exercises the pre-step-1 case. The run folders go where a form with ``tmp_path``
    as its data root mirrors them.
    """

    def build(name, run_names):
        session = tmp_path / name
        session.mkdir()
        for run_name in run_names:
            Path(
                run_folder_for_run(
                    str(session),
                    run_name,
                    output_root_folder=str(output_root_folder),
                    input_root_folder=str(tmp_path),
                )
            ).mkdir(parents=True)
        return str(session)

    return SimpleNamespace(
        output_root_folder=str(output_root_folder),
        input_root_folder=str(tmp_path),
        session_a=build("sessionA", ["1", "baseline"]),
        session_b=build("sessionB", ["1", "2"]),
        session_c=build("sessionC", []),
    )


class TestOutputsSelector:
    def test_outputs_selector_exists_and_is_filtered(self, parameter_form):
        assert isinstance(parameter_form.outputs_selector, pn.widgets.FileSelector)
        assert parameter_form.outputs_selector.file_pattern == "*output_*"

    def test_retarget_uses_the_output_root_folder(self, bare_parameter_form, tmp_path, output_root_folder):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        # root_directory must equal directory so Panel's startswith() validation in _dir_change
        # can't silently revert (especially on Windows where root_directory="/" resolves to a
        # potentially different drive than tmp_path).
        assert bare_parameter_form.outputs_selector.root_directory == str(output_root_folder)
        assert bare_parameter_form.outputs_selector.directory == str(output_root_folder)
        assert bare_parameter_form.outputs_selector.value == []

    def test_retarget_uses_the_input_root_when_the_roots_match(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.same_root_checkbox.value = True
        bare_parameter_form.files_1.value = [str(session)]
        assert bare_parameter_form.outputs_selector.root_directory == str(tmp_path)
        assert bare_parameter_form.outputs_selector.directory == str(tmp_path)

    def test_retarget_falls_back_to_the_output_root_when_files_1_cleared(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.files_1.value = []
        assert bare_parameter_form.outputs_selector.directory == str(output_root_folder)

    def test_retarget_multiple_sessions_shares_one_base_directory(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        bare_parameter_form.files_1.value = [str(session_a), str(session_b)]
        assert bare_parameter_form.outputs_selector.root_directory == str(output_root_folder)
        assert bare_parameter_form.outputs_selector.directory == str(output_root_folder)

    def test_collect_selected_outputs_groups_by_session(self, bare_parameter_form, tmp_path, output_root_folder):
        session_a = tmp_path / "sessionA"
        session_a.mkdir()
        session_b = tmp_path / "sessionB"
        session_b.mkdir()
        base = str(output_root_folder)
        root = str(tmp_path)
        run_a1 = run_folder_for_run(str(session_a), "run1", output_root_folder=base, input_root_folder=root)
        run_a2 = run_folder_for_run(str(session_a), "run2", output_root_folder=base, input_root_folder=root)
        run_b1 = run_folder_for_run(str(session_b), "run1", output_root_folder=base, input_root_folder=root)
        for path in (run_a1, run_a2, run_b1):
            Path(path).mkdir(parents=True, exist_ok=True)

        bare_parameter_form.files_1.value = [str(session_a), str(session_b)]
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
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        Path(
            run_folder_for_run(
                str(session),
                "baseline",
                output_root_folder=str(output_root_folder),
                input_root_folder=str(tmp_path),
            )
        ).mkdir(parents=True)

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

    def test_get_input_parameters_selected_outputs_reflects_selector(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_dir = run_folder_for_run(
            str(session), "baseline", output_root_folder=str(output_root_folder), input_root_folder=str(tmp_path)
        )
        Path(run_dir).mkdir(parents=True)

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        result = bare_parameter_form.getInputParameters()
        assert result["selected_runs"] == {str(session): ["baseline"]}


class TestRunNamePicker:
    def test_offers_every_run_name_any_selected_session_has(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        assert bare_parameter_form.run_names_for_all_sessions.options == ["1", "baseline", "2"]

    def test_session_without_run_folders_contributes_no_names(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_c]
        assert bare_parameter_form.run_names_for_all_sessions.options == ["1", "baseline"]

    def test_choosing_a_name_selects_that_run_in_every_session(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]
        assert bare_parameter_form._collect_selected_runs() == {
            sessions_with_runs.session_a: ["1"],
            sessions_with_runs.session_b: ["1"],
        }

    def test_name_only_one_session_has_selects_only_that_session(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["baseline"]
        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_a: ["baseline"]}

    def test_programmatic_selection_reaches_the_visible_pane(self, bare_parameter_form, sessions_with_runs):
        # FileSelector.value alone leaves the "Selected files" pane empty; the picker has to
        # re-enumerate the browser for a bulk choice to be visible to the user.
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]
        assert sorted(bare_parameter_form.outputs_selector._selector.value) == [
            run_folder_for_run(
                sessions_with_runs.session_a,
                "1",
                output_root_folder=sessions_with_runs.output_root_folder,
                input_root_folder=sessions_with_runs.input_root_folder,
            ),
            run_folder_for_run(
                sessions_with_runs.session_b,
                "1",
                output_root_folder=sessions_with_runs.output_root_folder,
                input_root_folder=sessions_with_runs.input_root_folder,
            ),
        ]

    def test_dropping_a_name_deselects_only_the_runs_it_named(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["1", "2"]

        bare_parameter_form.run_names_for_all_sessions.value = ["2"]

        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_b: ["2"]}

    def test_runs_picked_in_the_tree_survive_a_later_bulk_choice(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        hand_picked = run_folder_for_run(
            sessions_with_runs.session_a,
            "baseline",
            output_root_folder=sessions_with_runs.output_root_folder,
            input_root_folder=sessions_with_runs.input_root_folder,
        )
        bare_parameter_form.outputs_selector.value = [hand_picked]

        bare_parameter_form.run_names_for_all_sessions.value = ["1"]
        bare_parameter_form.run_names_for_all_sessions.value = []

        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_a: ["baseline"]}

    def test_removing_a_session_preserves_the_other_sessions_choices(self, bare_parameter_form, sessions_with_runs):
        # Regression for #462: dropping one session used to wipe every run choice.
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]

        bare_parameter_form.files_1.value = [sessions_with_runs.session_a]

        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_a: ["1"]}

    def test_newly_added_session_inherits_the_current_choice(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]

        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]

        assert bare_parameter_form._collect_selected_runs() == {
            sessions_with_runs.session_a: ["1"],
            sessions_with_runs.session_b: ["1"],
        }

    def test_name_gone_from_disk_leaves_the_picker_when_its_session_does(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a, sessions_with_runs.session_b]
        bare_parameter_form.run_names_for_all_sessions.value = ["1", "2"]

        bare_parameter_form.files_1.value = [sessions_with_runs.session_a]

        assert bare_parameter_form.run_names_for_all_sessions.options == ["1", "baseline"]
        assert bare_parameter_form.run_names_for_all_sessions.value == ["1"]

    def test_refresh_individual_outputs_offers_new_runs_and_keeps_the_selection(
        self, bare_parameter_form, sessions_with_runs
    ):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]
        Path(
            run_folder_for_run(
                sessions_with_runs.session_a,
                "2",
                output_root_folder=sessions_with_runs.output_root_folder,
                input_root_folder=sessions_with_runs.input_root_folder,
            )
        ).mkdir()

        bare_parameter_form.refresh_individual_outputs()

        assert bare_parameter_form.run_names_for_all_sessions.options == ["1", "2", "baseline"]
        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_a: ["1"]}

    def test_switching_source_mode_and_back_keeps_the_local_selection(self, bare_parameter_form, sessions_with_runs):
        bare_parameter_form.files_1.value = [sessions_with_runs.session_a]
        bare_parameter_form.run_names_for_all_sessions.value = ["1"]

        bare_parameter_form.source_mode.value = "dandi"
        assert bare_parameter_form._collect_selected_runs() == {}

        bare_parameter_form.source_mode.value = "local"
        assert bare_parameter_form.run_names_for_all_sessions.value == ["1"]
        assert bare_parameter_form._collect_selected_runs() == {sessions_with_runs.session_a: ["1"]}


class TestOutputBaseDirectory:
    def test_reports_the_chosen_directory(self, bare_parameter_form, tmp_path, output_root_folder):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]

        assert bare_parameter_form.output_root_folder == str(output_root_folder)
        assert bare_parameter_form.input_root_folder == str(tmp_path)

    def test_mirrors_the_session_path_under_the_chosen_directory(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "mouse1" / "day1"
        session.mkdir(parents=True)
        bare_parameter_form.files_1.value = [str(session)]

        assert run_directory_root(
            session_path=str(session),
            output_root_folder=bare_parameter_form.output_root_folder,
            input_root_folder=bare_parameter_form.input_root_folder,
        ) == str(output_root_folder / "mouse1" / "day1")

    def test_matching_the_roots_reports_the_input_root_as_the_output_root(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.same_root_checkbox.value = True

        assert bare_parameter_form.output_root_folder == str(tmp_path)
        assert bare_parameter_form.output_root_selector.visible is False

    def test_get_input_parameters_creates_each_sessions_mirrored_directory(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        sessions = []
        for parent in ("mouse1", "mouse2"):
            session = tmp_path / parent / "day1"
            session.mkdir(parents=True)
            sessions.append(str(session))
        bare_parameter_form.files_1.value = sessions

        result = bare_parameter_form.getInputParameters()

        assert result["output_root_folder"] == str(output_root_folder)
        assert result["input_root_folder"] == str(tmp_path)
        assert (output_root_folder / "mouse1" / "day1").is_dir()
        assert (output_root_folder / "mouse2" / "day1").is_dir()

    def test_matching_roots_write_the_runs_inside_the_session(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.same_root_checkbox.value = True

        result = bare_parameter_form.getInputParameters()

        assert result["output_root_folder"] == str(tmp_path)
        assert run_folder_for_run(
            str(session),
            "1",
            output_root_folder=result["output_root_folder"],
            input_root_folder=result["input_root_folder"],
        ) == str(session / "output_1")

    def test_get_input_parameters_requires_an_output_directory(self, unconfigured_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        unconfigured_parameter_form.input_root_selector.value = [str(tmp_path)]
        unconfigured_parameter_form.files_1.value = [str(session)]

        with pytest.raises(ValueError, match="No output root folder chosen"):
            unconfigured_parameter_form.getInputParameters()

    def test_get_input_parameters_requires_a_input_root_folder(
        self, unconfigured_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        unconfigured_parameter_form.output_root_selector.value = [str(output_root_folder)]
        unconfigured_parameter_form.files_1.value = [str(session)]

        with pytest.raises(ValueError, match="No input root folder chosen"):
            unconfigured_parameter_form.getInputParameters()

    def test_sessions_sharing_a_folder_name_keep_separate_output_directories(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        sessions = []
        for parent in ("mouse1", "mouse2"):
            session = tmp_path / parent / "day1"
            session.mkdir(parents=True)
            sessions.append(str(session))
        bare_parameter_form.files_1.value = sessions

        assert bare_parameter_form.getInputParameters()["session_folders"] == sessions
        assert (output_root_folder / "mouse1" / "day1").is_dir()
        assert (output_root_folder / "mouse2" / "day1").is_dir()

    def test_a_session_outside_the_input_root_folder_is_rejected(self, bare_parameter_form, tmp_path):
        outside = tmp_path.parent / "outside_session"
        outside.mkdir(exist_ok=True)
        bare_parameter_form.files_1.value = [str(outside)]

        with pytest.raises(ValueError, match="not inside the input root folder"):
            bare_parameter_form.getInputParameters()

    def test_sessions_sharing_a_folder_name_are_allowed_when_the_roots_match(self, bare_parameter_form, tmp_path):
        sessions = []
        for parent in ("mouse1", "mouse2"):
            session = tmp_path / parent / "day1"
            session.mkdir(parents=True)
            sessions.append(str(session))
        bare_parameter_form.files_1.value = sessions
        bare_parameter_form.same_root_checkbox.value = True

        assert bare_parameter_form.getInputParameters()["session_folders"] == sessions

    def test_runs_are_found_where_the_chosen_base_directory_holds_them(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_folder = run_folder_for_run(
            str(session), "baseline", output_root_folder=str(output_root_folder), input_root_folder=str(tmp_path)
        )
        Path(run_folder).mkdir(parents=True)

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.run_names_for_all_sessions.value = ["baseline"]

        assert bare_parameter_form._collect_selected_runs() == {str(session): ["baseline"]}

    def test_matching_the_roots_re_points_the_run_selection(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        (session / "output_inside").mkdir()
        bare_parameter_form.files_1.value = [str(session)]
        assert bare_parameter_form.run_names_for_all_sessions.options == []

        bare_parameter_form.same_root_checkbox.value = True

        assert bare_parameter_form.run_names_for_all_sessions.options == ["inside"]


class TestRootFolderSelection:
    def test_the_card_leads_the_page(self, parameter_form):
        assert parameter_form.root_folder_selection.title == "Root Folder Selection"
        assert parameter_form.template.main[0] is parameter_form.root_folder_selection

    def test_the_card_opens_itself_while_a_root_is_missing(self, unconfigured_parameter_form):
        assert unconfigured_parameter_form.root_folder_selection.collapsed is False

    def test_the_card_folds_away_when_the_roots_are_known_at_launch(
        self, panel_extension, frontend_base_dir, tmp_path, output_root_folder
    ):
        form = ParameterForm(
            template=pn.template.BootstrapTemplate(title="Test"),
            start_path=str(frontend_base_dir),
            input_root_folder=str(tmp_path),
            output_root_folder=str(output_root_folder),
        )

        assert form.root_folder_selection.collapsed is True

    def test_the_roots_named_on_the_command_line_show_in_the_browsers(
        self, panel_extension, frontend_base_dir, tmp_path, output_root_folder
    ):
        """Setting ``value`` alone leaves the browser drawn where it was, so the choice
        would not appear until the user made the widget re-list its directory."""
        form = ParameterForm(
            template=pn.template.BootstrapTemplate(title="Test"),
            start_path=str(frontend_base_dir),
            input_root_folder=str(tmp_path),
            output_root_folder=str(output_root_folder),
        )

        assert form.input_root_selector._selector.value == [str(tmp_path)]
        assert form.output_root_selector._selector.value == [str(output_root_folder)]

    def test_choosing_the_roots_by_hand_leaves_the_card_open(self, bare_parameter_form):
        """Folding the card away under the user's cursor would be jarring; it settles at launch."""
        assert bare_parameter_form.root_folder_selection.collapsed is False

    def test_matching_the_roots_hides_the_output_browser(self, bare_parameter_form, tmp_path):
        assert bare_parameter_form.output_root_selector.visible is True

        bare_parameter_form.same_root_checkbox.value = True

        assert bare_parameter_form.output_root_selector.visible is False
        assert bare_parameter_form.output_root_folder == str(tmp_path)

    def test_clearing_the_checkbox_brings_the_chosen_output_root_back(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        bare_parameter_form.same_root_checkbox.value = True
        bare_parameter_form.same_root_checkbox.value = False

        assert bare_parameter_form.output_root_selector.visible is True
        assert bare_parameter_form.output_root_folder == str(output_root_folder)

    def test_the_session_browser_follows_the_input_root(self, bare_parameter_form, tmp_path):
        """Moving the browser's directory alone leaves its listing where it was."""
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "session_a").mkdir(parents=True)

        bare_parameter_form.input_root_selector.value = [str(elsewhere)]

        assert bare_parameter_form.files_1.directory == str(elsewhere)
        assert bare_parameter_form.files_1._cwd == str(elsewhere)
        assert any("session_a" in str(path) for path in bare_parameter_form.files_1._selector.options.values())


class TestLegacyRunFolders:
    def test_a_pre_beta4_run_is_offered_alongside_a_current_one(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.same_root_checkbox.value = True
        (session / "sessionA_output_old").mkdir()
        (session / "output_new").mkdir()

        bare_parameter_form.files_1.value = [str(session)]

        assert sorted(bare_parameter_form.run_names_for_all_sessions.options) == ["new", "old"]

    def test_finding_one_warns_that_the_name_is_deprecated(self, bare_parameter_form, tmp_path, notifications):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.same_root_checkbox.value = True
        (session / "sessionA_output_old").mkdir()
        bare_parameter_form.files_1.value = [str(session)]

        bare_parameter_form._warn_about_legacy_run_folders()

        assert len(notifications) == 1
        level, message, kwargs = notifications[0]
        assert level == "warning"
        assert "sessionA_output_old" in message
        assert "will be removed in a future release" in message
        # Named by folder, not by full path, and short enough to read at a glance.
        assert str(session) not in message
        # Persistent, or a step running afterwards would clear it before it was read.
        assert kwargs == {"duration": 0}

    def test_a_run_named_only_the_current_way_warns_about_nothing(self, bare_parameter_form, tmp_path, notifications):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.same_root_checkbox.value = True
        (session / "output_1").mkdir()
        bare_parameter_form.files_1.value = [str(session)]

        bare_parameter_form._warn_about_legacy_run_folders()

        assert notifications == []


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

    def test_add_to_template_appends_the_roots_first_then_input_and_output(self, parameter_form):
        main = parameter_form.template.main
        # The roots lead: they are the project constants the per-run choices sit under.
        assert main[0] is parameter_form.root_folder_selection
        assert main[1] is parameter_form.input_folder_selection
        assert main[2] is parameter_form.output_folder_selection
        assert main[3] is parameter_form.individual
        assert main[4] is parameter_form.group
        assert len(main) == 5


class TestTitledBox:
    def test_heading_carries_the_title(self, panel_extension):
        box = _titled_box(title="Signal Filtering", read_by="Step 3", contents=[], width=960)

        assert box.objects[0].object == "### Signal Filtering"

    def test_second_pane_names_the_consuming_steps(self, panel_extension):
        box = _titled_box(title="Signal Filtering", read_by="Step 3 and Group Analysis", contents=[], width=960)

        assert box.objects[1].object == "*Read by Step 3 and Group Analysis*"

    def test_contents_follow_the_heading(self, panel_extension):
        widget = pn.widgets.IntInput(name="Cores", value=2, width=150)

        box = _titled_box(title="Parallel Execution", read_by="Step 2", contents=[widget], width=960)

        assert box.objects[2] is widget

    def test_panes_sit_inside_the_declared_width(self, panel_extension):
        box = _titled_box(title="Metric Binning", read_by="Step 4", contents=[], width=960)

        assert box.width == 960
        assert [pane.width for pane in box.objects] == [920, 920]


class TestTableHeading:
    def test_label_is_rendered_bold(self, panel_extension):
        heading = _table_heading(label="Event comparisons", description="Pairs to compare.", width=200)

        assert heading.objects[0].object == "**Event comparisons**"

    def test_help_icon_carries_the_description(self, panel_extension):
        heading = _table_heading(label="Event comparisons", description="Pairs to compare.", width=200)

        assert isinstance(heading.objects[1], pn.widgets.TooltipIcon)
        assert heading.objects[1].value == "Pairs to compare."


class TestParameterHelp:
    """Every control explains itself, since the form carries no prose of its own."""

    def test_every_parameter_widget_has_a_description(self, parameter_form):
        undocumented = [
            widget.name
            for section in parameter_form.individual_parameters.objects
            for item in section
            for widget in (list(item) if isinstance(item, pn.Row) else [item])
            if isinstance(widget, pn.widgets.Widget)
            and not isinstance(widget, (pn.widgets.Tabulator, pn.widgets.Button, pn.widgets.TooltipIcon))
            and not widget.description
        ]

        assert undocumented == []

    def test_combine_data_explains_itself_in_the_input_card(self, parameter_form):
        assert "two separate data files" in parameter_form.combine_data.description

    def test_each_label_fits_beside_its_help_icon(self, parameter_form):
        # Panel renders the description as an icon after the label, so a widget narrower
        # than its own label pushes the icon over the control beside it.
        crowded = [
            (widget.name, widget.width)
            for section in parameter_form.individual_parameters.objects
            for item in section
            for widget in (list(item) if isinstance(item, pn.Row) else [item])
            if isinstance(widget, pn.widgets.Widget)
            and getattr(widget, "description", None)
            and widget.width
            and _label_width(widget.name) > widget.width
        ]

        assert crowded == []


class TestParameterSections:
    def test_sections_appear_in_pipeline_order(self, parameter_form):
        titles = [section.objects[0].object for section in parameter_form.individual_parameters.objects]

        assert titles == [
            "### Parallel Execution",
            "### Control Channel Fitting",
            "### Signal Filtering",
            "### Z-score Normalization",
            "### PSTH Computation",
            "### Peak and AUC Measurement",
            "### Transient Detection",
            "### Metric Binning",
            "### Significance Testing",
        ]

    def test_the_card_does_not_claim_a_single_analysis_level(self, parameter_form):
        assert parameter_form.individual.title == "Parameter Selection"

    def test_the_shared_transient_controls_sit_with_the_detector(self, parameter_form):
        section = parameter_form.transients_param_wd
        widgets = [widget for item in section for widget in (list(item) if isinstance(item, pn.Row) else [item])]

        assert parameter_form.transients in widgets
        assert parameter_form.useTransientsAsEvents in widgets

    def test_the_psth_metric_sits_with_the_psth_window(self, parameter_form):
        section = parameter_form.psth_param_wd
        widgets = [widget for item in section for widget in (list(item) if isinstance(item, pn.Row) else [item])]

        assert parameter_form.computePsth in widgets
        assert parameter_form.baselineCorrectionStart in widgets


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
    "photobleaching_detrend": True,
    "timeForLightsTurnOn": 7,
    "filter_window": 42,
    "zscore_method": "modified z-score",
    "baselineWindowStart": 2,
    "baselineWindowEnd": 9,
    "nSecPrev": -3,
    "nSecPost": 8,
    "computeCorr": True,
    "useTransientsAsEvents": True,
    "timeInterval": 5,
    "bin_psth_trials": 4,
    "use_time_or_trials": "# of trials",
    "baselineCorrectionStart": -2,
    "baselineCorrectionEnd": 1,
    "peak_startPoint": [-4.0, 1.0, 6.0] + [float("nan")] * 7,
    "peak_endPoint": [1.0, 4.0, 11.0] + [float("nan")] * 7,
    "auc_units": "seconds",
    "selectForComputePsth": "dff",
    "selectForTransientsComputation": "Both",
    "moving_window": 12,
    "highAmpFilt": 5,
    "transientsThresh": 6,
    "computeBinnedMetrics": True,
    "computePsthSignificance": True,
    "psthSignificanceAlpha": 0.01,
    "psthBootstrapResamples": 500,
    "binnedMetricsWidth": 60,
}


def _write_run_with_parameters(session_dir, run_name, parameters, *, output_root_folder, input_root_folder):
    """Create a run directory for session_dir holding a GuPPyParamtersUsed.json."""
    run_dir = run_folder_for_run(
        str(session_dir), run_name, output_root_folder=str(output_root_folder), input_root_folder=str(input_root_folder)
    )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / "GuPPyParamtersUsed.json").open("w") as parameters_file:
        json.dump(parameters, parameters_file)
    return run_dir


class TestParameterAutoPopulate:
    def test_set_input_parameters_round_trips_get_input_parameters(self, parameter_form):
        parameter_form.setInputParameters(SAVED_PARAMETERS)
        result = parameter_form.getInputParameters()
        for key in parameter_form._scalar_parameter_widgets():
            assert result[key] == SAVED_PARAMETERS[key], f"{key} did not round-trip"
        # NaN tail entries compare equal only via isnan.
        np.testing.assert_array_equal(result["peak_startPoint"], SAVED_PARAMETERS["peak_startPoint"])
        np.testing.assert_array_equal(result["peak_endPoint"], SAVED_PARAMETERS["peak_endPoint"])

    def test_set_input_parameters_ignores_unknown_keys(self, parameter_form):
        # guppy_version has no backing widget and must be ignored without error.
        parameter_form.setInputParameters({"guppy_version": "x", "nSecPost": 99})
        assert parameter_form.nSecPost.value == 99

    def test_set_input_parameters_ignores_retired_artifact_keys(self, parameter_form):
        """Snapshots still record the artifact keys as provenance, but the form has no widgets for them."""
        parameter_form.setInputParameters(
            {"removeArtifacts": True, "artifactsRemovalMethod": "concatenate", "nSecPost": 99}
        )
        assert parameter_form.nSecPost.value == 99
        assert "removeArtifacts" not in parameter_form.getInputParameters()

    def test_selecting_output_run_populates_widgets(self, bare_parameter_form, tmp_path, output_root_folder):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_dir = _write_run_with_parameters(
            session, "baseline", SAVED_PARAMETERS, output_root_folder=output_root_folder, input_root_folder=tmp_path
        )

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        assert bare_parameter_form.timeForLightsTurnOn.value == 7
        assert bare_parameter_form.z_score_computation.value == "modified z-score"
        assert bare_parameter_form.combine_data.value is True
        assert bare_parameter_form.nSecPrev.value == -3
        assert list(bare_parameter_form.df_widget.value["Peak Start time"])[:3] == [-4.0, 1.0, 6.0]

    def test_agreeing_runs_populate_widgets(self, bare_parameter_form, tmp_path, output_root_folder):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_a = _write_run_with_parameters(
            session, "run_a", SAVED_PARAMETERS, output_root_folder=output_root_folder, input_root_folder=tmp_path
        )
        run_b = _write_run_with_parameters(
            session, "run_b", SAVED_PARAMETERS, output_root_folder=output_root_folder, input_root_folder=tmp_path
        )

        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_a, run_b]

        assert bare_parameter_form.timeForLightsTurnOn.value == 7

    def test_conflicting_runs_leave_form_unchanged(self, bare_parameter_form, tmp_path, output_root_folder):
        session = tmp_path / "sessionA"
        session.mkdir()
        run_a = _write_run_with_parameters(
            session, "run_a", SAVED_PARAMETERS, output_root_folder=output_root_folder, input_root_folder=tmp_path
        )
        conflicting = {**SAVED_PARAMETERS, "timeForLightsTurnOn": 99}
        run_b = _write_run_with_parameters(
            session, "run_b", conflicting, output_root_folder=output_root_folder, input_root_folder=tmp_path
        )

        default_time = bare_parameter_form.timeForLightsTurnOn.value
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_a, run_b]

        # Conflicting snapshots: form must be left untouched for the user to reconcile.
        assert bare_parameter_form.timeForLightsTurnOn.value == default_time

    def test_selecting_run_without_json_is_noop(self, bare_parameter_form, tmp_path):
        session = tmp_path / "sessionA"
        session.mkdir()
        bare_parameter_form.same_root_checkbox.value = True
        run_dir = run_folder_for_run(
            str(session), "fresh", output_root_folder=str(tmp_path), input_root_folder=str(tmp_path)
        )
        Path(run_dir).mkdir()

        default_time = bare_parameter_form.timeForLightsTurnOn.value
        bare_parameter_form.files_1.value = [str(session)]
        bare_parameter_form.outputs_selector.value = [run_dir]

        assert bare_parameter_form.timeForLightsTurnOn.value == default_time

    def test_choosing_a_run_name_populates_widgets_from_every_session_it_selects(
        self, bare_parameter_form, tmp_path, output_root_folder
    ):
        sessions = []
        for name in ("sessionA", "sessionB", "sessionC"):
            session = tmp_path / name
            session.mkdir()
            _write_run_with_parameters(
                session, "shared", SAVED_PARAMETERS, output_root_folder=output_root_folder, input_root_folder=tmp_path
            )
            sessions.append(str(session))
        bare_parameter_form.files_1.value = sessions

        bare_parameter_form.run_names_for_all_sessions.value = ["shared"]

        assert bare_parameter_form.timeForLightsTurnOn.value == 7
