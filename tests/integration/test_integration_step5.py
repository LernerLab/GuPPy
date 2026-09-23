import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import pandas as pd
import panel as pn
import pytest

from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_SUBDIR = "csv/sample_data_csv_1"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}


def metric_selectors(tab):
    """Return the "Metric" selectors rendered inside one dashboard tab."""
    return [widget for widget in tab.select(pn.widgets.Select) if widget.name == "Metric"]


@pytest.mark.parametrize(
    "step5_fixture_name, expected_event_substring",
    [
        ("step5_output_csv", "ttl"),
        ("step5_output_tdt", "port_entries"),
        ("step5_output_npm", "ttl"),
        ("step5_output_doric", "ttl"),
        ("step5_output_nwb", "ttl"),
        ("step5_output_pyphotometry", "ttl"),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_5", "sample_doric_1", "nwb_mock", "pyphotometry_pulsed"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5(step5_fixture_name, expected_event_substring, request):
    """
    Validate Step 5 (visualizeResults) for representative integration sessions.

    Patches VisualizationDashboard.show to prevent a web server from starting,
    then asserts that the orchestration logic correctly loaded step-4 output data
    and constructed valid ParameterizedPlotter instances.
    """
    pipeline_state = request.getfixturevalue(step5_fixture_name)
    captured_dashboards: list[VisualizationDashboard] = pipeline_state["captured_dashboards"]

    assert len(captured_dashboards) >= 1, "visualizeResults created no VisualizationDashboard instances"

    for dashboard in captured_dashboards:
        assert isinstance(dashboard, VisualizationDashboard)
        assert isinstance(dashboard.plotter, ParameterizedPlotter)

        event_selector_objects = dashboard.plotter.event_selector_objects
        assert isinstance(event_selector_objects, list)
        assert (
            len(event_selector_objects) >= 1
        ), "ParameterizedPlotter has no event_selector_objects — data was not loaded"

        columns_dict = dashboard.plotter.columns_dict
        assert isinstance(columns_dict, dict)
        assert len(columns_dict) >= 1, "ParameterizedPlotter columns_dict is empty — data was not loaded"

        dataframe = dashboard.plotter.df_new
        assert isinstance(dataframe, pd.DataFrame)
        assert not dataframe.empty, "ParameterizedPlotter df_new is empty — PSTH data was not read"

        # The fixtures run step 4 with its default metric, so the z-score is the only
        # one on disk: it is what the plotter loaded and what labels the y axis.
        assert dashboard.available_metrics == ["z_score"]
        assert dashboard.plotter.Y_Label == "z-score"

    # Confirm at least one dashboard has an event matching the expected TTL store_id
    all_events = [event for dashboard in captured_dashboards for event in dashboard.plotter.event_selector_objects]
    matching_events = [event for event in all_events if expected_event_substring in event]
    assert matching_events, f"No event containing '{expected_event_substring}' found among loaded events: {all_events}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5_offers_only_the_metric_step4_computed(tmp_path):
    """
    When step 4 computed only one metric, the dashboard's metric selector offers that
    metric alone, rather than the step failing up front.
    """
    source_session = STUBBED_TESTING_DATA / SESSION_SUBDIR
    assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path / "data_root"
    temporary_base_directory.mkdir(parents=True, exist_ok=True)
    session_name = source_session.name
    session_copy = temporary_base_directory / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in list(Path(session_copy).glob(f"{session_name}_output_*")):
        assert Path(output_directory).is_dir()
        shutil.rmtree(output_directory)
    parameters_path = session_copy / "GuPPyParamtersUsed.json"
    if parameters_path.exists():
        parameters_path.unlink()

    common_kwargs = dict(
        base_dir=str(temporary_base_directory),
        selected_folders=[str(session_copy)],
    )
    selected_runs = {str(session_copy): ["1"]}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    # Step 4: compute only dff (not z_score)
    step4(
        **common_kwargs,
        select_for_compute_psth="dff",
        select_for_transients="dff",
        selected_runs=selected_runs,
    )

    hv.extension("bokeh")

    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, **kwargs):
        original_init(self, **kwargs)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step5(**common_kwargs, selected_runs=selected_runs)

    assert len(captured_dashboards) == 1
    dashboard = captured_dashboards[0]
    assert dashboard.available_metrics == ["dff"]
    assert dashboard.plotter.Y_Label == "\u0394F/F"

    for tab in (dashboard._psth_tab, dashboard._heatmap_tab):
        (selector,) = metric_selectors(tab)
        assert selector.options == {"\u0394F/F": "dff"}
        assert selector.value == "dff"
        assert selector.disabled is False
