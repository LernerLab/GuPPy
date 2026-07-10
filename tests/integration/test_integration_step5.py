import glob
import os
import shutil
from unittest.mock import patch

import holoviews as hv
import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5

SESSION_SUBDIR = "csv/sample_data_csv_1"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}


@pytest.mark.parametrize(
    "step5_fixture_name, expected_event_substring",
    [
        ("step5_output_csv", "ttl"),
        ("step5_output_tdt", "port_entries"),
        ("step5_output_npm", "ttl"),
        ("step5_output_doric", "ttl"),
        ("step5_output_nwb", "ttl"),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
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

    # Confirm at least one dashboard has an event matching the expected TTL store_id
    all_events = [event for dashboard in captured_dashboards for event in dashboard.plotter.event_selector_objects]
    matching_events = [event for event in all_events if expected_event_substring in event]
    assert matching_events, f"No event containing '{expected_event_substring}' found among loaded events: {all_events}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step5_raises_when_visualization_metric_not_computed_in_step4(tmp_path):
    """
    Step 5 must raise a ValueError with an actionable message when the requested
    visualization metric ('z_score') was not computed in step 4 (which only
    produced 'dff' outputs).

    The error message should:
    - Name the missing metric.
    - Name the session(s) it is missing in.
    - Tell the user to either change the visualization selection or re-run step 4.
    """
    source_session = STUBBED_TESTING_DATA / SESSION_SUBDIR
    assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path / "data_root"
    temporary_base_directory.mkdir(parents=True, exist_ok=True)
    session_name = source_session.name
    session_copy = temporary_base_directory / session_name
    shutil.copytree(source_session, session_copy)

    for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
        assert os.path.isdir(output_directory)
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

    # Step 5: request z_score visualization — must raise an actionable ValueError
    with patch.object(VisualizationDashboard, "show", lambda self: None):
        with pytest.raises(ValueError) as exc_info:
            step5(
                **common_kwargs,
                visualize_zscore_or_dff="z_score",
                selected_runs=selected_runs,
            )

    message = str(exc_info.value)
    assert "z_score" in message, f"Error message should mention the missing metric. Got: {message}"
    assert (
        str(session_copy) in message or session_name in message
    ), f"Error message should name the session. Got: {message}"
    assert "dff" in message, f"Error message should suggest the available alternative. Got: {message}"
    assert "step 4" in message.lower(), f"Error message should mention step 4. Got: {message}"
