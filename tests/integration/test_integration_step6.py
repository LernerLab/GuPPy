import pandas as pd
import pytest

from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.visualization_dashboard import VisualizationDashboard


@pytest.mark.parametrize(
    "step6_fixture_name, expected_event_substring",
    [
        ("step6_output_csv", "ttl"),
        ("step6_output_tdt", "port_entries"),
        ("step6_output_npm", "ttl"),
        ("step6_output_doric", "ttl"),
        ("step6_output_nwb", "ttl"),
    ],
    ids=["csv_generic", "tdt_clean", "sample_npm_1", "sample_doric_1", "nwb_mock"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_step6(step6_fixture_name, expected_event_substring, request):
    """
    Validate Step 6 (visualizeResults) for representative integration sessions.

    Patches VisualizationDashboard.show to prevent a web server from starting,
    then asserts that the orchestration logic correctly loaded step-5 output data
    and constructed valid ParameterizedPlotter instances.
    """
    pipeline_state = request.getfixturevalue(step6_fixture_name)
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

    # Confirm at least one dashboard has an event matching the expected TTL storename
    all_events = [event for dashboard in captured_dashboards for event in dashboard.plotter.event_selector_objects]
    matching_events = [event for event in all_events if expected_event_substring in event]
    assert matching_events, f"No event containing '{expected_event_substring}' found among loaded events: {all_events}"
