import socket
import time

import numpy as np
import pandas as pd
import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.frontend.parameterized_plotter import ParameterizedPlotter
from guppy.frontend.visualization_dashboard import VisualizationDashboard

_EVENTS = ["event1", "event2"]
_COLUMNS = ["trial_1", "trial_2", "trial_3", "bin_1", "timestamps", "mean", "err", "bin_err_1"]
_N_TIMEPOINTS = 30
_TIMESTAMPS = np.linspace(-5.0, 10.0, _N_TIMEPOINTS)


@pytest.fixture(scope="session")
def visualization_server_url(panel_extension, tmp_path_factory):
    """Serve the Visualization Dashboard on a free local port.

    Constructs a minimal ParameterizedPlotter with synthetic zero-valued data
    (the same pattern used in tests/unit/frontend/conftest.py) and yields
    the base URL of the running server.
    """
    temporary_path = tmp_path_factory.mktemp("visualization_plots")

    def make_event_dataframe():
        return pd.DataFrame(
            {column: (_TIMESTAMPS if column == "timestamps" else np.zeros(_N_TIMEPOINTS)) for column in _COLUMNS}
        )

    columns_dict = {event: _COLUMNS for event in _EVENTS}
    df_new = pd.concat([make_event_dataframe() for event in _EVENTS], keys=_EVENTS, axis=1)

    plotter = ParameterizedPlotter(
        event_selector_objects=_EVENTS,
        event_selector_heatmap_objects=_EVENTS,
        selector_for_multipe_events_plot_objects=_EVENTS,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["trial_1", "mean"],
        heatmap_y_objects=["1 - trial_1", "2 - trial_2", "All"],
        psth_y_objects=None,
        filepath=str(temporary_path),
        columns_dict=columns_dict,
        df_new=df_new,
        x_min=-5.0,
        x_max=10.0,
    )
    dashboard = VisualizationDashboard(plotter=plotter, basename="test_session")
    template = dashboard.build_template()

    port = scanPortsAndFind()
    pn.serve(template, port=port, show=False, threaded=True)

    for _ in range(50):
        try:
            connection = socket.create_connection(("localhost", port), timeout=0.1)
            connection.close()
            break
        except OSError:
            time.sleep(0.05)

    yield f"http://localhost:{port}"

    pn.state.kill_all_servers()


@pytest.mark.ui
def test_page_title_contains_visualization_gui(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("PSTH").first).to_be_visible()
    assert "Visualization GUI" in page.title()


@pytest.mark.ui
def test_psth_tab_visible(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("PSTH").first).to_be_visible()


@pytest.mark.ui
def test_heat_map_tab_visible(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("Heat Map").first).to_be_visible()


@pytest.mark.ui
def test_trials_checkbox_options_visible(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("just trials").first).to_be_visible()


@pytest.mark.ui
def test_trial_timestamps_multiselect_visible(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("Trial # - Timestamps").first).to_be_visible()


@pytest.mark.ui
def test_save_psth_button_visible(page, visualization_server_url):
    page.goto(visualization_server_url)
    expect(page.get_by_text("PSTH").first).to_be_visible()
    expect(page.get_by_role("button", name="Save").first).to_be_visible()
