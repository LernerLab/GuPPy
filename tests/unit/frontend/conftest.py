import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.frontend.parameterized_plotter import ParameterizedPlotter


@pytest.fixture(scope="session")
def panel_extension():
    """Call pn.extension() exactly once for the entire test session.

    Panel requires this before any widget instantiation.
    """
    pn.extension()


@pytest.fixture(scope="session")
def plotter(tmp_path_factory, panel_extension):
    """Minimal _TestablePlotter for testing construction and watcher methods.

    Two events, each with 8 columns ordered so that columns[-4] == "timestamps":
      [trial_1, trial_2, trial_3, bin_1, timestamps, mean, err, bin_err_1]

    After remove_cols: [trial_1, trial_2, trial_3, bin_1, mean]
    remove_cols[:-2]  = [trial_1, trial_2, trial_3]   (for psth_y / heatmap_y labeling)
    x_value[-4]       = "timestamps"                  (for _update_x_y)
    y = remove_cols[-2] = "bin_1"                     (set by _update_x_y watcher on init)
    """
    columns = ["trial_1", "trial_2", "trial_3", "bin_1", "timestamps", "mean", "err", "bin_err_1"]
    n_timepoints = 30
    timestamps = np.linspace(-5.0, 10.0, n_timepoints)

    def make_event_dataframe():
        return pd.DataFrame(
            {column: (timestamps if column == "timestamps" else np.zeros(n_timepoints)) for column in columns}
        )

    events = ["event1", "event2"]
    columns_dict = {event: columns for event in events}
    df_new = {event: make_event_dataframe() for event in events}
    filepath = str(tmp_path_factory.mktemp("plotter"))

    return ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["trial_1", "mean"],
        heatmap_y_objects=["1 - trial_1", "2 - trial_2", "All"],
        psth_y_objects=None,
        filepath=filepath,
        columns_dict=columns_dict,
        df_new=df_new,
        x_min=-5.0,
        x_max=10.0,
    )
