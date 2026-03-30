from guppy.frontend.parameterized_plotter import (
    ParameterizedPlotter,
    make_dir,
    remove_cols,
)

# ---------------------------------------------------------------------------
# remove_cols utility
# ---------------------------------------------------------------------------


def test_remove_cols_removes_err_timestamps_and_bin_err():
    columns = ["trial_1", "trial_2", "timestamps", "mean", "err", "bin_1", "bin_err_trial_1"]
    result = remove_cols(columns)
    assert "err" not in result
    assert "timestamps" not in result
    assert "bin_err_trial_1" not in result


def test_remove_cols_keeps_trial_mean_and_bin_columns():
    columns = ["trial_1", "trial_2", "timestamps", "mean", "err", "bin_1", "bin_err_trial_1"]
    result = remove_cols(columns)
    assert result == ["trial_1", "trial_2", "mean", "bin_1"]


# ---------------------------------------------------------------------------
# make_dir utility
# ---------------------------------------------------------------------------


def test_make_dir_creates_saved_plots_directory(tmp_path):
    make_dir(str(tmp_path))
    assert (tmp_path / "saved_plots").is_dir()


def test_make_dir_returns_correct_path(tmp_path):
    result = make_dir(str(tmp_path))
    assert result == str(tmp_path / "saved_plots")


def test_make_dir_is_idempotent(tmp_path):
    make_dir(str(tmp_path))
    # Second call must not raise even though directory already exists
    result = make_dir(str(tmp_path))
    assert result == str(tmp_path / "saved_plots")


# ---------------------------------------------------------------------------
# ParameterizedPlotter
# ---------------------------------------------------------------------------


class TestParameterizedPlotter:
    def test_constructs(self, plotter):
        assert isinstance(plotter, ParameterizedPlotter)

    def test_event_selector_set_to_first_event(self, plotter):
        assert plotter.event_selector == "event1"

    def test_event_selector_heatmap_set_to_first_event(self, plotter):
        assert plotter.event_selector_heatmap == "event1"

    def test_selector_for_multiple_events_set_to_first_event(self, plotter):
        assert plotter.selector_for_multipe_events_plot == ["event1"]

    def test_x_set_to_timestamps(self, plotter):
        # x_objects[0] == "timestamps"
        assert plotter.x == "timestamps"

    def test_y_set_to_second_to_last_y_object(self, plotter):
        # __init__ sets self.y = self.y_objects[-2] explicitly after the watcher fires.
        # y_objects == ["trial_1", "mean"] so y_objects[-2] == "trial_1".
        assert plotter.y == "trial_1"

    def test_heatmap_y_set_to_last_heatmap_y_object(self, plotter):
        # heatmap_y_objects[-1] == "All"
        assert plotter.heatmap_y == ["All"]

    def test_x_limit_bounds_set_from_x_min_x_max(self, plotter):
        assert plotter.param.X_Limit.bounds == (-5.0, 10.0)

    def test_default_select_trials_checkbox(self, plotter):
        assert plotter.select_trials_checkbox == ["just trials"]

    def test_default_y_label(self, plotter):
        assert plotter.Y_Label == "y"

    def test_default_save_options(self, plotter):
        assert plotter.save_options == "None"

    def test_default_height_plot(self, plotter):
        assert plotter.Height_Plot == 300

    def test_default_width_plot(self, plotter):
        assert plotter.Width_Plot == 1000

    def test_update_x_y_fires_on_event_selector_change(self, plotter):
        # Changing event_selector triggers _update_x_y watcher.
        # columns[-4] == "timestamps"; remove_cols(columns)[-2] == "bin_1"
        plotter.event_selector = "event2"
        assert plotter.x == "timestamps"
        assert plotter.y == "bin_1"
        # Reset
        plotter.event_selector = "event1"

    def test_update_df_fires_on_event_selector_heatmap_change(self, plotter):
        # Changing event_selector_heatmap triggers _update_df watcher.
        # After update, heatmap_y should reset to ["All"] (last element).
        plotter.event_selector_heatmap = "event2"
        assert plotter.heatmap_y == ["All"]
        # Reset
        plotter.event_selector_heatmap = "event1"

    def test_update_psth_y_fires_on_construction(self, plotter):
        # _update_psth_y fires when event_selector is set in __init__.
        # remove_cols(columns)[:-2] == ["trial_1", "trial_2", "trial_3"]
        # psth_y should be ["1 - trial_1"] (first element)
        assert plotter.psth_y == ["1 - trial_1"]

    def test_update_psth_y_fires_on_event_selector_change(self, plotter):
        # After changing event_selector, psth_y should update to first trial of new event.
        plotter.event_selector = "event2"
        assert plotter.psth_y == ["1 - trial_1"]
        # Reset
        plotter.event_selector = "event1"
