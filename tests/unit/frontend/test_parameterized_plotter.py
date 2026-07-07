import os
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from guppy.frontend.parameterized_plotter import (
    ParameterizedPlotter,
    make_dir,
    overview_y_options,
    remove_cols,
)


class _FakeRange:
    """Minimal stand-in for a Bokeh ``Range1d`` used to exercise _range_sync_hook."""

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.callbacks = []

    def on_change(self, attr, callback):
        self.callbacks.append(callback)


class _FakeFigure:
    def __init__(self):
        self.x_range = _FakeRange(0.0, 1.0)
        self.y_range = _FakeRange(-1.0, 1.0)


class _FakePlot:
    def __init__(self, figure):
        self.state = figure


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
# overview_y_options utility
# ---------------------------------------------------------------------------


def test_overview_y_options_keeps_mean_all_and_bins_drops_trials():
    columns = ["trial_1", "trial_2", "bin_(0-5)", "timestamps", "mean", "err", "bin_err_(0-5)", "All"]
    result = overview_y_options(columns)
    assert result == ["bin_(0-5)", "mean", "All"]


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


@pytest.fixture
def single_trial_plotter(tmp_path, panel_extension):
    """Plotter whose event heatmap has a single trial row.

    This is the shape a group average takes when only one selected session
    contributes to an event (e.g. one 'novel object' session averaged alongside
    one 'novel female' session). Under the "All" heatmap selection the ``bin_1``
    row is dropped, leaving exactly one trial row so ``heatmap()`` takes its
    single-trial branch.
    """
    columns = ["trial_1", "bin_1", "timestamps", "mean", "err", "bin_err_1"]
    n_timepoints = 30
    timestamps = np.linspace(-5.0, 10.0, n_timepoints)

    def make_event_dataframe():
        return pd.DataFrame(
            {column: (timestamps if column == "timestamps" else np.arange(float(n_timepoints))) for column in columns}
        )

    events = ["event1", "event2"]
    columns_dict = {event: columns + ["All"] for event in events}
    df_new = pd.concat([make_event_dataframe() for event in events], keys=events, axis=1)

    return ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["mean", "All"],
        heatmap_y_objects=["1 - trial_1", "All"],
        psth_y_objects=None,
        filepath=str(tmp_path),
        columns_dict=columns_dict,
        df_new=df_new,
        x_min=-5.0,
        x_max=10.0,
    )


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

    def test_y_options_exclude_individual_trials(self, plotter):
        # The overview y-selector keeps only whole-event views (bins, mean, All),
        # not individual trial columns; here that leaves ["bin_1", "mean"].
        assert plotter.param["y"].objects == ["bin_1", "mean"]

    def test_y_default_is_overview_option(self, plotter):
        # Default is objects[-2]; with ["bin_1", "mean"] that is "bin_1".
        assert plotter.y == "bin_1"

    def test_heatmap_y_set_to_last_heatmap_y_object(self, plotter):
        # heatmap_y_objects[-1] == "All"
        assert plotter.heatmap_y == ["All"]

    def test_x_ranges_initialized_from_x_min_x_max(self, plotter):
        # Each plot's X range is seeded with the padded PSTH window in __init__.
        assert plotter.cont_X == (-5.0, 10.0)
        assert plotter.overlay_X == (-5.0, 10.0)
        assert plotter.trials_X == (-5.0, 10.0)

    def test_y_ranges_start_unset_for_autofit(self, plotter):
        # Y ranges are left as None so the first render auto-fits to the data.
        assert plotter.cont_Y is None
        assert plotter.overlay_Y is None
        assert plotter.trials_Y is None

    def test_default_trace_color(self, plotter):
        assert plotter.trace_color == "#0000ff"

    def test_default_mean_color(self, plotter):
        assert plotter.mean_color == "#000000"

    def test_default_select_trials_checkbox(self, plotter):
        assert plotter.select_trials_checkbox == ["just trials"]

    def test_default_y_label(self, plotter):
        assert plotter.Y_Label == "y"

    def test_default_save_options(self, plotter):
        assert plotter.save_options_cont == "None"
        assert plotter.save_options_overlay == "None"
        assert plotter.save_options_trials == "None"

    def test_default_overlay_palette(self, plotter):
        assert plotter.overlay_palette == "Category10"

    def test_default_height_plot(self, plotter):
        assert plotter.Height_Plot == 300

    def test_default_width_plot(self, plotter):
        assert plotter.Width_Plot == 1000

    def test_update_x_y_fires_on_event_selector_change(self, plotter):
        # Changing event_selector triggers _update_x_y watcher.
        # columns[-4] == "timestamps"; overview_y_options(columns)[-2] == "bin_1"
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

    def test_save_cont_plot_creates_png(self, plotter_for_save):
        plotter_for_save.save_options_cont = "save_png_format"
        plotter_for_save.save_cont_plot()
        assert Path(plotter_for_save.results_psth["op"] + ".png").exists()

    def test_save_overlay_plot_creates_svg(self, plotter_for_save):
        plotter_for_save.save_options_overlay = "save_svg_format"
        plotter_for_save.save_overlay_plot()
        assert Path(plotter_for_save.results_psth["op_combine"] + ".svg").exists()

    def test_save_trials_plot_creates_png(self, plotter_for_save):
        plotter_for_save.save_options_trials = "save_png_format"
        plotter_for_save.save_trials_plot()
        assert Path(plotter_for_save.results_psth["op_trials"] + ".png").exists()

    def test_save_cont_plot_returns_zero_when_none(self, plotter_for_save):
        plotter_for_save.save_options_cont = "None"
        assert plotter_for_save.save_cont_plot() == 0

    def test_save_overlay_plot_returns_zero_when_none(self, plotter_for_save):
        plotter_for_save.save_options_overlay = "None"
        assert plotter_for_save.save_overlay_plot() == 0

    def test_save_trials_plot_returns_zero_when_none(self, plotter_for_save):
        plotter_for_save.save_options_trials = "None"
        assert plotter_for_save.save_trials_plot() == 0

    def test_save_cont_plot_creates_png_and_svg_when_both(self, plotter_for_save):
        plotter_for_save.save_options_cont = "save_both_format"
        plotter_for_save.save_cont_plot()
        assert Path(plotter_for_save.results_psth["op"] + ".png").exists()
        assert Path(plotter_for_save.results_psth["op"] + ".svg").exists()

    def test_save_hm_plots_creates_png(self, plotter_for_save):
        plotter_for_save.save_options_heatmap = "save_png_format"
        plotter_for_save.save_hm_plots()
        assert Path(plotter_for_save.results_hm["op"] + ".png").exists()

    def test_save_hm_plots_creates_svg(self, plotter_for_save):
        plotter_for_save.save_options_heatmap = "save_svg_format"
        plotter_for_save.save_hm_plots()
        assert Path(plotter_for_save.results_hm["op"] + ".svg").exists()

    def test_save_hm_plots_returns_zero_when_save_options_none(self, plotter_for_save):
        plotter_for_save.save_options_heatmap = "None"
        result = plotter_for_save.save_hm_plots()
        assert result == 0

    def test_save_hm_plots_creates_png_and_svg_when_save_both(self, plotter_for_save):
        plotter_for_save.save_options_heatmap = "save_both_format"
        plotter_for_save.save_hm_plots()

        assert Path(plotter_for_save.results_hm["op"] + ".png").exists()
        assert Path(plotter_for_save.results_hm["op"] + ".svg").exists()

    def test_update_selector_handles_bin_series_selection(self, plotter):
        plotter.param.selector_for_multipe_events_plot.objects = ["event1", "event2", "event1_bin_1"]
        plotter.selector_for_multipe_events_plot = ["event1_bin_1"]

        plot = plotter.update_selector()

        assert plot is not None
        assert plotter.results_psth["op_combine"].endswith(os.path.join("saved_plots", "['event1_bin_1']_mean"))

    def test_cont_plot_all_trials_branch(self, plotter):
        plotter.param["y"].objects = ["trial_1", "trial_2", "trial_3", "bin_1", "mean", "All"]
        plotter.y = "All"

        plot = plotter.contPlot()

        assert plot is not None
        assert plotter.results_psth["op"].endswith(os.path.join("saved_plots", "event1_All"))

    def test_plot_specific_trials_mean_branch(self, plotter):
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.select_trials_checkbox = ["mean"]

        plot = plotter.plot_specific_trials()

        assert plot is not None

    def test_plot_specific_trials_mean_and_just_trials_branch(self, plotter):
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.select_trials_checkbox = ["mean", "just trials"]

        plot = plotter.plot_specific_trials()

        assert plot is not None

    def test_heatmap_non_all_selection_branch(self, plotter):
        plotter.heatmap_y = ["1 - trial_1", "2 - trial_2"]

        image = plotter.heatmap()

        assert image is not None
        assert plotter.results_hm["op"].endswith(os.path.join("saved_plots", "event1_heatmap"))

    def test_heatmap_single_trial_uses_datashaded_path(self, single_trial_plotter, plotter):
        # A single-trial heatmap used to build a raw QuadMesh spanning the full
        # ~30k-sample time axis, which overflowed Bokeh's client-side renderer
        # ("Maximum call stack size exceeded") and blanked the whole dashboard.
        # The fix duplicates the lone trial into a 2-row mesh so it flows through
        # the same datashaded overlay as multi-trial heatmaps -- so a single-trial
        # heatmap must return the same element type as a multi-trial one, not the
        # old bare single-row QuadMesh.
        single = single_trial_plotter.heatmap()
        multi = plotter.heatmap()

        assert single is not None
        assert not isinstance(single, hv.QuadMesh)
        assert type(single) is type(multi)
        # The datashaded element must render to Bokeh without raising.
        hv.render(single)

    def test_cont_plot_renders_with_custom_trace_color(self, plotter):
        # Default y is "bin_1", which hits the trace_color branch of contPlot.
        plotter.trace_color = "#ff8800"

        plot = plotter.contPlot()

        assert plot is not None

    def test_cont_plot_all_trials_renders_with_custom_trace_color(self, plotter):
        # The all-trials mean line uses trace_color (mean_color belongs to plot 3 now).
        plotter.param["y"].objects = ["trial_1", "trial_2", "trial_3", "bin_1", "mean", "All"]
        plotter.y = "All"
        plotter.trace_color = "#12ab34"

        plot = plotter.contPlot()

        assert plot is not None

    def test_plot_specific_trials_mean_renders_with_custom_trace_color(self, plotter):
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.select_trials_checkbox = ["mean"]
        plotter.trace_color = "#ff8800"

        plot = plotter.plot_specific_trials()

        assert plot is not None

    def test_plot_specific_trials_mean_and_trials_renders_with_custom_mean_color(self, plotter):
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.select_trials_checkbox = ["mean", "just trials"]
        plotter.mean_color = "#12ab34"

        plot = plotter.plot_specific_trials()

        assert plot is not None

    def test_plot_specific_trials_just_trials_renders(self, plotter):
        # Regression: the just-trials branch must auto-fit trials_Y before render,
        # otherwise HoloViews rejects ylim=None at render time.
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.select_trials_checkbox = ["just trials"]

        plot = plotter.plot_specific_trials()
        hv.render(plot)

        assert plotter.trials_Y is not None

    def test_hide_minor_ticks_defaults_to_false(self, plotter):
        assert plotter.hide_minor_ticks is False

    def test_minor_ticks_shown_by_default_on_cont_plot(self, plotter):
        # With the default (False) the freshly rendered figure keeps Bokeh's minor ticks.
        figure = hv.render(plotter.contPlot())
        assert figure.xaxis[0].minor_tick_line_color is not None
        assert figure.yaxis[0].minor_tick_line_color is not None

    def test_hide_minor_ticks_removes_them_on_cont_plot(self, plotter):
        plotter.hide_minor_ticks = True
        figure = hv.render(plotter.contPlot())
        assert figure.xaxis[0].minor_tick_line_color is None
        assert figure.yaxis[0].minor_tick_line_color is None

    def test_hide_minor_ticks_removes_them_on_comparison_plot(self, plotter):
        plotter.hide_minor_ticks = True
        figure = hv.render(plotter.update_selector())
        assert figure.xaxis[0].minor_tick_line_color is None
        assert figure.yaxis[0].minor_tick_line_color is None

    def test_hide_minor_ticks_removes_them_on_selected_trials_plot(self, plotter):
        plotter.psth_y = ["1 - trial_1", "2 - trial_2"]
        plotter.hide_minor_ticks = True
        figure = hv.render(plotter.plot_specific_trials())
        assert figure.xaxis[0].minor_tick_line_color is None
        assert figure.yaxis[0].minor_tick_line_color is None

    def test_range_sync_hook_writes_zoom_into_named_params(self, plotter):
        # The hook mirrors a Bokeh zoom/pan of the figure into the plot's own range params.
        hook = plotter._range_sync_hook("cont", "cont_X", "cont_Y")
        figure = _FakeFigure()
        hook(_FakePlot(figure), None)  # registers on_change callbacks

        figure.x_range.start, figure.x_range.end = 2.0, 6.0
        figure.y_range.start, figure.y_range.end = -0.5, 0.5
        figure.x_range.callbacks[0]("end", 1.0, 6.0)  # simulate Bokeh firing the callback

        assert plotter.cont_X == (2.0, 6.0)
        assert plotter.cont_Y == (-0.5, 0.5)

    def test_move_figure_to_range_moves_live_figure(self, plotter):
        # A typed box edit sets the range params then calls move_figure_to_range to
        # move the already-rendered figure in place (no re-render), which is what
        # keeps an interactive zoom from being interrupted.
        figure = _FakeFigure()
        plotter._figures["cont"] = figure

        plotter.cont_X = (0.0, 3.0)
        plotter.cont_Y = (-2.0, 2.0)
        plotter.move_figure_to_range("cont_X")

        assert (figure.x_range.start, figure.x_range.end) == (0.0, 3.0)
        assert (figure.y_range.start, figure.y_range.end) == (-2.0, 2.0)

    def test_range_param_change_alone_does_not_move_figure(self, plotter):
        # Setting a range param is no longer a figure side-effect on its own: a render
        # writes these params too and must not touch the stale figure. Only an explicit
        # move_figure_to_range (from a box edit) moves it.
        figure = _FakeFigure()
        plotter._figures["cont"] = figure
        original = (figure.x_range.start, figure.x_range.end, figure.y_range.start, figure.y_range.end)

        plotter.cont_X = (0.0, 3.0)
        plotter.cont_Y = (-2.0, 2.0)

        assert (figure.x_range.start, figure.x_range.end, figure.y_range.start, figure.y_range.end) == original

    def test_range_params_not_in_plot_dependencies(self, plotter):
        # Range params must stay out of the plot methods' @param.depends lists so a
        # zoom/pan does not trigger a re-render that would fight the user's gesture.
        for method in ("contPlot", "update_selector", "plot_specific_trials"):
            dependencies = {dependency.name for dependency in plotter.param.method_dependencies(method)}
            assert dependencies.isdisjoint({"cont_X", "cont_Y", "overlay_X", "overlay_Y", "trials_X", "trials_Y"})

    def test_range_sync_hook_only_touches_named_params(self, plotter):
        # Syncing cont_X/cont_Y must not disturb the overlay/trials ranges.
        hook = plotter._range_sync_hook("cont", "cont_X", "cont_Y")
        figure = _FakeFigure()
        hook(_FakePlot(figure), None)
        figure.x_range.start, figure.x_range.end = 2.0, 6.0
        figure.x_range.callbacks[0]("end", 1.0, 6.0)

        assert plotter.overlay_X == (-5.0, 10.0)
        assert plotter.trials_X == (-5.0, 10.0)

    def test_default_overlay_color_overrides_is_empty(self, plotter):
        assert plotter.overlay_color_overrides == {}

    def test_overlay_effective_colors_uses_palette_when_no_overrides(self, plotter):
        # Category10 palette; two events selected in order -> its first two colors.
        plotter.selector_for_multipe_events_plot = ["event1", "event2"]

        assert plotter.overlay_effective_colors() == {"event1": "#1f77b4", "event2": "#ff7f0e"}

    def test_overlay_effective_colors_override_wins(self, plotter):
        # An explicit override replaces only that event's color; the other keeps its palette color.
        plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        plotter.overlay_color_overrides = {"event2": "#123456"}

        assert plotter.overlay_effective_colors() == {"event1": "#1f77b4", "event2": "#123456"}

    def test_overlay_effective_colors_cycles_palette_beyond_its_length(self, plotter):
        # Dark2 has 8 colors; a 9th event wraps back to the first color.
        events = [f"event{i}" for i in range(9)]
        plotter.param.selector_for_multipe_events_plot.objects = events
        plotter.overlay_palette = "Dark2"
        plotter.selector_for_multipe_events_plot = events

        colors = plotter.overlay_effective_colors()
        assert colors["event0"] == "#1b9e77"
        assert colors["event8"] == colors["event0"]

    def test_update_selector_renders_with_override_color(self, plotter):
        plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        plotter.overlay_color_overrides = {"event1": "#abcdef"}

        plot = plotter.update_selector()

        assert plot is not None

    def test_overlay_color_overrides_in_update_selector_dependencies(self, plotter):
        dependencies = {dependency.name for dependency in plotter.param.method_dependencies("update_selector")}
        assert "overlay_color_overrides" in dependencies

    def test_update_selector_autofit_does_not_push_to_live_figure(self, plotter):
        # Regression: selecting a new event refits overlay_Y mid-render. That must NOT
        # mutate the already-rendered figure -- a raw Bokeh range write during a
        # re-render violates the server's document lock and crashes the callback.
        figure = _FakeFigure()
        plotter._figures["overlay"] = figure
        original_y_range = (figure.y_range.start, figure.y_range.end)

        plotter.selector_for_multipe_events_plot = ["event1", "event2"]
        plot = plotter.update_selector()

        assert plot is not None
        # The auto-fit still ran (so the number boxes reflect the new data)...
        assert plotter.overlay_Y is not None
        # ...but the live figure was left untouched: only a typed box edit moves it,
        # never a render, so the range param write here has no figure side-effect.
        assert (figure.y_range.start, figure.y_range.end) == original_y_range


# ---------------------------------------------------------------------------
# plotter_for_save fixture (function-scoped, used only in this module)
# ---------------------------------------------------------------------------


@pytest.fixture
def plotter_for_save(tmp_path, panel_extension):
    """ParameterizedPlotter with results_psth and results_hm pre-populated.

    Uses a minimal hv.Curve so save_psth_plot / save_hm_plots can render and
    export without needing a full pipeline run.
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
    df_new = pd.concat([make_event_dataframe() for event in events], keys=events, axis=1)

    plotter = ParameterizedPlotter(
        event_selector_objects=events,
        event_selector_heatmap_objects=events,
        selector_for_multipe_events_plot_objects=events,
        color_map_objects=["plasma", "viridis"],
        x_objects=["timestamps"],
        y_objects=["trial_1", "mean"],
        heatmap_y_objects=["1 - trial_1", "2 - trial_2", "All"],
        psth_y_objects=None,
        filepath=str(tmp_path),
        columns_dict=columns_dict,
        df_new=df_new,
        x_min=-5.0,
        x_max=10.0,
    )

    saved_plots_dir = tmp_path / "saved_plots"
    saved_plots_dir.mkdir()
    curve = hv.Curve(([0, 1, 2], [0, 1, 0]))
    op_prefix = str(saved_plots_dir / "test_plot")

    plotter.results_psth = {
        "plot_combine": curve,
        "op_combine": op_prefix + "_combine",
        "plot": curve,
        "op": op_prefix,
        "trials": curve,
        "op_trials": op_prefix + "_trials",
    }
    plotter.results_hm = {
        "plot": curve,
        "op": op_prefix + "_hm",
    }

    return plotter
