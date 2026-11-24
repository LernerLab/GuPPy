import holoviews as hv
import numpy as np
import pytest
from bokeh.models import Plot

from guppy.visualization.raw_data import (
    build_event_raster,
    build_raw_traces,
    event_colors,
)
from guppy_test_data import resolve_plot


def test_event_colors_skip_the_trace_hue_and_cycle():
    colors = event_colors(event_labels=[f"event_{index}" for index in range(10)])
    assert colors["event_0"] == "#7B4FD0"
    assert colors["event_1"] == "#D08C15"
    assert colors["event_4"] == "#6B7785"
    assert colors["event_5"] == "#7B4FD0"


class TestBuildEventRaster:
    @pytest.fixture
    def raster(self, panel_extension):
        return build_event_raster(
            event_times={"reward": np.array([1.5, 4.0]), "lever": np.array([2.0])},
            colors={"reward": "#7B4FD0", "lever": "#D08C15"},
        )

    def test_draws_one_row_of_ticks_per_event_with_the_first_on_top(self, raster):
        lever_row, reward_row = raster.values()
        assert list(reward_row.dimension_values("Time (s)")) == [1.5, 4.0]
        assert list(lever_row.dimension_values("Time (s)")) == [2.0]
        assert reward_row.opts.get("plot").kwargs["position"] == 1.1
        assert lever_row.opts.get("plot").kwargs["position"] == 0.1

    def test_colors_each_row_by_its_event(self, raster):
        lever_row, reward_row = raster.values()
        assert reward_row.opts.get("style").kwargs["color"] == "#7B4FD0"
        assert lever_row.opts.get("style").kwargs["color"] == "#D08C15"

    def test_labels_each_row_with_its_event(self, raster):
        assert [row.label for row in raster.values()] == ["lever", "reward"]
        assert raster.opts.get("plot").kwargs["yticks"] == [(0.5, "lever"), (1.5, "reward")]


class TestBuildRawTraces:
    @pytest.fixture
    def traces(self):
        return {
            "Control (A)": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
            "Signal (A)": (np.array([0.0, 2.0]), np.array([5.0, 7.0])),
        }

    @pytest.fixture
    def trace_colors(self):
        return {"Control (A)": "#1F6FD1", "Signal (A)": "#3A9E5C"}

    def test_stacks_one_panel_per_trace_with_its_own_times_then_the_raster(self, panel_extension, traces, trace_colors):
        layout = build_raw_traces(
            traces=traces,
            trace_colors=trace_colors,
            event_times={"reward": np.array([1.5])},
            colors={"reward": "#7B4FD0"},
        )

        control, signal, raster = layout.values()
        images = [resolve_plot(control), resolve_plot(signal)]
        assert [image.bounds.lbrt() for image in images] == [(0.0, 1.0, 2.0, 3.0), (0.0, 5.0, 2.0, 7.0)]
        assert [list(row.dimension_values("Time (s)")) for row in raster.values()] == [[1.5]]

    def test_titles_name_the_trace_and_the_value_axis_names_the_quantity(self, panel_extension, traces, trace_colors):
        layout = build_raw_traces(
            traces=traces,
            trace_colors=trace_colors,
            event_times={"reward": np.array([1.5])},
            colors={"reward": "#7B4FD0"},
        )

        plots = [model for model in hv.render(layout).references() if isinstance(model, Plot)]
        labels = {plot.title.text: plot.yaxis[0].axis_label for plot in plots}
        assert labels["Control (A)"] == "Raw Fluorescence (a.u.)"
        assert labels["Signal (A)"] == "Raw Fluorescence (a.u.)"

    def test_without_events_no_raster_is_drawn(self, panel_extension, traces, trace_colors):
        layout = build_raw_traces(traces=traces, trace_colors=trace_colors, event_times={}, colors={})

        assert len(layout) == 2
