import holoviews as hv
import numpy as np

from guppy.visualization.raw_data import build_raw_traces, event_colors
from guppy_test_data import resolve_plot


def test_event_colors_skip_the_trace_hue_and_cycle():
    colors = event_colors(event_labels=[f"event_{index}" for index in range(10)])
    assert colors["event_0"] == "#ff7f0e"
    assert colors["event_1"] == "#2ca02c"
    assert colors["event_8"] == "#17becf"
    assert colors["event_9"] == "#ff7f0e"


def test_build_raw_traces_stacks_one_panel_per_trace_with_its_own_times(panel_extension):
    layout = build_raw_traces(
        traces={
            "control_A": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
            "signal_A": (np.array([0.0, 2.0]), np.array([5.0, 7.0])),
        },
        event_times={"reward": np.array([1.5])},
        colors={"reward": "#2ca02c"},
        suptitle="output_1",
    )

    panels = [resolve_plot(panel) for panel in layout.values()]
    assert [panel.opts.get("plot").kwargs["title"] for panel in panels] == ["output_1 — control_A", "signal_A"]
    assert [panel.opts.get("plot").kwargs["show_legend"] for panel in panels] == [True, False]
    images = [next(item for item in panel.values() if isinstance(item, hv.RGB)) for panel in panels]
    assert [image.bounds.lbrt() for image in images] == [(0.0, 1.0, 2.0, 3.0), (0.0, 5.0, 2.0, 7.0)]
    lines = [next(item for item in panel.values() if isinstance(item, hv.VLines)) for panel in panels]
    assert [list(line.data["time (s)"]) for line in lines] == [[1.5], [1.5]]
    assert [line.opts.get("style").kwargs["color"] for line in lines] == ["#2ca02c", "#2ca02c"]
