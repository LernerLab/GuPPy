"""Display-time rendering for the raw traces Step 2 (Read Raw Data) writes."""

import holoviews as hv
import numpy as np

from .labels import TIME_LABEL
from .palette import CONTROL, PLOT_COLORS, SIGNAL
from .shading import shade_trace

# Events take the plot colors the control and signal do not, so an event line never reads as a trace.
_EVENT_PALETTE = [color for color in PLOT_COLORS if color not in (CONTROL, SIGNAL)]


def event_colors(*, event_labels: list[str]) -> dict[str, str]:
    """Assign each event a fixed hue, cycling through the palette.

    Parameters
    ----------
    event_labels : list of str
        Event labels, in the order they are offered.

    Returns
    -------
    dict
        Event label → hex color.
    """
    return {label: _EVENT_PALETTE[index % len(_EVENT_PALETTE)] for index, label in enumerate(event_labels)}


def build_raw_traces(
    *,
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    event_times: dict[str, np.ndarray],
    colors: dict[str, str],
) -> hv.Layout:
    """Build density-shaded raw traces stacked over a shared time axis, with event times marked.

    Every panel plots against the same time dimension, so bokeh links their axes: zooming
    one panel zooms them all. Each event is drawn as vertical lines in its own color on
    every panel, with the legend on the first.

    Parameters
    ----------
    traces : dict
        Panel title → ``(times, values)``, stacked in iteration order. Each trace carries
        its own sample times.
    event_times : dict
        Event label → times at which the event occurred.
    colors : dict
        Event label → color its lines are drawn in.

    Returns
    -------
    hv.Layout
        Vertically stacked panels, one per trace.
    """
    panels = []
    for index, (title, (times, values)) in enumerate(traces.items()):
        panel = shade_trace(hv.Curve((times, values), TIME_LABEL, title))
        for label, times_of_event in event_times.items():
            panel = panel * hv.VLines(list(times_of_event), kdims=[TIME_LABEL], label=label).opts(
                color=colors[label], alpha=0.6, line_width=1
            )
        panels.append(
            panel.opts(
                title=title,
                responsive=True,
                height=260,
                show_legend=index == 0,
                legend_position="right",
            )
        )
    return hv.Layout(panels).cols(1)
