"""Display-time rendering for the raw traces Step 2 (Read Raw Data) writes."""

import holoviews as hv
import numpy as np

from .labels import RAW_FLUORESCENCE_LABEL, TIME_LABEL
from .palette import CONTROL, PLOT_COLORS, SIGNAL
from .shading import shade_trace

# Events take the plot colors the control and signal do not, so an event tick never reads as a trace.
_EVENT_PALETTE = [color for color in PLOT_COLORS if color not in (CONTROL, SIGNAL)]

# Height of one event's row in the raster, and of the raster's title and time axis around the rows.
_RASTER_ROW_HEIGHT = 26
_RASTER_FRAME_HEIGHT = 90


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


def build_event_raster(*, event_times: dict[str, np.ndarray], colors: dict[str, str]) -> hv.Overlay:
    """Build a raster with one row of ticks per event, the first event on top.

    Parameters
    ----------
    event_times : dict
        Event label → times at which the event occurred.
    colors : dict
        Event label → color its ticks are drawn in.

    Returns
    -------
    hv.Overlay
        One ``Spikes`` row per event, labeled with the event on the y-axis.
    """
    labels = list(event_times)
    rows = [
        hv.Spikes(list(event_times[label]), kdims=[TIME_LABEL], label=label).opts(
            position=row + 0.1, spike_length=0.8, color=colors[label], line_width=1.5
        )
        for row, label in enumerate(reversed(labels))
    ]
    return hv.Overlay(rows).opts(
        title="Events",
        yticks=[(row + 0.5, label) for row, label in enumerate(reversed(labels))],
        ylim=(0, len(labels)),
        ylabel="",
        responsive=True,
        height=_RASTER_FRAME_HEIGHT + _RASTER_ROW_HEIGHT * len(labels),
        show_legend=False,
    )


def build_raw_traces(
    *,
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    trace_colors: dict[str, str],
    event_times: dict[str, np.ndarray],
    colors: dict[str, str],
) -> hv.Layout:
    """Build density-shaded raw traces stacked over a shared time axis, with an event raster below.

    Every panel plots against the same time dimension, so bokeh links their axes: zooming
    one panel zooms them all.

    Parameters
    ----------
    traces : dict
        Panel title → ``(times, values)``, stacked in iteration order. Each trace carries
        its own sample times.
    trace_colors : dict
        Panel title → color the trace is shaded in.
    event_times : dict
        Event label → times at which the event occurred. With no events, no raster is drawn.
    colors : dict
        Event label → color its ticks are drawn in.

    Returns
    -------
    hv.Layout
        Vertically stacked panels, one per trace, then the event raster.
    """
    panels = [
        shade_trace(hv.Curve((times, values), TIME_LABEL, title), color=trace_colors[title]).opts(
            title=title, ylabel=RAW_FLUORESCENCE_LABEL, responsive=True, height=260
        )
        for title, (times, values) in traces.items()
    ]
    if event_times:
        panels.append(build_event_raster(event_times=event_times, colors=colors))
    return hv.Layout(panels).cols(1)
