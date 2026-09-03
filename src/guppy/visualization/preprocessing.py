import logging
from collections.abc import Callable

import holoviews as hv
import numpy as np
from bokeh.models import BoxSelectTool, PanTool

from .shading import FIT_COLOR, shade_trace

logger = logging.getLogger(__name__)

# Height of the single trace the marking page shows, taller than a stacked panel because
# it is the only plot on the page.
MARKING_PANEL_HEIGHT = 420

# Bokeh arms one drag tool at a time, so marking and panning cannot both be the drag
# gesture; wheel zoom is named alongside whichever it is because naming active_tools at
# all replaces holoviews' defaults, and leaving it out silently disarms scroll-to-zoom.
_MARKING_ACTIVE_TOOLS = ["xbox_select", "wheel_zoom"]


def make_spans_pipe(*, windows: list[tuple[float, float]]) -> hv.streams.Pipe:
    """Build the stream that drives the shaded spans on a control/signal/fit layout.

    Sending a new window list on the returned pipe repaints only the spans; the
    density-shaded traces are not re-aggregated.

    Parameters
    ----------
    windows : list of (float, float)
        Initial ``(start, end)`` windows to shade.

    Returns
    -------
    hv.streams.Pipe
        Stream carrying the current window list.
    """
    return hv.streams.Pipe(data=list(windows))


def _spans_overlay(*, curve: hv.DynamicMap, spans: hv.streams.Pipe) -> hv.DynamicMap:
    """Overlay pipe-driven shaded vertical spans on a curve."""
    return curve * hv.DynamicMap(lambda data: hv.VSpans(list(data)).opts(color="orange", alpha=0.2), streams=[spans])


def _shaded_panel(*, curve: hv.DynamicMap, title: str, spans: hv.streams.Pipe | None) -> hv.DynamicMap:
    """Size and title one stacked panel, shading it with the span stream when given.

    Options are set on the composed panel rather than the bare curve: options set on an
    Overlay are dropped when the spans layer composes it into a new one, which left the
    fit panel at bokeh's 300x300 default while the two single-curve panels kept theirs.
    """
    shaded = curve if spans is None else _spans_overlay(curve=curve, spans=spans)
    return shaded.opts(title=title, responsive=True, height=220)


def build_stacked_traces(
    *,
    x: np.ndarray,
    traces: dict[str, np.ndarray],
    suptitle: str,
    spans: hv.streams.Pipe | None = None,
) -> hv.Layout:
    """Build density-shaded curves stacked over a shared time axis.

    Every panel plots against the same time dimension, so bokeh links their axes: zooming
    one panel zooms them all.

    Parameters
    ----------
    x : np.ndarray
        Time axis values shared by all curves.
    traces : dict
        Panel title → trace values against the same ``x``, stacked in iteration order.
    suptitle : str
        Session-level title prefix applied to the first panel.
    spans : hv.streams.Pipe, optional
        Stream carrying the ``(start, end)`` windows shaded on every curve. Omit to draw
        the traces unshaded.

    Returns
    -------
    hv.Layout
        Vertically stacked curves, one per trace.
    """
    panels = []
    for index, (title, values) in enumerate(traces.items()):
        curve = shade_trace(hv.Curve((x, values), "time (s)", title))
        panels.append(_shaded_panel(curve=curve, title=f"{suptitle} — {title}" if index == 0 else title, spans=spans))
    return hv.Layout(panels).cols(1)


def build_control_signal_fit(
    *,
    x: np.ndarray,
    control: np.ndarray,
    signal: np.ndarray,
    fit: np.ndarray,
    titles: list[str],
    suptitle: str,
    spans: hv.streams.Pipe | None = None,
    extra_traces: dict[str, np.ndarray] | None = None,
) -> hv.Layout:
    """Build stacked curves (control, signal, signal+fit) over a shared time axis.

    Every panel plots against the same time dimension, so bokeh links their axes: zooming
    one panel zooms them all.

    Parameters
    ----------
    x : np.ndarray
        Time axis values shared by all curves.
    control : np.ndarray
        Control channel trace (top).
    signal : np.ndarray
        Signal channel trace (middle).
    fit : np.ndarray
        Fitted control trace overlaid on the signal (bottom).
    titles : list[str]
        Titles for the three curves (control, signal, fit).
    suptitle : str
        Session-level title prefix applied to the control curve.
    spans : hv.streams.Pipe, optional
        Stream carrying the ``(start, end)`` windows shaded on every curve. Omit to draw
        the traces unshaded.
    extra_traces : dict, optional
        Further traces to stack below the fit, as title → values against the same ``x``.

    Returns
    -------
    hv.Layout
        Vertically stacked curves, one per trace.
    """
    control_curve = shade_trace(hv.Curve((x, control), "time (s)", titles[0]))
    signal_curve = shade_trace(hv.Curve((x, signal), "time (s)", titles[1]))
    fit_curve = shade_trace(hv.Curve((x, signal), "time (s)", titles[2])) * shade_trace(
        hv.Curve((x, fit), "time (s)", titles[2]), color=FIT_COLOR
    )

    def panel(curve: hv.DynamicMap, title: str) -> hv.DynamicMap:
        return _shaded_panel(curve=curve, title=title, spans=spans)

    panels = [
        panel(control_curve, f"{suptitle} — {titles[0]}"),
        panel(signal_curve, titles[1]),
        panel(fit_curve, titles[2]),
    ]
    for title, values in (extra_traces or {}).items():
        panels.append(panel(shade_trace(hv.Curve((x, values), "time (s)", title)), title))

    return hv.Layout(panels).cols(1)


def build_markable_trace(
    *,
    x: np.ndarray,
    values: np.ndarray,
    title: str,
    spans: hv.streams.Pipe,
    on_x_select: Callable[[float, float], None],
    overlay: np.ndarray | None = None,
    hooks: list[Callable[[object, object], None]] | None = None,
) -> hv.DynamicMap:
    """Build one density-shaded trace whose horizontal drags report the times they cover.

    Parameters
    ----------
    x : np.ndarray
        Time axis values.
    values : np.ndarray
        Trace values against ``x``.
    title : str
        Panel title, also the value dimension name.
    spans : hv.streams.Pipe
        Stream carrying the ``(start, end)`` windows shaded over the trace.
    on_x_select : callable
        Called with the ``(start, end)`` time bounds of a horizontal drag on the trace.
    overlay : np.ndarray, optional
        A second trace against the same ``x``, drawn over the first in the fit hue.
    hooks : list of callable, optional
        Bokeh hooks run against the rendered figure, as ``(plot, element)``.

    Returns
    -------
    hv.DynamicMap
        The shaded trace with its span layer.
    """
    curve = shade_trace(hv.Curve((x, values), "time (s)", title))
    # Both the select tool and the drag stream ride on the trace layer: bokeh builds a
    # figure's toolbar from what its individual layers ask for, and holoviews matches a
    # stream only to the layer named as its source.
    curve = curve.opts(tools=["xbox_select"])
    hv.streams.BoundsX(source=curve).add_subscriber(lambda boundsx: on_x_select(*boundsx))
    if overlay is not None:
        curve = curve * shade_trace(hv.Curve((x, overlay), "time (s)", title), color=FIT_COLOR)
    return _spans_overlay(curve=curve, spans=spans).opts(
        title=title,
        responsive=True,
        height=MARKING_PANEL_HEIGHT,
        active_tools=_MARKING_ACTIVE_TOOLS,
        hooks=list(hooks or []),
    )


def set_drag_gesture(*, figure: object, marking: bool) -> None:
    """Point a rendered figure's drag at marking or at panning, leaving the view as it is.

    Swapping the armed tool on the live bokeh model rather than rebuilding the plot is what
    lets a zoomed-in view survive the switch, which is the whole point of being able to
    navigate and then go back to marking.

    Parameters
    ----------
    figure : bokeh.plotting.figure
        The rendered figure to re-arm.
    marking : bool
        True to arm the horizontal box-select, False to arm panning.
    """
    wanted = BoxSelectTool if marking else PanTool
    figure.toolbar.active_drag = next(tool for tool in figure.toolbar.tools if isinstance(tool, wanted))
