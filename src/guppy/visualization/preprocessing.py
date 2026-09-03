import logging
from collections.abc import Callable

import holoviews as hv
import numpy as np

from .shading import FIT_COLOR, shade_trace

logger = logging.getLogger(__name__)


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


def _shaded_panel(
    *, curve: hv.DynamicMap, title: str, spans: hv.streams.Pipe | None, markable: bool = False
) -> hv.DynamicMap:
    """Size and title one stacked panel, shading it with the span stream when given.

    Options are set on the composed panel rather than the bare curve: options set on an
    Overlay are dropped when the spans layer composes it into a new one, which left the
    fit panel at bokeh's 300x300 default while the two single-curve panels kept theirs.
    """
    shaded = curve if spans is None else _spans_overlay(curve=curve, spans=spans)
    panel_options = dict(title=title, responsive=True, height=220)
    if markable:
        # Marking is what the panel is for, so a drag selects rather than pans; panning
        # and zooming stay one toolbar click away.
        panel_options["active_tools"] = ["xbox_select"]
    return shaded.opts(**panel_options)


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
    on_x_select: Callable[[float, float], None] | None = None,
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
    on_x_select : callable, optional
        Called with the ``(start, end)`` time bounds of a horizontal drag on any panel.
        Passing it arms the panels with a box-select tool and makes dragging their default
        gesture in place of panning; omit it to leave the panels read-only.

    Returns
    -------
    hv.Layout
        Vertically stacked curves, one per trace.
    """

    def trace(values: np.ndarray, title: str) -> hv.DynamicMap:
        """Shade one trace, arming it to report drags when the caller asked for marking."""
        curve = shade_trace(hv.Curve((x, values), "time (s)", title))
        if on_x_select is None:
            return curve
        # Both the select tool and the drag stream ride on the trace layer: bokeh builds a
        # figure's toolbar from what its individual layers ask for, and holoviews matches a
        # stream only to the layer named as its source.
        curve = curve.opts(tools=["xbox_select"])
        hv.streams.BoundsX(source=curve).add_subscriber(lambda boundsx: on_x_select(*boundsx))
        return curve

    def panel(curve: hv.DynamicMap, title: str) -> hv.DynamicMap:
        return _shaded_panel(curve=curve, title=title, spans=spans, markable=on_x_select is not None)

    fit_curve = trace(signal, titles[2]) * shade_trace(hv.Curve((x, fit), "time (s)", titles[2]), color=FIT_COLOR)
    panels = [
        panel(trace(control, titles[0]), f"{suptitle} — {titles[0]}"),
        panel(trace(signal, titles[1]), titles[1]),
        panel(fit_curve, titles[2]),
    ]
    for title, values in (extra_traces or {}).items():
        panels.append(panel(trace(values, title), title))

    return hv.Layout(panels).cols(1)
