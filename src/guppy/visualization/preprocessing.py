import logging

import holoviews as hv
import numpy as np

from .shading import FIT_COLOR, PLOT_WIDTH, shade_trace

logger = logging.getLogger(__name__)


def build_preprocessing_curve(*, suptitle: str, title: str, x: np.ndarray, y: np.ndarray) -> hv.DynamicMap:
    """Build a HoloViews curve of a preprocessing time series.

    Parameters
    ----------
    suptitle : str
        Session-level title prefix.
    title : str
        Trace title (e.g. the ``z_score_<site>`` basename).
    x : np.ndarray
        Time axis values.
    y : np.ndarray
        Signal values to plot.

    Returns
    -------
    hv.DynamicMap
        Density-shaded view of ``y`` versus ``x``.
    """
    curve = hv.Curve((x, y), "time (s)", title)
    return shade_trace(curve).opts(title=f"{suptitle} — {title}", width=PLOT_WIDTH, height=250)


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


def build_control_signal_fit(
    *,
    x: np.ndarray,
    control: np.ndarray,
    signal: np.ndarray,
    fit: np.ndarray,
    titles: list[str],
    suptitle: str,
    spans: hv.streams.Pipe,
) -> hv.Layout:
    """Build three stacked curves (control, signal, signal+fit) with pipe-driven shaded spans.

    Parameters
    ----------
    x : np.ndarray
        Time axis values shared by all three curves.
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
    spans : hv.streams.Pipe
        Stream carrying the ``(start, end)`` windows shaded on all three curves.

    Returns
    -------
    hv.Layout
        Three vertically stacked curves.
    """
    control_curve = shade_trace(hv.Curve((x, control), "time (s)", titles[0]))
    signal_curve = shade_trace(hv.Curve((x, signal), "time (s)", titles[1]))
    fit_curve = shade_trace(hv.Curve((x, signal), "time (s)", titles[2])) * shade_trace(
        hv.Curve((x, fit), "time (s)", titles[2]), color=FIT_COLOR
    )

    # Size and title the composed panel, not the bare curve: options set on an Overlay are
    # dropped when the spans layer composes it into a new one, which left the fit panel at
    # bokeh's 300x300 default while the two single-curve panels kept theirs.
    def panel(curve: hv.DynamicMap, title: str) -> hv.DynamicMap:
        return _spans_overlay(curve=curve, spans=spans).opts(title=title, width=PLOT_WIDTH, height=220)

    return hv.Layout(
        [
            panel(control_curve, f"{suptitle} — {titles[0]}"),
            panel(signal_curve, titles[1]),
            panel(fit_curve, titles[2]),
        ]
    ).cols(1)
