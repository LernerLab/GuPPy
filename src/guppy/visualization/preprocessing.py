import logging

import holoviews as hv
import numpy as np

from .shading import FIT_COLOR, PLOT_WIDTH, shade_trace

logger = logging.getLogger(__name__)

_ARTIFACTS_REMOVED_NOTE = "Note: Artifacts have been removed, but are not reflected in this plot."


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
    artifacts_have_been_removed: bool,
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
    artifacts_have_been_removed : bool
        When True, annotates the bottom curve that artifacts were removed.
    spans : hv.streams.Pipe
        Stream carrying the ``(start, end)`` windows shaded on all three curves.

    Returns
    -------
    hv.Layout
        Three vertically stacked curves.
    """
    control_curve = shade_trace(hv.Curve((x, control), "time (s)", titles[0])).opts(
        title=f"{suptitle} — {titles[0]}", width=PLOT_WIDTH, height=220
    )
    signal_curve = shade_trace(hv.Curve((x, signal), "time (s)", titles[1])).opts(
        title=titles[1], width=PLOT_WIDTH, height=220
    )
    fit_title = titles[2] + (f" ({_ARTIFACTS_REMOVED_NOTE})" if artifacts_have_been_removed else "")
    fit_curve = (
        shade_trace(hv.Curve((x, signal), "time (s)", titles[2]))
        * shade_trace(hv.Curve((x, fit), "time (s)", titles[2]), color=FIT_COLOR)
    ).opts(title=fit_title, width=PLOT_WIDTH, height=220)

    return hv.Layout(
        [
            _spans_overlay(curve=control_curve, spans=spans),
            _spans_overlay(curve=signal_curve, spans=spans),
            _spans_overlay(curve=fit_curve, spans=spans),
        ]
    ).cols(1)
