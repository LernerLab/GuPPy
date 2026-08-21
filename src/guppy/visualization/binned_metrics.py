"""Display-time rendering for whole-session time-binned metrics.

The binned values are drawn as bars spanning the bins they summarize, stacked
under the trace they were reduced from. Both panels share the ``time (s)``
dimension, so bokeh links their axes and zooming the bars zooms the trace with
them -- which is what makes a bin width that is too coarse or too fine visible
at a glance.
"""

import logging

import holoviews as hv
import numpy as np

from .shading import shade_trace

logger = logging.getLogger(__name__)

BAR_COLOR = "#1f77b4"


def build_bin_bars(
    *, bin_starts: np.ndarray, bin_ends: np.ndarray, values: np.ndarray, value_label: str
) -> hv.Rectangles:
    """Draw one bar per time bin, spanning that bin's true bounds.

    Bins whose value is NaN -- a stretch of recording lost to artifact removal --
    are left empty rather than drawn at zero.

    Parameters
    ----------
    bin_starts, bin_ends : np.ndarray
        Bin bounds in seconds. The final pair may be narrower than the rest.
    values : np.ndarray
        One value per bin.
    value_label : str
        Axis label for the values, e.g. ``"mean z-score"``.

    Returns
    -------
    hv.Rectangles
        Bars from zero to each bin's value. The first key dimension is
        ``"time (s)"``, so this panel links axes with a trace plotted against the
        same dimension.
    """
    bin_starts = np.asarray(bin_starts, dtype=float)
    bin_ends = np.asarray(bin_ends, dtype=float)
    values = np.asarray(values, dtype=float)

    drawable = ~np.isnan(values)
    bars = list(zip(bin_starts[drawable], np.zeros(np.count_nonzero(drawable)), bin_ends[drawable], values[drawable]))

    return hv.Rectangles(
        bars,
        kdims=["time (s)", value_label, "time (s) end", value_label + " end"],
    )


def build_binned_metrics_panel(
    *,
    timestamps: np.ndarray,
    trace: np.ndarray,
    trace_label: str,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    values: np.ndarray,
    value_label: str,
    suptitle: str,
    panel_height: int = 220,
) -> hv.Layout:
    """Stack the binned values under the full-session trace they reduce.

    Parameters
    ----------
    timestamps : np.ndarray
        Corrected time axis (s) of the trace.
    trace : np.ndarray
        Full-session trace the metric was reduced from.
    trace_label : str
        Axis label for the trace, e.g. ``"z-score"``.
    bin_starts, bin_ends : np.ndarray
        Bin bounds in seconds.
    values : np.ndarray
        One value per bin.
    value_label : str
        Axis label for the binned values.
    suptitle : str
        Session-level title prefix applied to the trace panel.
    panel_height : int, optional
        Height in pixels of each of the two panels.

    Returns
    -------
    hv.Layout
        The shaded trace above, the per-bin bars below, on a linked time axis.
    """
    trace_curve = shade_trace(hv.Curve((timestamps, trace), "time (s)", trace_label)).opts(
        title=f"{suptitle} — {trace_label}", responsive=True, height=panel_height
    )
    bars = build_bin_bars(bin_starts=bin_starts, bin_ends=bin_ends, values=values, value_label=value_label).opts(
        color=BAR_COLOR, line_color="white", title=value_label, responsive=True, height=panel_height
    )

    return hv.Layout([trace_curve, bars]).cols(1)
