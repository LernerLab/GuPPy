import logging

import holoviews as hv
import numpy as np

from .downsampling import PLOT_WIDTH, downsample_for_display

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
        Downsampled curve of ``y`` versus ``x``.
    """
    curve = hv.Curve((x, y), "time (s)", title).opts(title=f"{suptitle} — {title}", width=PLOT_WIDTH, height=250)
    return downsample_for_display(curve)


def _shade_windows(curve: hv.DynamicMap, windows: list[tuple[float, float]] | None) -> hv.DynamicMap:
    """Overlay shaded vertical spans for each ``(start, end)`` window on a curve."""
    if not windows:
        return curve
    overlay = curve
    for start, end in windows:
        overlay = overlay * hv.VSpan(float(start), float(end)).opts(color="orange", alpha=0.2)
    return overlay


def build_control_signal_fit(
    *,
    x: np.ndarray,
    control: np.ndarray,
    signal: np.ndarray,
    fit: np.ndarray,
    titles: list[str],
    suptitle: str,
    artifacts_have_been_removed: bool,
    windows: list[tuple[float, float]] | None = None,
) -> hv.Layout:
    """Build three stacked curves (control, signal, signal+fit) with optional shaded windows.

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
    windows : list of (float, float), optional
        Good-chunk ``(start, end)`` windows shaded as vertical spans on all three curves.

    Returns
    -------
    hv.Layout
        Three vertically stacked curves.
    """
    control_curve = downsample_for_display(
        hv.Curve((x, control), "time (s)", titles[0]).opts(
            title=f"{suptitle} — {titles[0]}", width=PLOT_WIDTH, height=220
        )
    )
    signal_curve = downsample_for_display(
        hv.Curve((x, signal), "time (s)", titles[1]).opts(title=titles[1], width=PLOT_WIDTH, height=220)
    )
    fit_title = titles[2] + (f" ({_ARTIFACTS_REMOVED_NOTE})" if artifacts_have_been_removed else "")
    fit_curve = (
        (
            downsample_for_display(hv.Curve((x, signal), "time (s)", titles[2]))
            * downsample_for_display(hv.Curve((x, fit), "time (s)", titles[2]))
        )
        .opts(hv.opts.Curve(width=PLOT_WIDTH, height=220))
        .opts(title=fit_title)
    )

    return hv.Layout(
        [
            _shade_windows(control_curve, windows),
            _shade_windows(signal_curve, windows),
            _shade_windows(fit_curve, windows),
        ]
    ).cols(1)
