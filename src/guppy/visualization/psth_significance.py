"""Display-time rendering for PSTH significance comparisons.

The estimate is drawn with its bootstrap confidence band, and the stretches whose
interval excludes zero are marked as a bar along the top of the plot. Keeping the band
rather than only the significance mask lets a reader see how close a non-significant
stretch came, and how wide the interval is where the sample size is small.
"""

import logging

import holoviews as hv
import numpy as np

logger = logging.getLogger(__name__)

ESTIMATE_COLOR = "#1f77b4"
BAND_COLOR = "#1f77b4"
SIGNIFICANT_COLOR = "#d62728"

# Fraction of the plotted vertical range the significance bar occupies.
_BAR_HEIGHT_FRACTION = 0.04


def significant_intervals(*, timestamps: np.ndarray, significant: np.ndarray) -> list[tuple[float, float]]:
    """Return the time bounds of each significant stretch.

    Parameters
    ----------
    timestamps : np.ndarray
        PSTH time axis.
    significant : np.ndarray
        Per-timepoint significance flags, as written by the significance step.

    Returns
    -------
    list of tuple of float
        ``(start, end)`` bounds in seconds, one per contiguous significant stretch.
    """
    flags = np.asarray(significant).astype(bool)
    timestamps = np.asarray(timestamps, dtype=float)

    intervals = []
    boundaries = np.diff(np.concatenate(([0], flags.view(np.int8), [0])))
    for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        intervals.append((timestamps[start], timestamps[end - 1]))

    return intervals


def build_significance_bar(
    *, timestamps: np.ndarray, significant: np.ndarray, bottom: float, top: float
) -> hv.Rectangles:
    """Draw the significant stretches as a bar spanning the top of the plot.

    Parameters
    ----------
    timestamps : np.ndarray
        PSTH time axis.
    significant : np.ndarray
        Per-timepoint significance flags.
    bottom, top : float
        Vertical range the bar is placed against.

    Returns
    -------
    hv.Rectangles
        One rectangle per significant stretch, empty when nothing is significant.
    """
    height = (top - bottom) * _BAR_HEIGHT_FRACTION
    bars = [
        (start, top - height, end, top)
        for start, end in significant_intervals(timestamps=timestamps, significant=significant)
    ]

    return hv.Rectangles(bars, kdims=["Time (s)", "value", "Time (s) end", "value end"])


def build_significance_panel(
    *,
    timestamps: np.ndarray,
    estimate: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    significant: np.ndarray,
    value_label: str,
    estimate_label: str,
    significance_level: float,
    title: str,
) -> hv.Overlay:
    """Compose the estimate, its confidence band, and the significance bar.

    Parameters
    ----------
    timestamps : np.ndarray
        PSTH time axis.
    estimate : np.ndarray
        Mean PSTH, or the difference between two mean PSTHs.
    ci_lower, ci_upper : np.ndarray
        Bootstrap confidence bounds on ``estimate``.
    significant : np.ndarray
        Per-timepoint significance flags.
    value_label : str
        Y-axis label.
    estimate_label : str
        Legend entry for the estimate curve.
    significance_level : float
        Alpha the interval was computed at, named in the legend.
    title : str
        Plot title naming the comparison.

    Returns
    -------
    hv.Overlay
        The band, the estimate, a zero reference line, and the significance bar.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)

    bottom = float(np.nanmin(ci_lower))
    top = float(np.nanmax(ci_upper))
    # Headroom so the significance bar does not sit on top of the band itself.
    top = top + (top - bottom) * (_BAR_HEIGHT_FRACTION * 2)

    band = hv.Spread(
        (timestamps, estimate, estimate - ci_lower, ci_upper - estimate),
        kdims=["Time (s)"],
        vdims=[value_label, "lower", "upper"],
        label=f"{int(round((1 - significance_level) * 100))}% confidence interval",
    ).opts(fill_alpha=0.3, fill_color=BAND_COLOR, line_width=0, show_legend=True)

    estimate_curve = hv.Curve(
        (timestamps, estimate), kdims=["Time (s)"], vdims=[value_label], label=estimate_label
    ).opts(color=ESTIMATE_COLOR, show_legend=True)

    zero_line = hv.HLine(0).opts(color="black", line_dash="dashed", line_width=1)

    bar = (
        build_significance_bar(timestamps=timestamps, significant=significant, bottom=bottom, top=top)
        .relabel(f"significant (alpha = {significance_level:g})")
        .opts(color=SIGNIFICANT_COLOR, line_color=None, show_legend=True)
    )

    return (band * estimate_curve * zero_line * bar).opts(
        title=title, ylim=(bottom, top), responsive=True, height=400, legend_position="top_left"
    )
