"""Display-time rendering for behavioral covariates against binned photometry metrics.

Two views of the same numbers: the covariate's own trace with the per-bin means it was
reduced to, drawn on the ``time (s)`` dimension so it stacks and links with the photometry
panels of :mod:`guppy.visualization.binned_metrics`, and a scatter of one against the other
with a point per bin.
"""

import logging

import holoviews as hv
import numpy as np

from .binned_metrics import build_bin_bars
from .shading import FIT_COLOR

logger = logging.getLogger(__name__)

COVARIATE_COLOR = "#9467bd"
POINT_COLOR = "#1f77b4"


def build_covariate_panel(
    *,
    covariate_timestamps: np.ndarray,
    covariate_values: np.ndarray,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    binned_values: np.ndarray,
    covariate_label: str,
    panel_height: int = 200,
) -> hv.Layout:
    """Stack a covariate's per-bin means under the raw series they were averaged from.

    The raw series is drawn as a marked line rather than a shaded density image: a
    hand-scored covariate carries a few samples per session, so every sample should be
    visible as a sample.

    Parameters
    ----------
    covariate_timestamps, covariate_values : np.ndarray
        The covariate as it was recorded, in seconds and its own units.
    bin_starts, bin_ends : np.ndarray
        Bin bounds in seconds, matching the photometry bins.
    binned_values : np.ndarray
        Per-bin covariate mean, one per bin.
    covariate_label : str
        Covariate name, used in the axis labels and titles.
    panel_height : int, optional
        Height in pixels of each of the two panels.

    Returns
    -------
    hv.Layout
        The raw covariate above, its per-bin means below, on a ``time (s)`` axis shared
        with the photometry panels.
    """
    covariate_timestamps = np.asarray(covariate_timestamps, dtype=float).ravel()
    covariate_values = np.asarray(covariate_values, dtype=float).ravel()

    raw_curve = hv.Curve((covariate_timestamps, covariate_values), "time (s)", covariate_label)
    raw_panel = (
        raw_curve.opts(color=COVARIATE_COLOR) * hv.Scatter(raw_curve).opts(color=COVARIATE_COLOR, size=5)
    ).opts(responsive=True, height=panel_height, title=covariate_label)
    binned_bars = build_bin_bars(
        bin_starts=bin_starts,
        bin_ends=bin_ends,
        values=binned_values,
        value_label=f"mean {covariate_label}",
    ).opts(
        color=COVARIATE_COLOR,
        line_color="white",
        responsive=True,
        height=panel_height,
        title=f"mean {covariate_label}",
    )

    return hv.Layout([raw_panel, binned_bars]).cols(1)


def build_covariate_scatter(
    *,
    covariate_values: np.ndarray,
    metric_values: np.ndarray,
    covariate_label: str,
    metric_label: str,
    pearson_r: float,
    n_bins: int,
    suptitle: str,
) -> hv.Overlay:
    """Scatter a per-bin photometry metric against a behavioral covariate.

    Bins where either value is missing are dropped, so the point count matches the
    ``n_bins`` reported alongside the coefficient. A least-squares line is drawn under
    the points, which is Pearson r made visible: the two are the same fit, so the line
    is only as steep as the reported coefficient says. It is omitted when the fit is
    undefined, which is the same condition that leaves ``pearson_r`` NaN.

    Parameters
    ----------
    covariate_values, metric_values : np.ndarray
        Per-bin covariate means and metric values, same length.
    covariate_label, metric_label : str
        Axis labels.
    pearson_r : float
        Coefficient shown in the title.
    n_bins : int
        Number of bins behind the coefficient.
    suptitle : str
        Prefix for the title, typically the recording site.

    Returns
    -------
    hv.Overlay
        The least-squares line under one point per usable bin.
    """
    covariate_values = np.asarray(covariate_values, dtype=float).ravel()
    metric_values = np.asarray(metric_values, dtype=float).ravel()

    usable = ~np.isnan(covariate_values) & ~np.isnan(metric_values)
    covariate_values = covariate_values[usable]
    metric_values = metric_values[usable]

    points = hv.Points((covariate_values, metric_values), kdims=[covariate_label, metric_label]).opts(
        color=POINT_COLOR, size=7
    )
    fit_line = hv.Curve(
        _least_squares_line(covariate_values=covariate_values, metric_values=metric_values),
        covariate_label,
        metric_label,
    ).opts(color=FIT_COLOR, line_width=1.5, alpha=0.8)

    return (fit_line * points).opts(
        responsive=True,
        height=380,
        title=f"{suptitle} — r = {pearson_r:.2f}, n = {n_bins} bins",
    )


def _least_squares_line(*, covariate_values: np.ndarray, metric_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fitted line across the covariate range, empty where the fit is undefined.

    A constant covariate has no slope, and a single point has no line, so both return
    nothing to draw rather than a horizontal line that would read as a real fit.
    """
    if len(np.unique(covariate_values)) < 2:
        return np.array([]), np.array([])

    slope, intercept = np.polyfit(covariate_values, metric_values, 1)
    endpoints = np.array([covariate_values.min(), covariate_values.max()])

    return endpoints, slope * endpoints + intercept
