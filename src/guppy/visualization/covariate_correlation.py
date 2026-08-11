"""Scatter of one per-bin photometry metric against one behavioral covariate.

Each point is a time bin. Points are colored by bin number so the passage of the
session is visible in the cloud, which is what separates a bin-by-bin relationship
from two series that merely drift together.
"""

import logging

import holoviews as hv
import numpy as np

logger = logging.getLogger(__name__)

POINT_CMAP = "viridis"


def build_covariate_scatter(
    *,
    covariate_values: np.ndarray,
    metric_values: np.ndarray,
    bin_numbers: np.ndarray,
    covariate_label: str,
    metric_label: str,
    pearson_r: float,
    spearman_rho: float,
    n_bins: int,
    suptitle: str,
) -> hv.Points:
    """Scatter a per-bin photometry metric against a behavioral covariate.

    Bins where either value is missing are dropped, so the point count matches the
    ``n_bins`` reported alongside the coefficients.

    Parameters
    ----------
    covariate_values, metric_values : np.ndarray
        Per-bin covariate means and metric values, same length.
    bin_numbers : np.ndarray
        Bin index for each row, used to color the points.
    covariate_label, metric_label : str
        Axis labels.
    pearson_r, spearman_rho : float
        Coefficients shown in the title.
    n_bins : int
        Number of bins behind the coefficients.
    suptitle : str
        Prefix for the title, typically the recording site.

    Returns
    -------
    hv.Points
        One point per usable bin.
    """
    covariate_values = np.asarray(covariate_values, dtype=float).ravel()
    metric_values = np.asarray(metric_values, dtype=float).ravel()
    bin_numbers = np.asarray(bin_numbers).ravel()

    usable = ~np.isnan(covariate_values) & ~np.isnan(metric_values)

    return hv.Points(
        (covariate_values[usable], metric_values[usable], bin_numbers[usable]),
        kdims=[covariate_label, metric_label],
        vdims=["bin"],
    ).opts(
        color="bin",
        cmap=POINT_CMAP,
        colorbar=True,
        size=7,
        responsive=True,
        height=380,
        title=f"{suptitle} — r = {pearson_r:.2f}, rho = {spearman_rho:.2f}, n = {n_bins} bins",
    )
