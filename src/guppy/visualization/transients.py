import logging

import holoviews as hv
import numpy as np

from .downsampling import PLOT_WIDTH, downsample_for_display

logger = logging.getLogger(__name__)


def build_peaks_overlay(
    *, title: str, suptitle: str, z_score: np.ndarray, timestamps: np.ndarray, peaksIndex: np.ndarray
) -> hv.DynamicMap:
    """Build a z-score trace overlaid with markers at detected transient peaks.

    Parameters
    ----------
    title : str
        Trace title.
    suptitle : str
        Session-level title prefix.
    z_score : np.ndarray
        Z-score signal values.
    timestamps : np.ndarray
        Time axis values aligned to ``z_score``.
    peaksIndex : np.ndarray
        Integer indices into ``z_score`` / ``timestamps`` marking detected peaks.

    Returns
    -------
    hv.DynamicMap
        Downsampled curve of the z-score trace overlaid with a scatter of peak markers.
        Only the trace is downsampled; every detected peak keeps its marker.
    """
    curve = downsample_for_display(hv.Curve((timestamps, z_score), "time (s)", title))
    peaks = hv.Scatter((timestamps[peaksIndex], z_score[peaksIndex])).opts(color="red", size=6)
    return (curve * peaks).opts(title=f"{suptitle} — {title}", width=PLOT_WIDTH, height=300)
