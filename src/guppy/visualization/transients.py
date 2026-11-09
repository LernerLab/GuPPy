import logging

import holoviews as hv
import numpy as np

from .palette import HIGHLIGHT
from .shading import shade_trace

logger = logging.getLogger(__name__)


def build_peaks_overlay(
    *, title: str, z_score: np.ndarray, timestamps: np.ndarray, peaksIndex: np.ndarray
) -> hv.DynamicMap:
    """Build a z-score trace overlaid with markers at detected transient peaks.

    Parameters
    ----------
    title : str
        Trace title.
    z_score : np.ndarray
        Z-score signal values.
    timestamps : np.ndarray
        Time axis values aligned to ``z_score``.
    peaksIndex : np.ndarray
        Integer indices into ``z_score`` / ``timestamps`` marking detected peaks.

    Returns
    -------
    hv.DynamicMap
        Density-shaded z-score trace overlaid with a scatter of peak markers.
        Only the trace is shaded; every detected peak keeps its marker.
    """
    curve = shade_trace(hv.Curve((timestamps, z_score), "time (s)", title))
    peaks = hv.Scatter((timestamps[peaksIndex], z_score[peaksIndex])).opts(color=HIGHLIGHT, size=6)
    return (curve * peaks).opts(title=title, responsive=True, height=300)
