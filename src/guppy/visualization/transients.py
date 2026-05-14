import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def visualize_peaks(
    title: str, suptitle: str, z_score: np.ndarray, timestamps: np.ndarray, peaksIndex: np.ndarray
) -> tuple[Figure, Axes]:
    """Plot a z-score trace with detected transient peaks overlaid.

    Parameters
    ----------
    title : str
        Axes title.
    suptitle : str
        Figure-level super-title.
    z_score : np.ndarray
        Z-score signal values.
    timestamps : np.ndarray
        Time axis values aligned to z_score.
    peaksIndex : np.ndarray
        Integer indices into z_score and timestamps marking detected peaks.

    Returns
    -------
    fig : Figure
        The created figure.
    ax : Axes
        The created axes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(timestamps, z_score, "-", timestamps[peaksIndex], z_score[peaksIndex], "o")
    ax.set_title(title)
    fig.suptitle(suptitle)

    return fig, ax
