import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI environment
if not os.getenv("CI"):
    plt.switch_backend("TKAgg")


def visualize_preprocessing(*, suptitle: str, title: str, x: np.ndarray, y: np.ndarray) -> tuple[Figure, Axes]:
    """Plot a preprocessing time series.

    Parameters
    ----------
    suptitle : str
        Figure-level super-title.
    title : str
        Axes title.
    x : np.ndarray
        Time axis values.
    y : np.ndarray
        Signal values to plot.

    Returns
    -------
    fig : Figure
        The created figure.
    ax : Axes
        The created axes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    fig.suptitle(suptitle)

    return fig, ax


def visualize_control_signal_fit(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    plot_name: list[str],
    name: str,
    artifacts_have_been_removed: bool,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Plot control channel, signal channel, and fitted signal in three stacked axes.

    Parameters
    ----------
    x : np.ndarray
        Time axis values shared by all three axes.
    y1 : np.ndarray
        Control channel trace (top axes).
    y2 : np.ndarray
        Signal channel trace (middle axes).
    y3 : np.ndarray
        Fitted control channel trace overlaid on signal (bottom axes).
    plot_name : list[str]
        Titles for the three axes (control, signal, fitted).
    name : str
        Figure super-title.
    artifacts_have_been_removed : bool
        When True, adds a note on the x-axis label that artifacts were removed.

    Returns
    -------
    fig : Figure
        The created figure.
    ax1, ax2, ax3 : Axes
        The three stacked axes.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    (line1,) = ax1.plot(x, y1)
    ax1.set_title(plot_name[0])
    ax2 = fig.add_subplot(312)
    (line2,) = ax2.plot(x, y2)
    ax2.set_title(plot_name[1])
    ax3 = fig.add_subplot(313)
    (line3,) = ax3.plot(x, y2)
    (line3,) = ax3.plot(x, y3)
    ax3.set_title(plot_name[2])
    fig.suptitle(name)

    hfont = {"fontname": "DejaVu Sans"}

    if artifacts_have_been_removed:
        ax3.set_xlabel("Time(s) \n Note : Artifacts have been removed, but are not reflected in this plot.", **hfont)
    else:
        ax3.set_xlabel("Time(s)", **hfont)
    return fig, ax1, ax2, ax3
