import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def visualize_peaks(title, suptitle, z_score, timestamps, peaksIndex):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(timestamps, z_score, "-", timestamps[peaksIndex], z_score[peaksIndex], "o")
    ax.set_title(title)
    fig.suptitle(suptitle)

    return fig, ax
