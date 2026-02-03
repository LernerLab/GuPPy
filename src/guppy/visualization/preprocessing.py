import logging
import os

import matplotlib.pyplot as plt

from ..analysis.io_utils import (
    get_all_stores_for_combining_data,  # noqa: F401 -- Necessary for other modules that depend on preprocess.py
)

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI environment
if not os.getenv("CI"):
    plt.switch_backend("TKAgg")


def visualize_preprocessing(*, suptitle, title, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    fig.suptitle(suptitle)

    return fig, ax
