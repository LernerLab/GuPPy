import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from ..visualization.preprocessing import visualize_control_signal_fit

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI environment
if not os.getenv("CI"):
    plt.switch_backend("TKAgg")


class ArtifactRemovalWidget:

    def __init__(self, filepath, x, y1, y2, y3, plot_name, removeArtifacts):
        self.coords = []  # List to store selected coordinates
        self.filepath = filepath
        self.plot_name = plot_name

        if (y1 == 0).all() == True:
            y1 = np.zeros(x.shape[0])

        coords_path = os.path.join(filepath, "coordsForPreProcessing_" + plot_name[0].split("_")[-1] + ".npy")
        artifacts_have_been_removed = removeArtifacts and os.path.exists(coords_path)
        name = os.path.basename(filepath)
        self.fig, self.ax1, self.ax2, self.ax3 = visualize_control_signal_fit(
            x, y1, y2, y3, plot_name, name, artifacts_have_been_removed
        )

        self.cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_key_press(self, event):
        """Handle key press events for artifact selection.

        Pressing 'space' draws a vertical line at the cursor position to mark artifact boundaries.
        Pressing 'd' removes the most recently added line.
        """
        if event.key == " ":
            ix, iy = event.xdata, event.ydata
            logger.info(f"x = {ix}, y = {iy}")
            self.ax1.axvline(ix, c="black", ls="--")
            self.ax2.axvline(ix, c="black", ls="--")
            self.ax3.axvline(ix, c="black", ls="--")

            self.fig.canvas.draw()

            self.coords.append((ix, iy))

            return self.coords

        elif event.key == "d":
            if len(self.coords) > 0:
                logger.info(f"x = {self.coords[-1][0]}, y = {self.coords[-1][1]}; deleted")
                del self.coords[-1]
                self.ax1.lines[-1].remove()
                self.ax2.lines[-1].remove()
                self.ax3.lines[-1].remove()
                self.fig.canvas.draw()

            return self.coords

    def _on_close(self, _event):
        """Handle figure close event by saving coordinates and cleaning up."""
        if self.coords and len(self.coords) > 0:
            name_1 = self.plot_name[0].split("_")[-1]
            np.save(os.path.join(self.filepath, "coordsForPreProcessing_" + name_1 + ".npy"), self.coords)
            logger.info(
                f"Coordinates file saved at {os.path.join(self.filepath, 'coordsForPreProcessing_'+name_1+'.npy')}"
            )
        self.fig.canvas.mpl_disconnect(self.cid)
        self.coords = []
