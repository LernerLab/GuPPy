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

        if (y1 == 0).all() == True:
            y1 = np.zeros(x.shape[0])

        coords_path = os.path.join(filepath, "coordsForPreProcessing_" + plot_name[0].split("_")[-1] + ".npy")
        artifacts_have_been_removed = removeArtifacts and os.path.exists(coords_path)
        name = os.path.basename(filepath)
        fig, ax1, ax2, ax3 = visualize_control_signal_fit(x, y1, y2, y3, plot_name, name, artifacts_have_been_removed)

        # clicking 'space' key on keyboard will draw a line on the plot so that user can see what chunks are selected
        # and clicking 'd' key on keyboard will deselect the selected point
        def onclick(event):
            if event.key == " ":
                ix, iy = event.xdata, event.ydata
                logger.info(f"x = {ix}, y = {iy}")
                ax1.axvline(ix, c="black", ls="--")
                ax2.axvline(ix, c="black", ls="--")
                ax3.axvline(ix, c="black", ls="--")

                fig.canvas.draw()

                self.coords.append((ix, iy))

                return self.coords

            elif event.key == "d":
                if len(self.coords) > 0:
                    logger.info(f"x = {self.coords[-1][0]}, y = {self.coords[-1][1]}; deleted")
                    del self.coords[-1]
                    ax1.lines[-1].remove()
                    ax2.lines[-1].remove()
                    ax3.lines[-1].remove()
                    fig.canvas.draw()

                return self.coords

        # close the plot will save coordinates for all the selected chunks in the data
        def plt_close_event(event):
            if self.coords and len(self.coords) > 0:
                name_1 = plot_name[0].split("_")[-1]
                np.save(os.path.join(filepath, "coordsForPreProcessing_" + name_1 + ".npy"), self.coords)
                logger.info(
                    f"Coordinates file saved at {os.path.join(filepath, 'coordsForPreProcessing_'+name_1+'.npy')}"
                )
            fig.canvas.mpl_disconnect(cid)
            self.coords = []

        cid = fig.canvas.mpl_connect("key_press_event", onclick)
        cid = fig.canvas.mpl_connect("close_event", plt_close_event)
