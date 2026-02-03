import logging
import os

import matplotlib.pyplot as plt
import numpy as np

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


def visualize(filepath, x, y1, y2, y3, plot_name, removeArtifacts):

    # plotting control and signal data

    if (y1 == 0).all() == True:
        y1 = np.zeros(x.shape[0])

    coords_path = os.path.join(filepath, "coordsForPreProcessing_" + plot_name[0].split("_")[-1] + ".npy")
    artifacts_have_been_removed = removeArtifacts and os.path.exists(coords_path)
    name = os.path.basename(filepath)
    fig, ax1, ax2, ax3 = visualize_control_signal_fit(x, y1, y2, y3, plot_name, name, artifacts_have_been_removed)

    global coords
    coords = []

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

            global coords
            coords.append((ix, iy))

            return coords

        elif event.key == "d":
            if len(coords) > 0:
                logger.info(f"x = {coords[-1][0]}, y = {coords[-1][1]}; deleted")
                del coords[-1]
                ax1.lines[-1].remove()
                ax2.lines[-1].remove()
                ax3.lines[-1].remove()
                fig.canvas.draw()

            return coords

    # close the plot will save coordinates for all the selected chunks in the data
    def plt_close_event(event):
        global coords
        if coords and len(coords) > 0:
            name_1 = plot_name[0].split("_")[-1]
            np.save(os.path.join(filepath, "coordsForPreProcessing_" + name_1 + ".npy"), coords)
            logger.info(f"Coordinates file saved at {os.path.join(filepath, 'coordsForPreProcessing_'+name_1+'.npy')}")
        fig.canvas.mpl_disconnect(cid)
        coords = []

    cid = fig.canvas.mpl_connect("key_press_event", onclick)
    cid = fig.canvas.mpl_connect("close_event", plt_close_event)


def visualize_control_signal_fit(x, y1, y2, y3, plot_name, name, artifacts_have_been_removed):
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
