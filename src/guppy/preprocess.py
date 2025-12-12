import glob
import json
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from .analysis.artifact_removal import (
    addingNaNtoChunksWithArtifacts,
    processTimestampsForArtifacts,
)
from .analysis.combine_data import combineData
from .analysis.io_utils import (
    check_storeslistfile,
    check_TDT,
    find_files,
    get_all_stores_for_combining_data,  # noqa: F401 -- Necessary for other modules that depend on preprocess.py
    read_hdf5,
    takeOnlyDirs,
)
from .analysis.timestamp_correction import (
    create_control_channel,
    decide_naming_convention_and_applyCorrection,
    read_control_and_signal,
    timestampCorrection,
    write_corrected_timestamps,
)
from .analysis.z_score import compute_z_score

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI environment
if not os.getenv("CI"):
    plt.switch_backend("TKAgg")


def writeToFile(value: str):
    with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
        file.write(value)


# function to plot z_score
def visualize_z_score(filepath):

    name = os.path.basename(filepath)

    path = glob.glob(os.path.join(filepath, "z_score_*"))

    path = sorted(path)

    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = basename.split("_")[-1]
        x = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        y = read_hdf5("", path[i], "data")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(basename)
        fig.suptitle(name)
    # plt.show()


# function to plot deltaF/F
def visualize_dff(filepath):
    name = os.path.basename(filepath)

    path = glob.glob(os.path.join(filepath, "dff_*"))

    path = sorted(path)

    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = basename.split("_")[-1]
        x = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        y = read_hdf5("", path[i], "data")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(basename)
        fig.suptitle(name)
    # plt.show()


def visualize(filepath, x, y1, y2, y3, plot_name, removeArtifacts):

    # plotting control and signal data

    if (y1 == 0).all() == True:
        y1 = np.zeros(x.shape[0])

    coords_path = os.path.join(filepath, "coordsForPreProcessing_" + plot_name[0].split("_")[-1] + ".npy")
    name = os.path.basename(filepath)
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

    if removeArtifacts == True and os.path.exists(coords_path):
        ax3.set_xlabel("Time(s) \n Note : Artifacts have been removed, but are not reflected in this plot.", **hfont)
    else:
        ax3.set_xlabel("Time(s)", **hfont)

    global coords
    coords = []

    # clicking 'space' key on keyboard will draw a line on the plot so that user can see what chunks are selected
    # and clicking 'd' key on keyboard will deselect the selected point
    def onclick(event):
        # global ix, iy

        if event.key == " ":
            ix, iy = event.xdata, event.ydata
            logger.info(f"x = {ix}, y = {iy}")
            y1_max, y1_min = np.amax(y1), np.amin(y1)
            y2_max, y2_min = np.amax(y2), np.amin(y2)

            # ax1.plot([ix,ix], [y1_max, y1_min], 'k--')
            # ax2.plot([ix,ix], [y2_max, y2_min], 'k--')

            ax1.axvline(ix, c="black", ls="--")
            ax2.axvline(ix, c="black", ls="--")
            ax3.axvline(ix, c="black", ls="--")

            fig.canvas.draw()

            global coords
            coords.append((ix, iy))

            # if len(coords) == 2:
            #    fig.canvas.mpl_disconnect(cid)

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
    # multi = MultiCursor(fig.canvas, (ax1, ax2), color='g', lw=1, horizOn=False, vertOn=True)

    # plt.show()
    # return fig


# function to plot control and signal, also provide a feature to select chunks for artifacts removal
def visualizeControlAndSignal(filepath, removeArtifacts):
    path_1 = find_files(filepath, "control_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'control*'))

    path_2 = find_files(filepath, "signal_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'signal*'))

    path = sorted(path_1 + path_2, key=str.casefold)

    if len(path) % 2 != 0:
        logger.error("There are not equal number of Control and Signal data")
        raise Exception("There are not equal number of Control and Signal data")

    path = np.asarray(path).reshape(2, -1)

    for i in range(path.shape[1]):

        name_1 = ((os.path.basename(path[0, i])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, i])).split(".")[0]).split("_")

        ts_path = os.path.join(filepath, "timeCorrection_" + name_1[-1] + ".hdf5")
        cntrl_sig_fit_path = os.path.join(filepath, "cntrl_sig_fit_" + name_1[-1] + ".hdf5")
        ts = read_hdf5("", ts_path, "timestampNew")

        control = read_hdf5("", path[0, i], "data").reshape(-1)
        signal = read_hdf5("", path[1, i], "data").reshape(-1)
        cntrl_sig_fit = read_hdf5("", cntrl_sig_fit_path, "data").reshape(-1)

        plot_name = [
            (os.path.basename(path[0, i])).split(".")[0],
            (os.path.basename(path[1, i])).split(".")[0],
            (os.path.basename(cntrl_sig_fit_path)).split(".")[0],
        ]
        visualize(filepath, ts, control, signal, cntrl_sig_fit, plot_name, removeArtifacts)


# This function just creates placeholder Control-HDF5 files that are then immediately overwritten later on in the pipeline.
# TODO: Refactor this function to avoid unnecessary file creation.
# function to add control channel when there is no
# isosbestic control channel and update the storeslist file
def add_control_channel(filepath, arr):

    storenames = arr[0, :]
    storesList = np.char.lower(arr[1, :])

    keep_control = np.array([])
    # check a case if there is isosbestic control channel present
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower():
            name = storesList[i].split("_")[-1]
            new_str = "signal_" + str(name).lower()
            find_signal = [True for i in storesList if i == new_str]
            if len(find_signal) > 1:
                logger.error("Error in naming convention of files or Error in storesList file")
                raise Exception("Error in naming convention of files or Error in storesList file")
            if len(find_signal) == 0:
                logger.error(
                    "Isosbectic control channel parameter is set to False and still \
							 	 storeslist file shows there is control channel present"
                )
                raise Exception(
                    "Isosbectic control channel parameter is set to False and still \
							 	 storeslist file shows there is control channel present"
                )
        else:
            continue

    for i in range(storesList.shape[0]):
        if "signal" in storesList[i].lower():
            name = storesList[i].split("_")[-1]
            new_str = "control_" + str(name).lower()
            find_signal = [True for i in storesList if i == new_str]
            if len(find_signal) == 0:
                src, dst = os.path.join(filepath, arr[0, i] + ".hdf5"), os.path.join(
                    filepath, "cntrl" + str(i) + ".hdf5"
                )
                shutil.copyfile(src, dst)
                arr = np.concatenate((arr, [["cntrl" + str(i)], ["control_" + str(arr[1, i].split("_")[-1])]]), axis=1)

    np.savetxt(os.path.join(filepath, "storesList.csv"), arr, delimiter=",", fmt="%s")

    return arr


# function to execute timestamps corrections using functions timestampCorrection and decide_naming_convention_and_applyCorrection
def execute_timestamp_correction(folderNames, inputParameters):

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    isosbestic_control = inputParameters["isosbestic_control"]

    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        mode = "tdt" if check_TDT(folderNames[i]) else "csv"
        logger.debug(f"Timestamps corrections started for {filepath}")
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )

            if isosbestic_control == False:
                storesList = add_control_channel(filepath, storesList)

            control_and_signal_dicts = read_control_and_signal(filepath, storesList)
            name_to_data, name_to_timestamps, name_to_sampling_rate, name_to_npoints = control_and_signal_dicts
            corrected_name_to_timestamps, name_to_correctionIndex = timestampCorrection(
                timeForLightsTurnOn,
                storesList,
                name_to_timestamps,
                name_to_data,
                name_to_sampling_rate,
                name_to_npoints,
                mode=mode,
            )
            write_corrected_timestamps(
                filepath,
                corrected_name_to_timestamps,
                name_to_timestamps,
                name_to_sampling_rate,
                name_to_correctionIndex,
            )

            for k in range(storesList.shape[1]):
                decide_naming_convention_and_applyCorrection(
                    filepath, timeForLightsTurnOn, storesList[0, k], storesList[1, k], storesList
                )

            # check if isosbestic control is false and also if new control channel is added
            if isosbestic_control == False:
                create_control_channel(filepath, storesList, window=101)

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
            inputParameters["step"] += 1
        logger.info(f"Timestamps corrections finished for {filepath}")


# function to compute z-score and deltaF/F using functions : compute_z_score and/or processTimestampsForArtifacts
def execute_zscore(folderNames, inputParameters):

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    remove_artifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    plot_zScore_dff = inputParameters["plot_zScore_dff"]
    combine_data = inputParameters["combine_data"]
    isosbestic_control = inputParameters["isosbestic_control"]

    storesListPath = []
    for i in range(len(folderNames)):
        if combine_data == True:
            storesListPath.append([folderNames[i][0]])
        else:
            filepath = folderNames[i]
            storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))

    storesListPath = np.concatenate(storesListPath)

    for j in range(len(storesListPath)):
        filepath = storesListPath[j]
        storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)

        if remove_artifacts == True:
            logger.debug("Removing Artifacts from the data and correcting timestamps...")
            compute_z_score(filepath, inputParameters)
            if artifactsRemovalMethod == "concatenate":
                processTimestampsForArtifacts(filepath, timeForLightsTurnOn, storesList)
            else:
                addingNaNtoChunksWithArtifacts(filepath, storesList)
            visualizeControlAndSignal(filepath, remove_artifacts)
            logger.info("Artifacts from the data are removed and timestamps are corrected.")
        else:
            compute_z_score(filepath, inputParameters)
            visualizeControlAndSignal(filepath, remove_artifacts)

        if plot_zScore_dff == "z_score":
            visualize_z_score(filepath)
        if plot_zScore_dff == "dff":
            visualize_dff(filepath)
        if plot_zScore_dff == "Both":
            visualize_z_score(filepath)
            visualize_dff(filepath)

        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1

    plt.show()
    logger.info("Signal data and event timestamps are extracted.")


def extractTsAndSignal(inputParameters):

    logger.debug("Extracting signal data and event timestamps...")
    inputParameters = inputParameters

    # storesList = np.genfromtxt(inputParameters['storesListPath'], dtype='str', delimiter=',')

    folderNames = inputParameters["folderNames"]
    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    isosbestic_control = inputParameters["isosbestic_control"]
    remove_artifacts = inputParameters["removeArtifacts"]
    combine_data = inputParameters["combine_data"]

    inputParameters["step"] = 0
    logger.info(f"Remove Artifacts : {remove_artifacts}")
    logger.info(f"Combine Data : {combine_data}")
    logger.info(f"Isosbestic Control Channel : {isosbestic_control}")
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    # pbMaxValue = storesListPath.shape[0] + len(folderNames)
    # writeToFile(str((pbMaxValue+1)*10)+'\n'+str(10)+'\n')
    if combine_data == False:
        pbMaxValue = storesListPath.shape[0] + len(folderNames)
        writeToFile(str((pbMaxValue + 1) * 10) + "\n" + str(10) + "\n")
        execute_timestamp_correction(folderNames, inputParameters)
        execute_zscore(folderNames, inputParameters)
    else:
        pbMaxValue = 1 + len(folderNames)
        writeToFile(str((pbMaxValue) * 10) + "\n" + str(10) + "\n")
        execute_timestamp_correction(folderNames, inputParameters)
        storesList = check_storeslistfile(folderNames)
        op_folder = combineData(folderNames, inputParameters, storesList)
        execute_zscore(op_folder, inputParameters)


def main(input_parameters):
    try:
        extractTsAndSignal(input_parameters)
        logger.info("#" * 400)
    except Exception as e:
        with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
            file.write(str(-1) + "\n")
        logger.error(str(e))
        raise e


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
