import glob
import json
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis.analysis import (
    addingNaNValues,
    check_cntrl_sig_length,
    eliminateData,
    eliminateTs,
    execute_controlFit_dff,
    helper_create_control_channel,
    removeTTLs,
    z_score_computation,
)
from .analysis.io_utils import (
    check_TDT,
    decide_naming_convention,
    fetchCoords,
    find_files,
    get_all_stores_for_combining_data,
    read_hdf5,
    takeOnlyDirs,
    write_hdf5,
)
from .combineDataFn import processTimestampsForCombiningData

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI environment
if not os.getenv("CI"):
    plt.switch_backend("TKAgg")


# Category: Visualization/User Input
# Reason: Writes progress updates to file for GUI progress bar - couples backend to GUI feedback mechanism
def writeToFile(value: str):
    with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
        file.write(value)


# Category: Routing
# Reason: Orchestrates reading HDF5 files, calling helper_create_control_channel, and writing results - coordinates I/O with computation
# main function to create control channel using
# signal channel and save it to a file
def create_control_channel(filepath, arr, window=5001):

    storenames = arr[0, :]
    storesList = arr[1, :]

    for i in range(storesList.shape[0]):
        event_name, event = storesList[i], storenames[i]
        if "control" in event_name.lower() and "cntrl" in event.lower():
            logger.debug("Creating control channel from signal channel using curve-fitting")
            name = event_name.split("_")[-1]
            signal = read_hdf5("signal_" + name, filepath, "data")
            timestampNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
            sampling_rate = np.full(timestampNew.shape, np.nan)
            sampling_rate[0] = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            control = helper_create_control_channel(signal, timestampNew, window)

            write_hdf5(control, event_name, filepath, "data")
            d = {"timestamps": timestampNew, "data": control, "sampling_rate": sampling_rate}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(os.path.dirname(filepath), event.lower() + ".csv"), index=False)
            logger.info("Control channel from signal channel created using curve-fitting")


# Category: Routing
# Reason: Orchestrates validation logic, file copying, and storesList updates - coordinates multiple operations and file manipulations
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


# Category: Routing
# Reason: Orchestrates timestamp correction workflow - loops through stores, coordinates reading/writing, calls validation and correction logic
# function to correct timestamps after eliminating first few seconds of the data (for csv data)
def timestampCorrection_csv(filepath, timeForLightsTurnOn, storesList):

    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    storenames = storesList[0, :]
    storesList = storesList[1, :]

    arr = []
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
            arr.append(storesList[i])

    arr = sorted(arr, key=str.casefold)
    try:
        arr = np.asarray(arr).reshape(2, -1)
    except:
        logger.error("Error in saving stores list file or spelling mistake for control or signal")
        raise Exception("Error in saving stores list file or spelling mistake for control or signal")

    indices = check_cntrl_sig_length(filepath, arr, storenames, storesList)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        idx = np.where(storesList == indices[i])[0]

        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        timestamp = read_hdf5(storenames[idx][0], filepath, "timestamps")
        sampling_rate = read_hdf5(storenames[idx][0], filepath, "sampling_rate")

        if name_1 == name_2:
            correctionIndex = np.where(timestamp >= timeForLightsTurnOn)[0]
            timestampNew = timestamp[correctionIndex]
            write_hdf5(timestampNew, "timeCorrection_" + name_1, filepath, "timestampNew")
            write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
            write_hdf5(np.asarray(sampling_rate), "timeCorrection_" + name_1, filepath, "sampling_rate")

        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrected and converted to seconds.")


# Category: Routing
# Reason: Orchestrates timestamp correction workflow for TDT format - loops through stores, coordinates timestamp expansion algorithm with I/O
# function to correct timestamps after eliminating first few seconds of the data (for TDT data)
def timestampCorrection_tdt(filepath, timeForLightsTurnOn, storesList):

    logger.debug(
        f"Correcting timestamps by getting rid of the first {timeForLightsTurnOn} seconds and convert timestamps to seconds"
    )
    storenames = storesList[0, :]
    storesList = storesList[1, :]

    arr = []
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
            arr.append(storesList[i])

    arr = sorted(arr, key=str.casefold)

    try:
        arr = np.asarray(arr).reshape(2, -1)
    except:
        logger.error("Error in saving stores list file or spelling mistake for control or signal")
        raise Exception("Error in saving stores list file or spelling mistake for control or signal")

    indices = check_cntrl_sig_length(filepath, arr, storenames, storesList)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        idx = np.where(storesList == indices[i])[0]

        if idx.shape[0] == 0:
            logger.error(f"{arr[0,i]} does not exist in the stores list file.")
            raise Exception("{} does not exist in the stores list file.".format(arr[0, i]))

        timestamp = read_hdf5(storenames[idx][0], filepath, "timestamps")
        npoints = read_hdf5(storenames[idx][0], filepath, "npoints")
        sampling_rate = read_hdf5(storenames[idx][0], filepath, "sampling_rate")

        if name_1 == name_2:
            timeRecStart = timestamp[0]
            timestamps = np.subtract(timestamp, timeRecStart)
            adder = np.arange(npoints) / sampling_rate
            lengthAdder = adder.shape[0]
            timestampNew = np.zeros((len(timestamps), lengthAdder))
            for i in range(lengthAdder):
                timestampNew[:, i] = np.add(timestamps, adder[i])
            timestampNew = (timestampNew.T).reshape(-1, order="F")
            correctionIndex = np.where(timestampNew >= timeForLightsTurnOn)[0]
            timestampNew = timestampNew[correctionIndex]

            write_hdf5(np.asarray([timeRecStart]), "timeCorrection_" + name_1, filepath, "timeRecStart")
            write_hdf5(timestampNew, "timeCorrection_" + name_1, filepath, "timestampNew")
            write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
            write_hdf5(np.asarray([sampling_rate]), "timeCorrection_" + name_1, filepath, "sampling_rate")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrected and converted to seconds.")
    # return timeRecStart, correctionIndex, timestampNew


# Category: Routing
# Reason: Orchestrates applying timestamp corrections - reads correction indices, applies different logic based on data type, writes results
# function to apply correction to control, signal and event timestamps
def applyCorrection(filepath, timeForLightsTurnOn, event, displayName, naming):

    cond = check_TDT(os.path.dirname(filepath))

    if cond == True:
        timeRecStart = read_hdf5("timeCorrection_" + naming, filepath, "timeRecStart")[0]

    timestampNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    correctionIndex = read_hdf5("timeCorrection_" + naming, filepath, "correctionIndex")

    if "control" in displayName.lower() or "signal" in displayName.lower():
        split_name = displayName.split("_")[-1]
        if split_name == naming:
            pass
        else:
            correctionIndex = read_hdf5("timeCorrection_" + split_name, filepath, "correctionIndex")
        arr = read_hdf5(event, filepath, "data")
        if (arr == 0).all() == True:
            arr = arr
        else:
            arr = arr[correctionIndex]
        write_hdf5(arr, displayName, filepath, "data")
    else:
        arr = read_hdf5(event, filepath, "timestamps")
        if cond == True:
            res = (arr >= timeRecStart).all()
            if res == True:
                arr = np.subtract(arr, timeRecStart)
                arr = np.subtract(arr, timeForLightsTurnOn)
            else:
                arr = np.subtract(arr, timeForLightsTurnOn)
        else:
            arr = np.subtract(arr, timeForLightsTurnOn)
        write_hdf5(arr, displayName + "_" + naming, filepath, "ts")

    # if isosbestic_control==False and 'control' in displayName.lower():
    # 	control = create_control_channel(filepath, displayName)
    # 	write_hdf5(control, displayName, filepath, 'data')


# Category: Routing
# Reason: Orchestrates naming validation and correction application - loops through channel pairs and delegates to applyCorrection
# function to check if naming convention was followed while saving storeslist file
# and apply timestamps correction using the function applyCorrection
def decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, event, displayName, storesList):

    logger.debug("Applying correction of timestamps to the data and event timestamps")
    storesList = storesList[1, :]

    arr = []
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
            arr.append(storesList[i])

    arr = sorted(arr, key=str.casefold)
    arr = np.asarray(arr).reshape(2, -1)

    for i in range(arr.shape[1]):
        name_1 = arr[0, i].split("_")[-1]
        name_2 = arr[1, i].split("_")[-1]
        # dirname = os.path.dirname(path[i])
        if name_1 == name_2:
            applyCorrection(filepath, timeForLightsTurnOn, event, displayName, name_1)
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info("Timestamps corrections applied to the data and event timestamps.")


# Category: Visualization/User Input
# Reason: Creates matplotlib plots to display z-score results - pure visualization with no computation
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


# Category: Visualization/User Input
# Reason: Creates matplotlib plots to display deltaF/F results - pure visualization with no computation
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


# Category: Visualization/User Input
# Reason: Interactive matplotlib GUI with keyboard event handlers for artifact selection - core user input mechanism that saves coordinates to disk
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


# Category: Visualization/User Input
# Reason: Orchestrates visualization of all control/signal pairs - reads data and delegates to visualize() for user interaction
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


# Category: Routing
# Reason: Orchestrates NaN replacement for all stores - loops through channels and coordinates calls to addingNaNValues and removeTTLs
def addingNaNtoChunksWithArtifacts(filepath, events):

    logger.debug("Replacing chunks with artifacts by NaN values.")
    storesList = events[1, :]

    path = decide_naming_convention(filepath)

    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        # dirname = os.path.dirname(path[i])
        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]
            for i in range(len(storesList)):
                if (
                    "control_" + name.lower() in storesList[i].lower()
                    or "signal_" + name.lower() in storesList[i].lower()
                ):  # changes done
                    data = addingNaNValues(filepath, storesList[i], name)
                    write_hdf5(data, storesList[i], filepath, "data")
                else:
                    if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
                        continue
                    else:
                        ts = removeTTLs(filepath, storesList[i], name)
                        write_hdf5(ts, storesList[i] + "_" + name, filepath, "ts")

        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
    logger.info("Chunks with artifacts are replaced by NaN values.")


# Category: Routing
# Reason: Orchestrates timestamp concatenation for artifact removal - loops through stores, coordinates eliminateData/eliminateTs calls and writes results
# main function to align timestamps for control, signal and event timestamps for artifacts removal
def processTimestampsForArtifacts(filepath, timeForLightsTurnOn, events):

    logger.debug("Processing timestamps to get rid of artifacts using concatenate method...")
    storesList = events[1, :]

    path = decide_naming_convention(filepath)

    timestamp_dict = dict()
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        # dirname = os.path.dirname(path[i])
        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            for i in range(len(storesList)):
                if (
                    "control_" + name.lower() in storesList[i].lower()
                    or "signal_" + name.lower() in storesList[i].lower()
                ):  # changes done
                    data, timestampNew = eliminateData(
                        filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name
                    )
                    write_hdf5(data, storesList[i], filepath, "data")
                else:
                    if "control" in storesList[i].lower() or "signal" in storesList[i].lower():
                        continue
                    else:
                        ts = eliminateTs(filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name)
                        write_hdf5(ts, storesList[i] + "_" + name, filepath, "ts")

            # timestamp_dict[name] = timestampNew
            write_hdf5(timestampNew, "timeCorrection_" + name, filepath, "timestampNew")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
    logger.info("Timestamps processed, artifacts are removed and good chunks are concatenated.")


# Category: Routing
# Reason: Orchestrates z-score computation for one channel - handles artifact removal logic, coordinates calls to execute_controlFit_dff and z_score_computation
# helper function to compute z-score and deltaF/F
def helper_z_score(control, signal, filepath, name, inputParameters):  # helper_z_score(control_smooth, signal_smooth):

    removeArtifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    filter_window = inputParameters["filter_window"]

    isosbestic_control = inputParameters["isosbestic_control"]
    tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
    coords_path = os.path.join(filepath, "coordsForPreProcessing_" + name + ".npy")

    logger.info("Remove Artifacts : ", removeArtifacts)

    if (control == 0).all() == True:
        control = np.zeros(tsNew.shape[0])

    z_score_arr = np.array([])
    norm_data_arr = np.full(tsNew.shape[0], np.nan)
    control_fit_arr = np.full(tsNew.shape[0], np.nan)
    temp_control_arr = np.full(tsNew.shape[0], np.nan)

    if removeArtifacts == True:
        coords = fetchCoords(filepath, name, tsNew)

        # for artifacts removal, each chunk which was selected by user is being processed individually and then
        # z-score is calculated
        for i in range(coords.shape[0]):
            tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
            if isosbestic_control == False:
                control_arr = helper_create_control_channel(signal[tsNew_index], tsNew[tsNew_index], window=101)
                signal_arr = signal[tsNew_index]
                norm_data, control_fit = execute_controlFit_dff(
                    control_arr, signal_arr, isosbestic_control, filter_window
                )
                temp_control_arr[tsNew_index] = control_arr
                if i < coords.shape[0] - 1:
                    blank_index = np.where((tsNew > coords[i, 1]) & (tsNew < coords[i + 1, 0]))[0]
                    temp_control_arr[blank_index] = np.full(blank_index.shape[0], np.nan)
            else:
                control_arr = control[tsNew_index]
                signal_arr = signal[tsNew_index]
                norm_data, control_fit = execute_controlFit_dff(
                    control_arr, signal_arr, isosbestic_control, filter_window
                )
            norm_data_arr[tsNew_index] = norm_data
            control_fit_arr[tsNew_index] = control_fit

        if artifactsRemovalMethod == "concatenate":
            norm_data_arr = norm_data_arr[~np.isnan(norm_data_arr)]
            control_fit_arr = control_fit_arr[~np.isnan(control_fit_arr)]
        z_score = z_score_computation(norm_data_arr, tsNew, inputParameters)
        z_score_arr = np.concatenate((z_score_arr, z_score))
    else:
        tsNew_index = np.arange(tsNew.shape[0])
        norm_data, control_fit = execute_controlFit_dff(control, signal, isosbestic_control, filter_window)
        z_score = z_score_computation(norm_data, tsNew, inputParameters)
        z_score_arr = np.concatenate((z_score_arr, z_score))
        norm_data_arr[tsNew_index] = norm_data  # np.concatenate((norm_data_arr, norm_data))
        control_fit_arr[tsNew_index] = control_fit  # np.concatenate((control_fit_arr, control_fit))

    # handle the case if there are chunks being cut in the front and the end
    if isosbestic_control == False and removeArtifacts == True:
        coords = coords.flatten()
        # front chunk
        idx = np.where((tsNew >= tsNew[0]) & (tsNew < coords[0]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
        # end chunk
        idx = np.where((tsNew > coords[-1]) & (tsNew <= tsNew[-1]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
        write_hdf5(temp_control_arr, "control_" + name, filepath, "data")

    return z_score_arr, norm_data_arr, control_fit_arr


# Category: Routing
# Reason: Orchestrates z-score computation for all channels in a session - loops through control/signal pairs, calls helper_z_score, writes results
# compute z-score and deltaF/F and save it to hdf5 file
def compute_z_score(filepath, inputParameters):

    logger.debug(f"Computing z-score for each of the data in {filepath}")
    remove_artifacts = inputParameters["removeArtifacts"]

    path_1 = find_files(filepath, "control_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'control*'))
    path_2 = find_files(filepath, "signal_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'signal*'))

    path = sorted(path_1 + path_2, key=str.casefold)

    b = np.divide(np.ones((100,)), 100)
    a = 1

    if len(path) % 2 != 0:
        logger.error("There are not equal number of Control and Signal data")
        raise Exception("There are not equal number of Control and Signal data")

    path = np.asarray(path).reshape(2, -1)

    for i in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, i])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, i])).split(".")[0]).split("_")
        # dirname = os.path.dirname(path[i])

        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            control = read_hdf5("", path[0, i], "data").reshape(-1)
            signal = read_hdf5("", path[1, i], "data").reshape(-1)
            # control_smooth = ss.filtfilt(b, a, control)
            # signal_smooth = ss.filtfilt(b, a, signal)
            # _score, dff = helper_z_score(control_smooth, signal_smooth)
            z_score, dff, control_fit = helper_z_score(control, signal, filepath, name, inputParameters)
            if remove_artifacts == True:
                write_hdf5(z_score, "z_score_" + name, filepath, "data")
                write_hdf5(dff, "dff_" + name, filepath, "data")
                write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
            else:
                write_hdf5(z_score, "z_score_" + name, filepath, "data")
                write_hdf5(dff, "dff_" + name, filepath, "data")
                write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info(f"z-score for the data in {filepath} computed.")


# Category: Routing
# Reason: Top-level orchestrator for timestamp correction across all sessions - loops through folders, coordinates timestamp correction workflow
# function to execute timestamps corrections using functions timestampCorrection and decide_naming_convention_and_applyCorrection
def execute_timestamp_correction(folderNames, inputParameters):

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    isosbestic_control = inputParameters["isosbestic_control"]

    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        cond = check_TDT(folderNames[i])
        logger.debug(f"Timestamps corrections started for {filepath}")
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )

            if isosbestic_control == False:
                storesList = add_control_channel(filepath, storesList)

            if cond == True:
                timestampCorrection_tdt(filepath, timeForLightsTurnOn, storesList)
            else:
                timestampCorrection_csv(filepath, timeForLightsTurnOn, storesList)

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


# Category: Routing
# Reason: Orchestrates reading and merging storeslist files from multiple sessions - loops through folders and consolidates results
# for combining data, reading storeslist file from both data and create a new storeslist array
def check_storeslistfile(folderNames):
    storesList = np.array([[], []])
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
                ),
                axis=1,
            )

    storesList = np.unique(storesList, axis=1)

    return storesList


# Category: Routing
# Reason: Orchestrates data combination workflow - validates sampling rates, coordinates processTimestampsForCombiningData, manages multi-session I/O
# function to combine data when there are two different data files for the same recording session
# it will combine the data, do timestamps processing and save the combined data in the first output folder.
def combineData(folderNames, inputParameters, storesList):

    logger.debug("Combining Data from different data files...")
    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    op_folder = []
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        op_folder.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))

    op_folder = list(np.concatenate(op_folder).flatten())
    sampling_rate_fp = []
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList_new = np.genfromtxt(
                os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=","
            ).reshape(2, -1)
            sampling_rate_fp.append(glob.glob(os.path.join(filepath, "timeCorrection_*")))

    # check if sampling rate is same for both data
    sampling_rate_fp = np.concatenate(sampling_rate_fp)
    sampling_rate = []
    for i in range(sampling_rate_fp.shape[0]):
        sampling_rate.append(read_hdf5("", sampling_rate_fp[i], "sampling_rate"))

    res = all(i == sampling_rate[0] for i in sampling_rate)
    if res == False:
        logger.error("To combine the data, sampling rate for both the data should be same.")
        raise Exception("To combine the data, sampling rate for both the data should be same.")

    # get the output folders informatinos
    op = get_all_stores_for_combining_data(op_folder)

    # processing timestamps for combining the data
    processTimestampsForCombiningData(op, timeForLightsTurnOn, storesList, sampling_rate[0])
    logger.info("Data is combined from different data files.")

    return op


# Category: Routing
# Reason: Top-level orchestrator for z-score computation and artifact removal - coordinates compute_z_score, artifact processing, and visualization calls
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


# Category: Routing
# Reason: Main entry point for Step 4 - orchestrates entire preprocessing workflow including timestamp correction, data combination, and z-score computation
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


# Category: Routing
# Reason: Top-level entry point wrapper - handles error catching and calls extractTsAndSignal
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
