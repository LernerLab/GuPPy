import glob
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from .analysis.artifact_removal import remove_artifacts
from .analysis.combine_data import combine_data
from .analysis.control_channel import add_control_channel, create_control_channel
from .analysis.io_utils import (
    check_storeslistfile,
    check_TDT,
    find_files,
    get_all_stores_for_combining_data,  # noqa: F401 -- Necessary for other modules that depend on preprocess.py
    get_coords,
    read_hdf5,
    takeOnlyDirs,
)
from .analysis.standard_io import (
    read_control_and_signal,
    read_coords_pairwise,
    read_corrected_data,
    read_corrected_data_dict,
    read_corrected_timestamps_pairwise,
    read_corrected_ttl_timestamps,
    read_ttl,
    write_artifact_removal,
    write_corrected_data,
    write_corrected_timestamps,
    write_corrected_ttl_timestamps,
    write_zscore,
)
from .analysis.timestamp_correction import correct_timestamps
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
            name_to_timestamps_ttl = read_ttl(filepath, storesList)

            timestamps_dicts = correct_timestamps(
                timeForLightsTurnOn,
                storesList,
                name_to_timestamps,
                name_to_data,
                name_to_sampling_rate,
                name_to_npoints,
                name_to_timestamps_ttl,
                mode=mode,
            )
            (
                name_to_corrected_timestamps,
                name_to_correctionIndex,
                name_to_corrected_data,
                compound_name_to_corrected_ttl_timestamps,
            ) = timestamps_dicts

            write_corrected_timestamps(
                filepath,
                name_to_corrected_timestamps,
                name_to_timestamps,
                name_to_sampling_rate,
                name_to_correctionIndex,
            )
            write_corrected_data(filepath, name_to_corrected_data)
            write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)

            # check if isosbestic control is false and also if new control channel is added
            if isosbestic_control == False:
                create_control_channel(filepath, storesList, window=101)

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
            inputParameters["step"] += 1
        logger.info(f"Timestamps corrections finished for {filepath}")


# function to compute z-score and deltaF/F
def execute_zscore(folderNames, inputParameters):

    plot_zScore_dff = inputParameters["plot_zScore_dff"]
    combine_data = inputParameters["combine_data"]
    remove_artifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    filter_window = inputParameters["filter_window"]
    isosbestic_control = inputParameters["isosbestic_control"]
    zscore_method = inputParameters["zscore_method"]
    baseline_start, baseline_end = inputParameters["baselineWindowStart"], inputParameters["baselineWindowEnd"]

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
        logger.debug(f"Computing z-score for each of the data in {filepath}")
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
            if name_1[-1] != name_2[-1]:
                logger.error("Error in naming convention of files or Error in storesList file")
                raise Exception("Error in naming convention of files or Error in storesList file")
            name = name_1[-1]

            control, signal, tsNew = read_corrected_data(path[0, i], path[1, i], filepath, name)
            coords = get_coords(filepath, name, tsNew, remove_artifacts)
            z_score, dff, control_fit, temp_control_arr = compute_z_score(
                control,
                signal,
                tsNew,
                coords,
                artifactsRemovalMethod,
                filter_window,
                isosbestic_control,
                zscore_method,
                baseline_start,
                baseline_end,
            )
            write_zscore(filepath, name, z_score, dff, control_fit, temp_control_arr)

        logger.info(f"z-score for the data in {filepath} computed.")

        if not remove_artifacts:
            visualizeControlAndSignal(filepath, removeArtifacts=remove_artifacts)

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
    logger.info("Z-score computation completed.")


# function to remove artifacts from z-score data
def execute_artifact_removal(folderNames, inputParameters):

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    combine_data = inputParameters["combine_data"]

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

        name_to_data = read_corrected_data_dict(filepath, storesList)
        pair_name_to_tsNew, pair_name_to_sampling_rate = read_corrected_timestamps_pairwise(filepath)
        pair_name_to_coords = read_coords_pairwise(filepath, pair_name_to_tsNew)
        compound_name_to_ttl_timestamps = read_corrected_ttl_timestamps(filepath, storesList)

        logger.debug("Removing artifacts from the data...")
        name_to_data, pair_name_to_timestamps, compound_name_to_ttl_timestamps = remove_artifacts(
            timeForLightsTurnOn,
            storesList,
            pair_name_to_tsNew,
            pair_name_to_sampling_rate,
            pair_name_to_coords,
            name_to_data,
            compound_name_to_ttl_timestamps,
            method=artifactsRemovalMethod,
        )

        write_artifact_removal(filepath, name_to_data, pair_name_to_timestamps, compound_name_to_ttl_timestamps)
        visualizeControlAndSignal(filepath, removeArtifacts=True)

        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1

    plt.show()
    logger.info("Artifact removal completed.")


# function to combine data when there are two different data files for the same recording session
# it will combine the data, do timestamps processing and save the combined data in the first output folder.
def execute_combine_data(folderNames, inputParameters, storesList):
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
    for filepaths_to_combine in op:
        combine_data(filepaths_to_combine, timeForLightsTurnOn, storesList, sampling_rate[0])
    logger.info("Data is combined from different data files.")

    return op


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
        if remove_artifacts == True:
            execute_artifact_removal(folderNames, inputParameters)
    else:
        pbMaxValue = 1 + len(folderNames)
        writeToFile(str((pbMaxValue) * 10) + "\n" + str(10) + "\n")
        execute_timestamp_correction(folderNames, inputParameters)
        storesList = check_storeslistfile(folderNames)
        op_folder = execute_combine_data(folderNames, inputParameters, storesList)
        execute_zscore(op_folder, inputParameters)
        if remove_artifacts == True:
            execute_artifact_removal(op_folder, inputParameters)


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
