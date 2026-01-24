import glob
import json
import logging
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis.io_utils import get_all_stores_for_combining_data, read_hdf5
from .analysis.transients import analyze_transients

logger = logging.getLogger(__name__)


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


def writeToFile(value: str):
    with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
        file.write(value)


def create_Df(filepath, arr, name, index=[], columns=[]):

    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index, columns=columns)

    df.to_hdf(op, key="df", mode="w")


def create_csv(filepath, arr, name, index=[], columns=[]):
    op = os.path.join(filepath, name)
    df = pd.DataFrame(arr, index=index, columns=columns)
    df.to_csv(op)


def read_Df(filepath, name):
    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df


def visuzlize_peaks(filepath, z_score, timestamps, peaksIndex):

    dirname = os.path.dirname(filepath)

    basename = (os.path.basename(filepath)).split(".")[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(timestamps, z_score, "-", timestamps[peaksIndex], z_score[peaksIndex], "o")
    ax.set_title(basename)
    fig.suptitle(os.path.basename(dirname))
    # plt.show()


def findFreqAndAmp(filepath, inputParameters, window=15, numProcesses=mp.cpu_count()):

    logger.debug("Calculating frequency and amplitude of transients in z-score data....")
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    highAmpFilt = inputParameters["highAmpFilt"]
    transientsThresh = inputParameters["transientsThresh"]

    if selectForTransientsComputation == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForTransientsComputation == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = basename.split("_")[-1]
        sampling_rate = read_hdf5("timeCorrection_" + name_1, filepath, "sampling_rate")[0]
        z_score = read_hdf5("", path[i], "data")
        ts = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        z_score, ts, peaksInd, peaks_occurrences, arr = analyze_transients(
            ts, window, numProcesses, highAmpFilt, transientsThresh, sampling_rate, z_score
        )
        fileName = [os.path.basename(os.path.dirname(filepath))]
        create_Df(filepath, arr, basename, index=fileName, columns=["freq (events/min)", "amplitude"])
        create_csv(
            filepath, arr, "freqAndAmp_" + basename + ".csv", index=fileName, columns=["freq (events/min)", "amplitude"]
        )
        create_csv(
            filepath,
            peaks_occurrences,
            "transientsOccurrences_" + basename + ".csv",
            index=np.arange(peaks_occurrences.shape[0]),
            columns=["timestamps", "amplitude"],
        )
        visuzlize_peaks(path[i], z_score, ts, peaksInd)
    logger.info("Frequency and amplitude of transients in z_score data are calculated.")


def makeAverageDir(filepath):

    op = os.path.join(filepath, "average")
    if not os.path.exists(op):
        os.mkdir(op)

    return op


def averageForGroup(folderNames, inputParameters):

    logger.debug("Combining results for frequency and amplitude of transients in z-score data...")
    path = []
    abspath = inputParameters["abspath"]
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    path_temp_len = []

    for i in range(len(folderNames)):
        if selectForTransientsComputation == "z_score":
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*"))
        elif selectForTransientsComputation == "dff":
            path_temp = glob.glob(os.path.join(folderNames[i], "dff_*"))
        else:
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*")) + glob.glob(
                os.path.join(folderNames[i], "dff_*")
            )

        path_temp_len.append(len(path_temp))

        for j in range(len(path_temp)):
            basename = (os.path.basename(path_temp[j])).split(".")[0]
            # name = name[0]
            temp = [folderNames[i], basename]
            path.append(temp)

    path_temp_len = np.asarray(path_temp_len)
    max_len = np.argmax(path_temp_len)

    naming = []
    for i in range(len(path)):
        naming.append(path[i][1])
    naming = np.unique(np.asarray(naming))

    new_path = [[] for _ in range(path_temp_len[max_len])]
    for i in range(len(path)):
        idx = np.where(naming == path[i][1])[0][0]
        new_path[idx].append(path[i])

    op = makeAverageDir(abspath)

    for i in range(len(new_path)):
        arr = []  # np.zeros((len(new_path[i]), 2))
        fileName = []
        temp_path = new_path[i]
        for j in range(len(temp_path)):
            if not os.path.exists(os.path.join(temp_path[j][0], "freqAndAmp_" + temp_path[j][1] + ".h5")):
                continue
            else:
                df = read_Df(temp_path[j][0], temp_path[j][1])
                arr.append(np.array([df["freq (events/min)"][0], df["amplitude"][0]]))
                fileName.append(os.path.basename(temp_path[j][0]))

        arr = np.asarray(arr)
        create_Df(op, arr, temp_path[j][1], index=fileName, columns=["freq (events/min)", "amplitude"])
        create_csv(
            op,
            arr,
            "freqAndAmp_" + temp_path[j][1] + ".csv",
            index=fileName,
            columns=["freq (events/min)", "amplitude"],
        )
    logger.info("Results for frequency and amplitude of transients in z-score data are combined.")


def executeFindFreqAndAmp(inputParameters):

    logger.info("Finding transients in z-score data and calculating frequency and amplitude....")

    inputParameters = inputParameters

    average = inputParameters["averageForGroup"]
    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    folderNames = inputParameters["folderNames"]
    combine_data = inputParameters["combine_data"]
    moving_window = inputParameters["moving_window"]
    numProcesses = inputParameters["numberOfCores"]
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            "Warning : # of cores parameter set is greater than the cores available \
			   available in your machine"
        )
        numProcesses = mp.cpu_count() - 1

    if average == True:
        execute_average_for_group(inputParameters, folderNamesForAvg)
    else:
        if combine_data == True:
            execute_find_freq_and_amp_combined(inputParameters, folderNames, moving_window, numProcesses)
        else:
            execute_find_freq_and_amp(inputParameters, folderNames, moving_window, numProcesses)

    logger.info("Transients in z-score data found and frequency and amplitude are calculated.")


def execute_find_freq_and_amp(inputParameters, folderNames, moving_window, numProcesses):
    for i in range(len(folderNames)):
        logger.debug(f"Finding transients in z-score data of {folderNames[i]} and calculating frequency and amplitude.")
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )
            findFreqAndAmp(filepath, inputParameters, window=moving_window, numProcesses=numProcesses)
            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
            inputParameters["step"] += 1
        logger.info("Transients in z-score data found and frequency and amplitude are calculated.")
    plt.show()


def execute_find_freq_and_amp_combined(inputParameters, folderNames, moving_window, numProcesses):
    storesListPath = []
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = list(np.concatenate(storesListPath).flatten())
    op = get_all_stores_for_combining_data(storesListPath)
    for i in range(len(op)):
        filepath = op[i][0]
        storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
        findFreqAndAmp(filepath, inputParameters, window=moving_window, numProcesses=numProcesses)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1
    plt.show()


def execute_average_for_group(inputParameters, folderNamesForAvg):
    if len(folderNamesForAvg) == 0:
        logger.error("Not a single folder name is provided in folderNamesForAvg in inputParamters File.")
        raise Exception("Not a single folder name is provided in folderNamesForAvg in inputParamters File.")
    storesListPath = []
    for i in range(len(folderNamesForAvg)):
        filepath = folderNamesForAvg[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    averageForGroup(storesListPath, inputParameters)
    writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
    inputParameters["step"] += 1


if __name__ == "__main__":
    try:
        executeFindFreqAndAmp(json.loads(sys.argv[1]))
        logger.info("#" * 400)
    except Exception as e:
        with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
            file.write(str(-1) + "\n")
        logger.error(str(e))
        raise e
