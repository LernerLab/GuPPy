import glob
import json
import logging
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.io_utils import (
    read_hdf5,
)
from ..analysis.standard_io import (
    read_transients_from_hdf5,
    write_freq_and_amp_to_csv,
    write_freq_and_amp_to_hdf5,
    write_transients_to_hdf5,
)
from ..analysis.transients import analyze_transients
from ..analysis.transients_average import averageForGroup
from ..frontend.progress import PB_STEPS_FILE, writeToFile
from ..utils.utils import get_all_stores_for_combining_data, takeOnlyDirs
from ..visualization.transients import visualize_peaks

logger = logging.getLogger(__name__)


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
        write_freq_and_amp_to_hdf5(filepath, arr, basename, index=fileName, columns=["freq (events/min)", "amplitude"])
        write_freq_and_amp_to_csv(
            filepath, arr, "freqAndAmp_" + basename + ".csv", index=fileName, columns=["freq (events/min)", "amplitude"]
        )
        write_freq_and_amp_to_csv(
            filepath,
            peaks_occurrences,
            "transientsOccurrences_" + basename + ".csv",
            index=np.arange(peaks_occurrences.shape[0]),
            columns=["timestamps", "amplitude"],
        )
        write_transients_to_hdf5(filepath, basename, z_score, ts, peaksInd)
    logger.info("Frequency and amplitude of transients in z_score data are calculated.")


def execute_visualize_peaks(folderNames, inputParameters):
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    for i in range(len(folderNames)):
        logger.debug(f"Finding transients in z-score data of {folderNames[i]} and calculating frequency and amplitude.")
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            if selectForTransientsComputation == "z_score":
                path = glob.glob(os.path.join(filepath, "z_score_*"))
            elif selectForTransientsComputation == "dff":
                path = glob.glob(os.path.join(filepath, "dff_*"))
            else:
                path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

            for i in range(len(path)):
                basename = (os.path.basename(path[i])).split(".")[0]
                z_score, ts, peaksInd = read_transients_from_hdf5(filepath, basename)

                suptitle = os.path.basename(os.path.dirname(path[i]))
                title = (os.path.basename(path[i])).split(".")[0]
                visualize_peaks(title, suptitle, z_score, ts, peaksInd)

    logger.info("Frequency and amplitude of transients in z_score data are visualized.")
    plt.show()


def execute_visualize_peaks_combined(folderNames, inputParameters):
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]

    storesListPath = []
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = list(np.concatenate(storesListPath).flatten())
    op = get_all_stores_for_combining_data(storesListPath)
    for i in range(len(op)):
        filepath = op[i][0]

        if selectForTransientsComputation == "z_score":
            path = glob.glob(os.path.join(filepath, "z_score_*"))
        elif selectForTransientsComputation == "dff":
            path = glob.glob(os.path.join(filepath, "dff_*"))
        else:
            path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

        for i in range(len(path)):
            basename = (os.path.basename(path[i])).split(".")[0]
            z_score, ts, peaksInd = read_transients_from_hdf5(filepath, basename)

            suptitle = os.path.basename(os.path.dirname(path[i]))
            title = (os.path.basename(path[i])).split(".")[0]
            visualize_peaks(title, suptitle, z_score, ts, peaksInd)

    logger.info("Frequency and amplitude of transients in z_score data are calculated.")
    plt.show()


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
        headless = bool(os.environ.get("GUPPY_BASE_DIR"))
        if combine_data == True:
            execute_find_freq_and_amp_combined(inputParameters, folderNames, moving_window, numProcesses)
            if not headless:
                execute_visualize_peaks_combined(folderNames, inputParameters)
        else:
            execute_find_freq_and_amp(inputParameters, folderNames, moving_window, numProcesses)
            if not headless:
                execute_visualize_peaks(folderNames, inputParameters)

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
            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
            inputParameters["step"] += 1
        logger.info("Transients in z-score data found and frequency and amplitude are calculated.")


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
    writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
    inputParameters["step"] += 1


if __name__ == "__main__":
    try:
        executeFindFreqAndAmp(json.loads(sys.argv[1]))
        logger.info("#" * 400)
    except Exception as e:
        writeToFile(str(-1) + "\n", file_path=PB_STEPS_FILE)
        logger.error(str(e))
        raise e
