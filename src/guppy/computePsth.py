# coding: utf-8

import glob
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from itertools import repeat

import numpy as np
from scipy import signal as ss

from .analysis.compute_psth import compute_psth
from .analysis.cross_correlation import computeCrossCorrelation
from .analysis.io_utils import (
    get_all_stores_for_combining_data,
    makeAverageDir,
    read_hdf5,
)
from .analysis.psth_average import averageForGroup
from .analysis.psth_peak_and_area import findPSTHPeakAndArea
from .analysis.psth_utils import create_Df

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


def psthForEachStorename(inputParameters):

    logger.info("Computing PSTH, Peak and Area for each event...")
    inputParameters = inputParameters

    # storesList = np.genfromtxt(inputParameters['storesListPath'], dtype='str', delimiter=',')

    average = inputParameters["averageForGroup"]
    combine_data = inputParameters["combine_data"]
    numProcesses = inputParameters["numberOfCores"]
    inputParameters["step"] = 0
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            "Warning : # of cores parameter set is greater than the cores available \
			   available in your machine"
        )
        numProcesses = mp.cpu_count() - 1

    logger.info(f"Average for group : {average}")

    # for average following if statement will be executed
    if average == True:
        execute_average_for_group(inputParameters)

    # for individual analysis following else statement will be executed
    else:
        if combine_data == True:
            execute_psth_combined(inputParameters)
        else:
            orchestrate_psth(inputParameters)
    logger.info("PSTH, Area and Peak are computed for all events.")
    return inputParameters


# function to create PSTH for each event using function helper_psth and save the PSTH to h5 file
def execute_compute_psth(filepath, event, inputParameters):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if "control" in event.lower() or "signal" in event.lower():
        return 0

    selectForComputePsth = inputParameters["selectForComputePsth"]
    bin_psth_trials = inputParameters["bin_psth_trials"]
    use_time_or_trials = inputParameters["use_time_or_trials"]
    nSecPrev, nSecPost = inputParameters["nSecPrev"], inputParameters["nSecPost"]
    baselineStart, baselineEnd = inputParameters["baselineCorrectionStart"], inputParameters["baselineCorrectionEnd"]
    timeInterval = inputParameters["timeInterval"]

    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    b = np.divide(np.ones((100,)), 100)
    a = 1

    for i in range(len(path)):
        logger.info(f"Computing PSTH for event {event}...")
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = basename.split("_")[-1]
        control = read_hdf5("control_" + name_1, os.path.dirname(path[i]), "data")
        if (control == 0).all() == True:
            signal = read_hdf5("signal_" + name_1, os.path.dirname(path[i]), "data")
            z_score = ss.filtfilt(b, a, signal)
            just_use_signal = True
        else:
            z_score = read_hdf5("", path[i], "data")
            just_use_signal = False

        sampling_rate = read_hdf5("timeCorrection_" + name_1, filepath, "sampling_rate")[0]
        ts = read_hdf5(event + "_" + name_1, filepath, "ts")
        if use_time_or_trials == "Time (min)" and bin_psth_trials > 0:
            corrected_timestamps = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        else:
            corrected_timestamps = None
        psth, psth_baselineUncorrected, cols = compute_psth(
            z_score,
            event,
            filepath,
            nSecPrev,
            nSecPost,
            timeInterval,
            bin_psth_trials,
            use_time_or_trials,
            baselineStart,
            baselineEnd,
            name_1,
            just_use_signal,
            sampling_rate,
            ts,
            corrected_timestamps,
        )

        create_Df(
            filepath,
            event + "_" + name_1 + "_baselineUncorrected",
            basename,
            psth_baselineUncorrected,
            columns=cols,
        )  # extra
        create_Df(filepath, event + "_" + name_1, basename, psth, columns=cols)
        logger.info(f"PSTH for event {event} computed.")


def orchestrate_psth(inputParameters):
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    writeToFile(str((storesListPath.shape[0] + storesListPath.shape[0] + 1) * 10) + "\n" + str(10) + "\n")
    for i in range(len(folderNames)):
        logger.debug(f"Computing PSTH, Peak and Area for each event in {folderNames[i]}")
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )

            with mp.Pool(numProcesses) as p:
                p.starmap(execute_compute_psth, zip(repeat(filepath), storesList[1, :], repeat(inputParameters)))

            with mp.Pool(numProcesses) as pq:
                pq.starmap(findPSTHPeakAndArea, zip(repeat(filepath), storesList[1, :], repeat(inputParameters)))

            with mp.Pool(numProcesses) as cr:
                cr.starmap(computeCrossCorrelation, zip(repeat(filepath), storesList[1, :], repeat(inputParameters)))

                # for k in range(storesList.shape[1]):
                # 	storenamePsth(filepath, storesList[1,k], inputParameters)
                # 	findPSTHPeakAndArea(filepath, storesList[1,k], inputParameters)

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
            inputParameters["step"] += 1
        logger.info(f"PSTH, Area and Peak are computed for all events in {folderNames[i]}.")


def execute_psth_combined(inputParameters):
    folderNames = inputParameters["folderNames"]
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*"))))
    storesListPath = list(np.concatenate(storesListPath).flatten())
    op = get_all_stores_for_combining_data(storesListPath)
    writeToFile(str((len(op) + len(op) + 1) * 10) + "\n" + str(10) + "\n")
    for i in range(len(op)):
        storesList = np.asarray([[], []])
        for j in range(len(op[i])):
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(os.path.join(op[i][j], "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
                ),
                axis=1,
            )
        storesList = np.unique(storesList, axis=1)
        for k in range(storesList.shape[1]):
            execute_compute_psth(op[i][0], storesList[1, k], inputParameters)
            findPSTHPeakAndArea(op[i][0], storesList[1, k], inputParameters)
            computeCrossCorrelation(op[i][0], storesList[1, k], inputParameters)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1


def execute_average_for_group(inputParameters):
    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    if len(folderNamesForAvg) == 0:
        logger.error("Not a single folder name is provided in folderNamesForAvg in inputParamters File.")
        raise Exception("Not a single folder name is provided in folderNamesForAvg in inputParamters File.")

    storesListPath = []
    for i in range(len(folderNamesForAvg)):
        filepath = folderNamesForAvg[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    storesList = np.asarray([[], []])
    for i in range(storesListPath.shape[0]):
        storesList = np.concatenate(
            (
                storesList,
                np.genfromtxt(os.path.join(storesListPath[i], "storesList.csv"), dtype="str", delimiter=",").reshape(
                    2, -1
                ),
            ),
            axis=1,
        )
    storesList = np.unique(storesList, axis=1)
    op = makeAverageDir(inputParameters["abspath"])
    np.savetxt(os.path.join(op, "storesList.csv"), storesList, delimiter=",", fmt="%s")
    pbMaxValue = 0
    for j in range(storesList.shape[1]):
        if "control" in storesList[1, j].lower() or "signal" in storesList[1, j].lower():
            continue
        else:
            pbMaxValue += 1
    writeToFile(str((1 + pbMaxValue + 1) * 10) + "\n" + str(10) + "\n")
    for k in range(storesList.shape[1]):
        if "control" in storesList[1, k].lower() or "signal" in storesList[1, k].lower():
            continue
        else:
            averageForGroup(storesListPath, storesList[1, k], inputParameters)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1


def main(input_parameters):
    try:
        inputParameters = psthForEachStorename(input_parameters)
        subprocess.call([sys.executable, "-m", "guppy.findTransientsFreqAndAmp", json.dumps(inputParameters)])
        logger.info("#" * 400)
    except Exception as e:
        with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
            file.write(str(-1) + "\n")
        logger.error(str(e))
        raise e


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
