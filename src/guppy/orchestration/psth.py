# coding: utf-8

import glob
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
from itertools import repeat

import numpy as np
from scipy import signal as ss

from ..analysis.compute_psth import compute_psth
from ..analysis.cross_correlation import compute_cross_correlation
from ..analysis.io_utils import (
    make_dir_for_cross_correlation,
    makeAverageDir,
    read_hdf5,
    write_hdf5,
)
from ..analysis.psth_average import averageForGroup
from ..analysis.psth_peak_and_area import compute_psth_peak_and_area
from ..analysis.psth_utils import (
    create_Df_for_cross_correlation,
    create_Df_for_psth,
    getCorrCombinations,
)
from ..analysis.standard_io import (
    write_peak_and_area_to_csv,
    write_peak_and_area_to_hdf5,
)
from ..frontend.progress import PB_STEPS_FILE, writeToFile
from ..utils.utils import get_all_stores_for_combining_data, read_Df, takeOnlyDirs

logger = logging.getLogger(__name__)


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
        psth, psth_baselineUncorrected, cols, ts = compute_psth(
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
        write_hdf5(ts, event + "_" + name_1, filepath, "ts")

        create_Df_for_psth(
            filepath,
            event + "_" + name_1 + "_baselineUncorrected",
            basename,
            psth_baselineUncorrected,
            columns=cols,
        )  # extra
        create_Df_for_psth(filepath, event + "_" + name_1, basename, psth, columns=cols)
        logger.info(f"PSTH for event {event} computed.")


# function to compute PSTH peak and area using the function helperPSTHPeakAndArea save the values to h5 and csv files.
def execute_compute_psth_peak_and_area(filepath, event, inputParameters):

    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if "control" in event.lower() or "signal" in event.lower():
        return 0

    # sampling_rate = read_hdf5(storesList[0,0], filepath, 'sampling_rate')
    peak_startPoint = inputParameters["peak_startPoint"]
    peak_endPoint = inputParameters["peak_endPoint"]
    selectForComputePsth = inputParameters["selectForComputePsth"]

    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    for i in range(len(path)):
        logger.info(f"Computing peak and area for PSTH mean signal for event {event}...")
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = basename.split("_")[-1]
        sampling_rate = read_hdf5("timeCorrection_" + name_1, filepath, "sampling_rate")[0]
        psth = read_Df(filepath, event + "_" + name_1, basename)
        cols = list(psth.columns)
        regex = re.compile("bin_[(]")
        bin_names = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
        regex_trials = re.compile("[+-]?([0-9]*[.])?[0-9]+")
        trials_names = [cols[i] for i in range(len(cols)) if regex_trials.match(cols[i])]
        psth_mean_bin_names = trials_names + bin_names + ["mean"]
        psth_mean_bin_mean = np.asarray(psth[psth_mean_bin_names])
        timestamps = np.asarray(psth["timestamps"]).ravel()  # np.asarray(read_Df(filepath, 'ts_psth', '')).ravel()
        peak_area = compute_psth_peak_and_area(
            psth_mean_bin_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint
        )  # peak, area =
        # arr = np.array([[peak, area]])
        fileName = [os.path.basename(os.path.dirname(filepath))]
        index = [fileName[0] + "_" + s for s in psth_mean_bin_names]
        write_peak_and_area_to_hdf5(
            filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index
        )  # columns=['peak', 'area']
        write_peak_and_area_to_csv(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
        logger.info(f"Peak and Area for PSTH mean signal for event {event} computed.")


def execute_compute_cross_correlation(filepath, event, inputParameters):
    isCompute = inputParameters["computeCorr"]
    removeArtifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    if isCompute == True:
        if removeArtifacts == True and artifactsRemovalMethod == "concatenate":
            raise Exception(
                "For cross-correlation, when removeArtifacts is True, artifacts removal method\
                            should be replace with NaNs and not concatenate"
            )
        corr_info, type = getCorrCombinations(filepath, inputParameters)
        if "control" in event.lower() or "signal" in event.lower():
            return
        else:
            for i in range(1, len(corr_info)):
                logger.debug(f"Computing cross-correlation for event {event}...")
                for j in range(len(type)):
                    psth_a = read_Df(filepath, event + "_" + corr_info[i - 1], type[j] + "_" + corr_info[i - 1])
                    psth_b = read_Df(filepath, event + "_" + corr_info[i], type[j] + "_" + corr_info[i])
                    sample_rate = 1 / (psth_a["timestamps"][1] - psth_a["timestamps"][0])
                    psth_a = psth_a.drop(columns=["timestamps", "err", "mean"])
                    psth_b = psth_b.drop(columns=["timestamps", "err", "mean"])
                    cols_a, cols_b = np.array(psth_a.columns), np.array(psth_b.columns)
                    if np.intersect1d(cols_a, cols_b).size > 0:
                        cols = list(np.intersect1d(cols_a, cols_b))
                    else:
                        cols = list(cols_a)
                    arr_A, arr_B = np.array(psth_a).T, np.array(psth_b).T
                    cross_corr = compute_cross_correlation(arr_A, arr_B, sample_rate)
                    cols.append("timestamps")
                    create_Df_for_cross_correlation(
                        make_dir_for_cross_correlation(filepath),
                        "corr_" + event,
                        type[j] + "_" + corr_info[i - 1] + "_" + corr_info[i],
                        cross_corr,
                        cols,
                    )
                logger.info(f"Cross-correlation for event {event} computed.")


def orchestrate_psth(inputParameters):
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    writeToFile(
        str((storesListPath.shape[0] + storesListPath.shape[0] + 1) * 10) + "\n" + str(10) + "\n",
        file_path=PB_STEPS_FILE,
    )
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
                pq.starmap(
                    execute_compute_psth_peak_and_area, zip(repeat(filepath), storesList[1, :], repeat(inputParameters))
                )

            with mp.Pool(numProcesses) as cr:
                cr.starmap(
                    execute_compute_cross_correlation, zip(repeat(filepath), storesList[1, :], repeat(inputParameters))
                )

                # for k in range(storesList.shape[1]):
                # 	storenamePsth(filepath, storesList[1,k], inputParameters)
                # 	findPSTHPeakAndArea(filepath, storesList[1,k], inputParameters)

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
            inputParameters["step"] += 1
        logger.info(f"PSTH, Area and Peak are computed for all events in {folderNames[i]}.")


def execute_psth_combined(inputParameters):
    folderNames = inputParameters["folderNames"]
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(folderNames[i], "*_output_*"))))
    storesListPath = list(np.concatenate(storesListPath).flatten())
    op = get_all_stores_for_combining_data(storesListPath)
    writeToFile(str((len(op) + len(op) + 1) * 10) + "\n" + str(10) + "\n", file_path=PB_STEPS_FILE)
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
            execute_compute_psth_peak_and_area(op[i][0], storesList[1, k], inputParameters)
            execute_compute_cross_correlation(op[i][0], storesList[1, k], inputParameters)
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
    writeToFile(str((1 + pbMaxValue + 1) * 10) + "\n" + str(10) + "\n", file_path=PB_STEPS_FILE)
    for k in range(storesList.shape[1]):
        if "control" in storesList[1, k].lower() or "signal" in storesList[1, k].lower():
            continue
        else:
            averageForGroup(storesListPath, storesList[1, k], inputParameters)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n")
        inputParameters["step"] += 1


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


def main(input_parameters):
    try:
        inputParameters = psthForEachStorename(input_parameters)
        subprocess.call([sys.executable, "-m", "guppy.orchestration.transients", json.dumps(inputParameters)])
        logger.info("#" * 400)
    except Exception as e:
        writeToFile(str(-1) + "\n", file_path=PB_STEPS_FILE)
        logger.error(str(e))
        raise e


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
