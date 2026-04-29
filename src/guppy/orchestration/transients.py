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
from ..frontend.progress import PB_STEPS_FILE, subprocess_main_handler, writeToFile
from ..utils.utils import get_all_stores_for_combining_data, takeOnlyDirs
from ..visualization.transients import visualize_peaks

logger = logging.getLogger(__name__)


def findFreqAndAmp(filepath, inputParameters, window=15, numProcesses=mp.cpu_count()):
    """Detect transients and compute their frequency and amplitude for one output directory.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    inputParameters : dict
        Full pipeline input parameters.
    window : int, optional
        Moving-window size in seconds used for transient detection.
    numProcesses : int, optional
        Number of worker processes for parallel transient analysis.
    """
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
    """Plot detected transient peaks for each individual session.

    Parameters
    ----------
    folderNames : list of str
        Session folder paths.
    inputParameters : dict
        Full pipeline input parameters.
    """
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
    """Plot detected transient peaks for combined (multi-session) data.

    Parameters
    ----------
    folderNames : list of str
        Session folder paths.
    inputParameters : dict
        Full pipeline input parameters.
    """
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
    """Entry point for step-5 transient analysis: dispatches to the appropriate sub-routine.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
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
            f"Number of cores requested ({numProcesses}) exceeds available cores "
            f"({mp.cpu_count()}); using {mp.cpu_count() - 1}."
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
    """Compute transient frequency and amplitude for each individual session.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    folderNames : list of str
        Session folder paths.
    moving_window : int
        Moving-window size in seconds for transient detection.
    numProcesses : int
        Number of parallel worker processes.
    """
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
    """Compute transient frequency and amplitude for combined (multi-session) data.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    folderNames : list of str
        Session folder paths.
    moving_window : int
        Moving-window size in seconds for transient detection.
    numProcesses : int
        Number of parallel worker processes.
    """
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
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1


def execute_average_for_group(inputParameters, folderNamesForAvg):
    """Average transient frequency and amplitude results across all group sessions.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    folderNamesForAvg : list of str
        Session folder paths selected for group averaging.

    Raises
    ------
    ValueError
        When ``folderNamesForAvg`` is empty.
    """
    if len(folderNamesForAvg) == 0:
        message = (
            "No folders selected for group averaging (folderNamesForAvg is empty in inputParameters). "
            "Select folders in the 'Group Folders for Averaging' picker before running the average step."
        )
        logger.error(message)
        raise ValueError(message)
    storesListPath = []
    for i in range(len(folderNamesForAvg)):
        filepath = folderNamesForAvg[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    averageForGroup(storesListPath, inputParameters)
    writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
    inputParameters["step"] += 1


@subprocess_main_handler
def main(input_parameters):
    """Subprocess entry point for the transient-analysis step.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters deserialized from the subprocess argument.
    """
    executeFindFreqAndAmp(input_parameters)


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
