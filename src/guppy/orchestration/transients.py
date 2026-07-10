import glob
import json
import logging
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from .group_utils import gather_group_run_folders
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
from ..utils.utils import (
    get_all_stores_for_combining_data,
    is_headless,
    select_run_folders,
)
from ..visualization.transients import visualize_peaks

logger = logging.getLogger(__name__)


def findFreqAndAmp(
    filepath: str, inputParameters: dict[str, object], window: int = 15, numProcesses: int = mp.cpu_count()
) -> None:
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
        timestamps = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        z_score, timestamps, peaksInd, peaks_occurrences, freq_and_amp = analyze_transients(
            timestamps, window, numProcesses, highAmpFilt, transientsThresh, sampling_rate, z_score
        )
        fileName = [os.path.basename(os.path.dirname(filepath))]
        write_freq_and_amp_to_hdf5(
            filepath, freq_and_amp, basename, index=fileName, columns=["freq (events/min)", "amplitude"]
        )
        write_freq_and_amp_to_csv(
            filepath,
            freq_and_amp,
            "freqAndAmp_" + basename + ".csv",
            index=fileName,
            columns=["freq (events/min)", "amplitude"],
        )
        write_freq_and_amp_to_csv(
            filepath,
            peaks_occurrences,
            "transientsOccurrences_" + basename + ".csv",
            index=np.arange(peaks_occurrences.shape[0]),
            columns=["timestamps", "amplitude"],
        )
        write_transients_to_hdf5(filepath, basename, z_score, timestamps, peaksInd)
    logger.info("Frequency and amplitude of transients in z_score data are calculated.")


def execute_visualize_peaks(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """Plot detected transient peaks for each individual session.

    Parameters
    ----------
    session_folders : list of str
        Session folder paths.
    inputParameters : dict
        Full pipeline input parameters.
    """
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    selected_runs = inputParameters.get("selected_runs") or {}
    for i in range(len(session_folders)):
        logger.debug(
            f"Finding transients in z-score data of {session_folders[i]} and calculating frequency and amplitude."
        )
        filepath = session_folders[i]
        run_folders = select_run_folders(filepath, selected_runs.get(filepath))
        for j in range(len(run_folders)):
            filepath = run_folders[j]
            if selectForTransientsComputation == "z_score":
                path = glob.glob(os.path.join(filepath, "z_score_*"))
            elif selectForTransientsComputation == "dff":
                path = glob.glob(os.path.join(filepath, "dff_*"))
            else:
                path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

            for i in range(len(path)):
                basename = (os.path.basename(path[i])).split(".")[0]
                z_score, timestamps, peaksInd = read_transients_from_hdf5(filepath, basename)

                suptitle = os.path.basename(os.path.dirname(path[i]))
                title = (os.path.basename(path[i])).split(".")[0]
                visualize_peaks(title, suptitle, z_score, timestamps, peaksInd)

    logger.info("Frequency and amplitude of transients in z_score data are visualized.")
    plt.show()


def execute_visualize_peaks_combined(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """Plot detected transient peaks for combined (multi-session) data.

    Parameters
    ----------
    session_folders : list of str
        Session folder paths.
    inputParameters : dict
        Full pipeline input parameters.
    """
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    selected_runs = inputParameters.get("selected_runs") or {}

    run_folders = []
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        run_folders.append(select_run_folders(filepath, selected_runs.get(filepath)))
    run_folders = list(np.concatenate(run_folders).flatten())
    combined_output_groups = get_all_stores_for_combining_data(run_folders)
    for i in range(len(combined_output_groups)):
        filepath = combined_output_groups[i][0]

        if selectForTransientsComputation == "z_score":
            path = glob.glob(os.path.join(filepath, "z_score_*"))
        elif selectForTransientsComputation == "dff":
            path = glob.glob(os.path.join(filepath, "dff_*"))
        else:
            path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

        for i in range(len(path)):
            basename = (os.path.basename(path[i])).split(".")[0]
            z_score, timestamps, peaksInd = read_transients_from_hdf5(filepath, basename)

            suptitle = os.path.basename(os.path.dirname(path[i]))
            title = (os.path.basename(path[i])).split(".")[0]
            visualize_peaks(title, suptitle, z_score, timestamps, peaksInd)

    logger.info("Frequency and amplitude of transients in z_score data are calculated.")
    plt.show()


def executeFindFreqAndAmp(inputParameters: dict[str, object]) -> None:
    """Entry point for step-4 transient analysis: dispatches to the appropriate sub-routine.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    logger.info("Finding transients in z-score data and calculating frequency and amplitude....")

    inputParameters = inputParameters

    average = inputParameters["averageForGroup"]
    group_session_folders = inputParameters["group_session_folders"]
    session_folders = inputParameters["session_folders"]
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
        execute_average_for_group(inputParameters, group_session_folders)
    else:
        headless = is_headless()
        if combine_data == True:
            execute_find_freq_and_amp_combined(inputParameters, session_folders, moving_window, numProcesses)
            if not headless:
                execute_visualize_peaks_combined(session_folders, inputParameters)
        else:
            execute_find_freq_and_amp(inputParameters, session_folders, moving_window, numProcesses)
            if not headless:
                execute_visualize_peaks(session_folders, inputParameters)

    logger.info("Transients in z-score data found and frequency and amplitude are calculated.")


def execute_find_freq_and_amp(
    inputParameters: dict[str, object], session_folders: list[str], moving_window: int, numProcesses: int
) -> None:
    """Compute transient frequency and amplitude for each individual session.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    session_folders : list of str
        Session folder paths.
    moving_window : int
        Moving-window size in seconds for transient detection.
    numProcesses : int
        Number of parallel worker processes.
    """
    selected_runs = inputParameters.get("selected_runs") or {}
    for i in range(len(session_folders)):
        logger.debug(
            f"Finding transients in z-score data of {session_folders[i]} and calculating frequency and amplitude."
        )
        filepath = session_folders[i]
        run_folders = select_run_folders(filepath, selected_runs.get(filepath))
        for j in range(len(run_folders)):
            filepath = run_folders[j]
            store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )
            findFreqAndAmp(filepath, inputParameters, window=moving_window, numProcesses=numProcesses)
            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
            inputParameters["step"] += 1
        logger.info("Transients in z-score data found and frequency and amplitude are calculated.")


def execute_find_freq_and_amp_combined(
    inputParameters: dict[str, object], session_folders: list[str], moving_window: int, numProcesses: int
) -> None:
    """Compute transient frequency and amplitude for combined (multi-session) data.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    session_folders : list of str
        Session folder paths.
    moving_window : int
        Moving-window size in seconds for transient detection.
    numProcesses : int
        Number of parallel worker processes.
    """
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders = []
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        run_folders.append(select_run_folders(filepath, selected_runs.get(filepath)))
    run_folders = list(np.concatenate(run_folders).flatten())
    combined_output_groups = get_all_stores_for_combining_data(run_folders)
    for i in range(len(combined_output_groups)):
        filepath = combined_output_groups[i][0]
        store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
        findFreqAndAmp(filepath, inputParameters, window=moving_window, numProcesses=numProcesses)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1


def execute_average_for_group(inputParameters: dict[str, object], group_session_folders: list[str]) -> None:
    """Average transient frequency and amplitude results across all group sessions.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    group_session_folders : list of str
        Session folder paths selected for group averaging.

    Raises
    ------
    ValueError
        When ``group_session_folders`` is empty.
    """
    run_folders = gather_group_run_folders(inputParameters, group_session_folders)
    averageForGroup(run_folders, inputParameters)
    writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
    inputParameters["step"] += 1


@subprocess_main_handler
def main(input_parameters: dict[str, object]) -> None:
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
