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

from .group_utils import gather_group_stores_list_paths
from .save_parameters import save_parameters
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
from ..frontend.progress import PB_STEPS_FILE, subprocess_main_handler, writeToFile
from ..utils.utils import get_all_stores_for_combining_data, read_Df, select_output_dirs
from ..utils.validation import validate_peak_windows, validate_window_bounds

logger = logging.getLogger(__name__)


def execute_compute_psth(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save the PSTH for a single event in one session output folder.

    Parameters
    ----------
    filepath : str
        Path to the session output directory (e.g. ``<session>_output_1``).
    event : str
        Raw event name from storesList row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
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
    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]

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
        timestamps = read_hdf5(event + "_" + name_1, filepath, "ts")
        if use_time_or_trials == "Time (min)" and bin_psth_trials > 0:
            corrected_timestamps = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        else:
            corrected_timestamps = None
        psth, psth_baselineUncorrected, columns, timestamps = compute_psth(
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
            timestamps,
            corrected_timestamps,
            timeForLightsTurnOn,
        )
        write_hdf5(timestamps, event + "_" + name_1, filepath, "ts")

        create_Df_for_psth(
            filepath,
            event + "_" + name_1 + "_baselineUncorrected",
            basename,
            psth_baselineUncorrected,
            columns=columns,
        )
        create_Df_for_psth(filepath, event + "_" + name_1, basename, psth, columns=columns)
        logger.info(f"PSTH for event {event} computed.")


def execute_compute_psth_peak_and_area(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save PSTH peak and area for a single event.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Raw event name from storesList row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if "control" in event.lower() or "signal" in event.lower():
        return 0

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
        columns = list(psth.columns)
        regex = re.compile("bin_[(]")
        bin_names = [columns[i] for i in range(len(columns)) if regex.match(columns[i])]
        regex_trials = re.compile("[+-]?([0-9]*[.])?[0-9]+")
        trials_names = [columns[i] for i in range(len(columns)) if regex_trials.match(columns[i])]
        psth_mean_bin_names = trials_names + bin_names + ["mean"]
        psth_mean_bin_mean = np.asarray(psth[psth_mean_bin_names])
        timestamps = np.asarray(psth["timestamps"]).ravel()
        peak_area = compute_psth_peak_and_area(
            psth_mean_bin_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint
        )
        fileName = [os.path.basename(os.path.dirname(filepath))]
        index = [fileName[0] + "_" + name for name in psth_mean_bin_names]
        write_peak_and_area_to_hdf5(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
        write_peak_and_area_to_csv(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
        logger.info(f"Peak and Area for PSTH mean signal for event {event} computed.")


def execute_compute_cross_correlation(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save cross-correlation between brain regions for a single event.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Raw event name from storesList row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
    isCompute = inputParameters["computeCorr"]
    removeArtifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    if isCompute == True:
        if removeArtifacts == True and artifactsRemovalMethod == "concatenate":
            raise ValueError(
                "For cross-correlation, when removeArtifacts is True, the artifacts removal method "
                "must be 'replace with NaNs' and not 'concatenate'. Change 'Method for Artifact "
                "Removal' in the Input Parameters GUI."
            )
        corr_info, type = getCorrCombinations(filepath, inputParameters)
        if len(corr_info) < 2:
            if corr_info:
                raise ValueError(
                    f"Cross-correlation requires at least two distinct signal regions, but only one was "
                    f"found: '{corr_info[0]}'. Please either disable compute_cross_correlation or add a "
                    f"second signal region in step 1."
                )
            else:
                raise ValueError(
                    "Cross-correlation requires at least two distinct signal regions, but no signal "
                    "regions were found. Please either disable compute_cross_correlation or add signal "
                    "regions in step 1."
                )
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
                    columns_a, columns_b = np.array(psth_a.columns), np.array(psth_b.columns)
                    if np.intersect1d(columns_a, columns_b).size > 0:
                        columns = list(np.intersect1d(columns_a, columns_b))
                    else:
                        columns = list(columns_a)
                    psth_array_a, psth_array_b = np.array(psth_a).T, np.array(psth_b).T
                    cross_corr = compute_cross_correlation(psth_array_a, psth_array_b, sample_rate)
                    columns.append("timestamps")
                    create_Df_for_cross_correlation(
                        make_dir_for_cross_correlation(filepath),
                        "corr_" + event,
                        type[j] + "_" + corr_info[i - 1] + "_" + corr_info[i],
                        cross_corr,
                        columns,
                    )
                logger.info(f"Cross-correlation for event {event} computed.")


def orchestrate_psth(inputParameters: dict[str, object]) -> None:
    """Run PSTH, peak/area, and cross-correlation for each individual session.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    selected_outputs = inputParameters.get("selectedOutputs") or {}
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(select_output_dirs(folderNames[i], selected_outputs.get(folderNames[i])))
    storesListPath = np.concatenate(storesListPath)
    writeToFile(
        str((storesListPath.shape[0] + storesListPath.shape[0] + 1) * 10) + "\n" + str(10) + "\n",
        file_path=PB_STEPS_FILE,
    )
    for i in range(len(folderNames)):
        logger.debug(f"Computing PSTH, Peak and Area for each event in {folderNames[i]}")
        storesListPath = select_output_dirs(folderNames[i], selected_outputs.get(folderNames[i]))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )

            with mp.Pool(numProcesses) as psth_pool:
                psth_pool.starmap(
                    execute_compute_psth, zip(repeat(filepath), storesList[1, :], repeat(inputParameters))
                )

            with mp.Pool(numProcesses) as peak_area_pool:
                peak_area_pool.starmap(
                    execute_compute_psth_peak_and_area, zip(repeat(filepath), storesList[1, :], repeat(inputParameters))
                )

            with mp.Pool(numProcesses) as cross_correlation_pool:
                cross_correlation_pool.starmap(
                    execute_compute_cross_correlation, zip(repeat(filepath), storesList[1, :], repeat(inputParameters))
                )

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
            inputParameters["step"] += 1
        logger.info(f"PSTH, Area and Peak are computed for all events in {folderNames[i]}.")


def execute_psth_combined(inputParameters: dict[str, object]) -> None:
    """Run PSTH computation for combined (multi-session) data.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    folderNames = inputParameters["folderNames"]
    selected_outputs = inputParameters.get("selectedOutputs") or {}
    storesListPath = []
    for i in range(len(folderNames)):
        storesListPath.append(select_output_dirs(folderNames[i], selected_outputs.get(folderNames[i])))
    storesListPath = list(np.concatenate(storesListPath).flatten())
    combined_output_groups = get_all_stores_for_combining_data(storesListPath)
    writeToFile(
        str((len(combined_output_groups) + len(combined_output_groups) + 1) * 10) + "\n" + str(10) + "\n",
        file_path=PB_STEPS_FILE,
    )
    for i in range(len(combined_output_groups)):
        storesList = np.asarray([[], []])
        for j in range(len(combined_output_groups[i])):
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(
                        os.path.join(combined_output_groups[i][j], "storesList.csv"), dtype="str", delimiter=","
                    ).reshape(2, -1),
                ),
                axis=1,
            )
        storesList = np.unique(storesList, axis=1)
        for k in range(storesList.shape[1]):
            execute_compute_psth(combined_output_groups[i][0], storesList[1, k], inputParameters)
            execute_compute_psth_peak_and_area(combined_output_groups[i][0], storesList[1, k], inputParameters)
            execute_compute_cross_correlation(combined_output_groups[i][0], storesList[1, k], inputParameters)
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1


def _validate_fiber_regions_consistent_for_group(storesListPath: np.ndarray) -> None:
    """Check that every session shares the same fiber (control/signal) storenames.

    Group averaging buckets each session's data by its fiber-region basename
    (``z_score_<region>`` / ``dff_<region>``) and averages each behavioral event
    independently, skipping sessions that lack a given event.  Sessions may
    therefore differ in their *event* storenames — that is the intended
    cross-condition workflow (e.g. ``novelobject`` sessions averaged alongside
    ``novelfemale1`` sessions).  What must agree is the set of *fiber* storenames:
    averaging across different brain regions produces meaningless per-region
    single-session "averages".  Detect that mismatch up-front and raise a clear
    error listing the offending sessions.

    Fiber storenames follow the codebase-wide convention that their names contain
    ``control`` or ``signal``; every other storename is treated as a behavioral
    event and ignored here.

    Raises
    ------
    ValueError
        When the sessions disagree on the set of fiber (control/signal) storenames.
    """
    per_session_fibers = {}
    for output_dir in storesListPath:
        session_stores_list = np.genfromtxt(
            os.path.join(output_dir, "storesList.csv"), dtype="str", delimiter=","
        ).reshape(2, -1)
        fiber_stores = tuple(
            sorted(
                name for name in set(session_stores_list[1, :]) if "control" in name.lower() or "signal" in name.lower()
            )
        )
        per_session_fibers[output_dir] = fiber_stores

    unique_fiber_sets = set(per_session_fibers.values())
    if len(unique_fiber_sets) <= 1:
        return

    session_lines = "\n".join(
        f"  - {os.path.basename(os.path.dirname(output_dir))}: "
        f"{', '.join(stores) if stores else '(no control/signal storenames)'}"
        for output_dir, stores in per_session_fibers.items()
    )
    raise ValueError(
        "Group averaging requires every selected session to share the same fiber "
        "regions, but the selected sessions have mismatched control/signal "
        "storenames:\n"
        f"{session_lines}\n"
        "Event storenames may differ across sessions, but the control/signal "
        "storenames must match. Fix the storename labels in step 1, deselect the "
        "mismatched sessions, or disable 'Average Group? (bool)'."
    )


def _validate_psth_window_parameters(inputParameters: dict[str, object]) -> None:
    """Upfront PSTH-window validation, run before any HDF5 IO.

    Why: peak-window ordering used to surface only deep inside
    ``compute_psth_peak_and_area`` (after step 4 had begun), and the PSTH
    baseline-correction window had no equivalent of the z-score baseline
    validation added in PR #283. Catching both here gives the user a Panel
    notification before progress starts.
    """
    validate_peak_windows(
        peak_starts=inputParameters["peak_startPoint"],
        peak_ends=inputParameters["peak_endPoint"],
    )
    baselineCorrectionStart = inputParameters["baselineCorrectionStart"]
    baselineCorrectionEnd = inputParameters["baselineCorrectionEnd"]
    # (0, 0) is the documented sentinel for "skip baseline correction"
    # (see baselineCorrection in compute_psth.py and the GUI tooltip).
    if baselineCorrectionStart == 0 and baselineCorrectionEnd == 0:
        return
    validate_window_bounds(
        start=baselineCorrectionStart,
        end=baselineCorrectionEnd,
        ts_min=float(inputParameters["nSecPrev"]),
        ts_max=float(inputParameters["nSecPost"]),
        start_name="baselineCorrectionStart",
        end_name="baselineCorrectionEnd",
        range_label="PSTH window",
    )


def execute_average_for_group(inputParameters: dict[str, object]) -> None:
    """Average PSTH results across all selected sessions in the group.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; must contain a non-empty
        ``folderNamesForAvg`` list.

    Raises
    ------
    ValueError
        When ``folderNamesForAvg`` is empty or the fiber (control/signal)
        storenames are inconsistent across sessions.
    """
    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    storesListPath = gather_group_stores_list_paths(inputParameters, folderNamesForAvg)

    _validate_fiber_regions_consistent_for_group(storesListPath)

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
    average_dir = makeAverageDir(inputParameters["abspath"])
    np.savetxt(os.path.join(average_dir, "storesList.csv"), storesList, delimiter=",", fmt="%s")
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
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1


def psthForEachStorename(inputParameters: dict[str, object]) -> dict[str, object]:
    """Entry point for step-4 PSTH computation: validates parameters and dispatches to the appropriate sub-routine.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    dict
        The same ``inputParameters`` dict, potentially updated with the step
        counter used for progress tracking.
    """
    logger.info("Computing PSTH, Peak and Area for each event...")
    inputParameters = inputParameters

    _validate_psth_window_parameters(inputParameters)

    average = inputParameters["averageForGroup"]
    combine_data = inputParameters["combine_data"]
    numProcesses = inputParameters["numberOfCores"]
    inputParameters["step"] = 0

    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration.
    # Group runs aggregate over the average/ dir rather than the individual
    # selectedOutputs, so skip the snapshot there (steps 2-3 already wrote it per session).
    if not average:
        save_parameters(inputParameters=inputParameters)
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            f"Number of cores requested ({numProcesses}) exceeds available cores "
            f"({mp.cpu_count()}); using {mp.cpu_count() - 1}."
        )
        numProcesses = mp.cpu_count() - 1

    logger.info(f"Average for group : {average}")

    # Group-average analysis aggregates PSTHs across all sessions in the group.
    if average == True:
        execute_average_for_group(inputParameters)

    # Otherwise each session is analyzed individually.
    else:
        if combine_data == True:
            execute_psth_combined(inputParameters)
        else:
            orchestrate_psth(inputParameters)
    logger.info("PSTH, Area and Peak are computed for all events.")
    return inputParameters


@subprocess_main_handler
def main(input_parameters: dict[str, object]) -> None:
    """Run step-4 PSTH computation and chain to the transients step.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters deserialized from the subprocess argument.
    """
    inputParameters = psthForEachStorename(input_parameters)
    subprocess.call([sys.executable, "-m", "guppy.orchestration.transients", json.dumps(inputParameters)])


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
