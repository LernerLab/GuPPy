import glob
import logging
import multiprocessing as mp
import os

import numpy as np

from .group_utils import gather_group_run_folders
from ..analysis.binned_metrics import compute_binned_metrics
from ..analysis.io_utils import (
    metric_from_preprocessed_label,
    read_hdf5,
    recording_site_from_preprocessed_label,
)
from ..analysis.standard_io import (
    write_binned_metrics_to_csv,
    write_binned_metrics_to_hdf5,
    write_freq_and_amp_to_csv,
    write_freq_and_amp_to_hdf5,
    write_transients_as_event_to_hdf5,
    write_transients_to_hdf5,
)
from ..analysis.transients import analyze_transients
from ..analysis.transients_average import averageForGroup
from ..utils import progress
from ..utils.utils import (
    get_all_stores_for_combining_data,
    select_run_folders,
)

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
    useTransientsAsEvents = inputParameters["useTransientsAsEvents"]

    if selectForTransientsComputation == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForTransientsComputation == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    # Occurrence times per recording site per metric, kept for the binned metrics
    # below; with "Both" selected each site is visited twice, so they are
    # accumulated here and consumed once the loop is done.
    site_to_transient_timestamps = {}

    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = recording_site_from_preprocessed_label(basename)
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
        site_to_transient_timestamps.setdefault(name_1, {})[metric_from_preprocessed_label(basename)] = (
            peaks_occurrences[:, 0]
        )
        if useTransientsAsEvents == True:
            write_transients_as_event_to_hdf5(filepath, basename, peaks_occurrences[:, 0])
    logger.info("Frequency and amplitude of transients in z_score data are calculated.")

    if inputParameters["computeBinnedMetrics"] == True:
        findBinnedMetrics(filepath, inputParameters, site_to_transient_timestamps)


def findBinnedMetrics(
    filepath: str, inputParameters: dict[str, object], site_to_transient_timestamps: dict[str, dict[str, np.ndarray]]
) -> None:
    """Reduce each recording site to one row per fixed-width time bin.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    inputParameters : dict
        Full pipeline input parameters; uses ``binnedMetricsWidth``.
    site_to_transient_timestamps : dict
        Detected transient occurrence times (s), keyed by recording site and then
        by the metric they were detected on.
    """
    logger.debug("Computing binned metrics over the whole session....")
    bin_width = inputParameters["binnedMetricsWidth"]

    for recording_site in sorted(site_to_transient_timestamps):
        # Read the traces fresh: the detection loop above rebinds its own
        # z_score/timestamps to the NaN-stripped arrays, which carry a compressed
        # time axis that would misplace every bin edge.
        timestamps = read_hdf5("timeCorrection_" + recording_site, filepath, "timestampNew")
        z_score = read_hdf5("z_score_" + recording_site, filepath, "data")
        dff = read_hdf5("dff_" + recording_site, filepath, "data")

        binned_metrics = compute_binned_metrics(
            z_score=z_score,
            dff=dff,
            timestamps=timestamps,
            transient_timestamps=site_to_transient_timestamps[recording_site],
            bin_width=bin_width,
        )
        write_binned_metrics_to_hdf5(filepath, binned_metrics, recording_site)
        write_binned_metrics_to_csv(filepath, binned_metrics, recording_site)

    logger.info("Binned metrics computed.")


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
    elif combine_data == True:
        execute_find_freq_and_amp_combined(inputParameters, session_folders, moving_window, numProcesses)
    else:
        execute_find_freq_and_amp(inputParameters, session_folders, moving_window, numProcesses)

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
            progress.advance()
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
        progress.advance()


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
    progress.advance()
