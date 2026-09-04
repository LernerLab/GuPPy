import glob
import logging
import multiprocessing as mp
import os
import re
from itertools import repeat

import numpy as np
from scipy import signal as ss

from .psth_significance import execute_compute_psth_significance
from .save_parameters import read_artifact_provenance, save_parameters
from .transients import executeFindFreqAndAmp
from ..analysis.compute_psth import compute_psth
from ..analysis.cross_correlation import compute_cross_correlation
from ..analysis.io_utils import (
    is_continuous_label,
    make_dir_for_cross_correlation,
    read_hdf5,
    recording_site_from_preprocessed_label,
    write_hdf5,
)
from ..analysis.psth_peak_and_area import compute_psth_peak_and_area
from ..analysis.psth_utils import (
    create_Df_for_cross_correlation,
    create_Df_for_psth,
    getCorrCombinations,
    match_trials_by_timestamp,
)
from ..analysis.standard_io import (
    write_peak_and_area_to_csv,
    write_peak_and_area_to_hdf5,
)
from ..utils import progress
from ..utils.progress import step_error_handler
from ..utils.stores_list import read_stores_list
from ..utils.utils import (
    event_labels_for_analysis,
    get_all_stores_for_combining_data,
    read_Df,
    resolve_run_folders,
    select_run_folders,
    transient_event_labels,
)
from ..utils.validation import (
    validate_peak_windows,
    validate_psth_comparisons,
    validate_window_bounds,
)

logger = logging.getLogger(__name__)


def execute_compute_psth(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save the PSTH for a single event in one session output folder.

    Parameters
    ----------
    filepath : str
        Path to the session output directory (e.g. ``<session>_output_1``).
    event : str
        Raw event name from store_array row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if is_continuous_label(event):
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
        logger.info("Computing PSTH for event %s...", event)
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = recording_site_from_preprocessed_label(basename)
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
        recordingStart = read_hdf5("timeCorrection_" + name_1, filepath, "recordingStart")[0]
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
            recordingStart,
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
        logger.info("PSTH for event %s computed.", event)


def execute_compute_psth_peak_and_area(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save PSTH peak and area for a single event.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Raw event name from store_array row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if is_continuous_label(event):
        return 0

    peak_startPoint = inputParameters["peak_startPoint"]
    peak_endPoint = inputParameters["peak_endPoint"]
    auc_units = inputParameters["auc_units"]
    selectForComputePsth = inputParameters["selectForComputePsth"]

    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    for i in range(len(path)):
        logger.info("Computing peak and area for PSTH mean signal for event %s...", event)
        basename = (os.path.basename(path[i])).split(".")[0]
        name_1 = recording_site_from_preprocessed_label(basename)
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
            psth_mean_bin_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint, auc_units=auc_units
        )
        fileName = [os.path.basename(os.path.dirname(filepath))]
        index = [fileName[0] + "_" + name for name in psth_mean_bin_names]
        write_peak_and_area_to_hdf5(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
        write_peak_and_area_to_csv(filepath, peak_area, event + "_" + name_1 + "_" + basename, index=index)
        logger.info("Peak and Area for PSTH mean signal for event %s computed.", event)


def execute_compute_cross_correlation(filepath: str, event: str, inputParameters: dict[str, object]) -> None:
    """Compute and save cross-correlation between recording sites for a single event.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Raw event name from store_array row 1.
    inputParameters : dict
        Full pipeline input parameters.
    """
    isCompute = inputParameters["computeCorr"]
    # Each recording site's transients are its own event train, so the two sites' PSTHs
    # share no trials and there is nothing to correlate.
    if event in transient_event_labels(inputParameters=inputParameters):
        return
    removeArtifacts, artifactsRemovalMethod = read_artifact_provenance(destination=filepath)
    if isCompute == True:
        if removeArtifacts == True and artifactsRemovalMethod == "concatenate":
            raise ValueError(
                f"Cross-correlation cannot run on concatenated data, but the outputs in '{filepath}' were "
                "produced by the Remove Artifacts step using the 'concatenate' method. Re-run Select "
                "Artifact Windows with the method set to 'replace with NaN' followed by Remove Artifacts, "
                "or disable Compute Cross-correlation."
            )
        corr_info, type = getCorrCombinations(filepath, inputParameters)
        if len(corr_info) < 2:
            if corr_info:
                raise ValueError(
                    f"Cross-correlation requires at least two distinct signal recording sites, but only one "
                    f"was found: '{corr_info[0]}'. Please either disable compute_cross_correlation or add a "
                    f"second signal recording site in step 1."
                )
            else:
                raise ValueError(
                    "Cross-correlation requires at least two distinct signal recording sites, but no signal "
                    "recording sites were found. Please either disable compute_cross_correlation or add "
                    "signal recording sites in step 1."
                )
        if is_continuous_label(event):
            return
        else:
            for i in range(1, len(corr_info)):
                logger.debug("Computing cross-correlation for event %s...", event)
                for j in range(len(type)):
                    psth_a = read_Df(filepath, event + "_" + corr_info[i - 1], type[j] + "_" + corr_info[i - 1])
                    psth_b = read_Df(filepath, event + "_" + corr_info[i], type[j] + "_" + corr_info[i])
                    sample_rate = 1 / (psth_a["timestamps"][1] - psth_a["timestamps"][0])
                    psth_a = psth_a.drop(columns=["timestamps", "err", "mean"])
                    psth_b = psth_b.drop(columns=["timestamps", "err", "mean"])
                    # Uneven artifact removal can leave the two recording sites with a
                    # slightly different set of surviving trials; pair them by event timestamp
                    # so only the trials both recording sites kept are correlated (and labeled).
                    indices_a, indices_b, matched_labels = match_trials_by_timestamp(
                        list(psth_a.columns), list(psth_b.columns)
                    )
                    if len(matched_labels) == 0:
                        raise ValueError(
                            f"No matching trials between recording sites '{corr_info[i - 1]}' and "
                            f"'{corr_info[i]}' for event '{event}'; cross-correlation cannot be computed."
                        )
                    if len(matched_labels) < max(len(psth_a.columns), len(psth_b.columns)):
                        logger.warning(
                            "Recording sites '%s' and '%s' have a different set of surviving trials for event '%s' "
                            "(uneven artifact removal): %s vs %s trials, %s matched. Cross-correlating only the matched "
                            "trials.",
                            corr_info[i - 1],
                            corr_info[i],
                            event,
                            len(psth_a.columns),
                            len(psth_b.columns),
                            len(matched_labels),
                        )
                    psth_array_a = np.array(psth_a).T[indices_a]
                    psth_array_b = np.array(psth_b).T[indices_b]
                    cross_corr = compute_cross_correlation(psth_array_a, psth_array_b, sample_rate)
                    columns = [str(label) for label in matched_labels]
                    columns.append("timestamps")
                    create_Df_for_cross_correlation(
                        make_dir_for_cross_correlation(filepath),
                        "corr_" + event,
                        type[j] + "_" + corr_info[i - 1] + "_" + corr_info[i],
                        cross_corr,
                        columns,
                    )
                logger.info("Cross-correlation for event %s computed.", event)


def orchestrate_psth(inputParameters: dict[str, object]) -> None:
    """Run PSTH, peak/area, and cross-correlation for each individual session.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    session_folders = inputParameters["session_folders"]
    numProcesses = inputParameters["numberOfCores"]
    # Pinned rather than inherited: this runs on a background thread of the Panel server
    # process, and forking a process that has other live threads can leave a lock they
    # held (logging, HDF5) permanently locked in the child.
    spawn_context = mp.get_context("spawn")
    selected_runs = inputParameters.get("selected_runs") or {}
    for i in range(len(session_folders)):
        logger.debug("Computing PSTH, Peak and Area for each event in %s", session_folders[i])
        run_folders = select_run_folders(session_folders[i], selected_runs.get(session_folders[i]))
        for j in range(len(run_folders)):
            filepath = run_folders[j]
            store_array = read_stores_list(run_folder=filepath)
            event_labels = event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters)

            # Each pool is closed and joined before leaving its block. The context manager's
            # __exit__ calls terminate(), which signals every worker and then blocks in waitpid()
            # until it is gone; a worker that is slow to exit never gets there and the parent waits
            # forever. starmap has already returned, so there is nothing to abort -- close() lets
            # each worker exit on its own.
            with spawn_context.Pool(numProcesses) as psth_pool:
                psth_pool.starmap(execute_compute_psth, zip(repeat(filepath), event_labels, repeat(inputParameters)))
                psth_pool.close()
                psth_pool.join()

            with spawn_context.Pool(numProcesses) as peak_area_pool:
                peak_area_pool.starmap(
                    execute_compute_psth_peak_and_area,
                    zip(repeat(filepath), event_labels, repeat(inputParameters)),
                )
                peak_area_pool.close()
                peak_area_pool.join()

            with spawn_context.Pool(numProcesses) as cross_correlation_pool:
                cross_correlation_pool.starmap(
                    execute_compute_cross_correlation, zip(repeat(filepath), event_labels, repeat(inputParameters))
                )
                cross_correlation_pool.close()
                cross_correlation_pool.join()

            progress.advance()

            execute_compute_psth_significance(filepath, inputParameters)
            if inputParameters["computePsthSignificance"]:
                progress.advance()
        logger.info("PSTH, Area and Peak are computed for all events in %s.", session_folders[i])


def execute_psth_combined(inputParameters: dict[str, object]) -> None:
    """Run PSTH computation for combined (multi-session) data.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    session_folders = inputParameters["session_folders"]
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders = []
    for i in range(len(session_folders)):
        run_folders.append(select_run_folders(session_folders[i], selected_runs.get(session_folders[i])))
    run_folders = list(np.concatenate(run_folders).flatten())
    combined_output_groups = get_all_stores_for_combining_data(run_folders)
    for i in range(len(combined_output_groups)):
        store_array = np.asarray([[], []])
        for j in range(len(combined_output_groups[i])):
            store_array = np.concatenate(
                (
                    store_array,
                    read_stores_list(run_folder=combined_output_groups[i][j]),
                ),
                axis=1,
            )
        store_array = np.unique(store_array, axis=1)
        for event in event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters):
            execute_compute_psth(combined_output_groups[i][0], event, inputParameters)
            execute_compute_psth_peak_and_area(combined_output_groups[i][0], event, inputParameters)
            execute_compute_cross_correlation(combined_output_groups[i][0], event, inputParameters)
        progress.advance()

        execute_compute_psth_significance(combined_output_groups[i][0], inputParameters)
        if inputParameters["computePsthSignificance"]:
            progress.advance()


def _validate_psth_significance_parameters(inputParameters: dict[str, object]) -> None:
    """Upfront validation of the PSTH significance parameters, run before any HDF5 IO.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Raises
    ------
    ValueError
        If a comparison row is half-filled or self-referential, or if the moving-average
        filter is disabled while significance testing is on.
    """
    if not inputParameters["computePsthSignificance"]:
        return

    validate_psth_comparisons(
        comparisons_a=inputParameters["psthComparisonsA"],
        comparisons_b=inputParameters["psthComparisonsB"],
    )

    # The significance run-length threshold is derived from the filter window, so with
    # filtering disabled every isolated significant sample would survive and the
    # multiple-comparisons control would silently vanish.
    if inputParameters["filter_window"] == 0:
        message = (
            "filter_window=0 disables the moving-average filter, but PSTH significance testing "
            "derives its minimum significant duration from it. Set a filter window, or turn off "
            "Compute PSTH Significance."
        )
        logger.error(message)
        raise ValueError(message)


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


def _validate_events_overlap_signal(inputParameters: dict[str, object]) -> None:
    """Upfront check that each event store shares a timeline with the signal, run before any worker.

    An event whose PSTH window only partly overlaps the signal is legitimate — ``rowFormation``
    NaN-pads it. An event whose window does not overlap at all can only ever produce an
    all-NaN trial, and a whole store of them means the events and the signal are on different
    clocks. That used to surface as an out-of-range index deep inside the PSTH pools.
    """
    nSecPrev = float(inputParameters["nSecPrev"])
    nSecPost = float(inputParameters["nSecPost"])
    selectForComputePsth = inputParameters["selectForComputePsth"]
    # Transient event files are written by executeFindFreqAndAmp, which runs after this check;
    # their timestamps are on the corrected basis by construction and cannot be off-clock.
    transient_labels = set(transient_event_labels(inputParameters=inputParameters))

    for filepath in resolve_run_folders(inputParameters["session_folders"], inputParameters):
        store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
        events = [
            event.replace("\\", "_").replace("/", "_")
            for event in event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters)
            if not is_continuous_label(event) and event not in transient_labels
        ]
        if not events:
            continue

        # Mirror the worker's site resolution so we validate exactly what will be computed.
        if selectForComputePsth == "z_score":
            preprocessed_paths = glob.glob(os.path.join(filepath, "z_score_*"))
        elif selectForComputePsth == "dff":
            preprocessed_paths = glob.glob(os.path.join(filepath, "dff_*"))
        else:
            preprocessed_paths = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(
                os.path.join(filepath, "dff_*")
            )
        recording_sites = dict.fromkeys(
            recording_site_from_preprocessed_label(os.path.basename(path).split(".")[0]) for path in preprocessed_paths
        )

        for name_1 in recording_sites:
            timestamps = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
            signal_start, signal_end = float(timestamps[0]), float(timestamps[-1])
            for event in events:
                event_timestamps = read_hdf5(event + "_" + name_1, filepath, "ts")
                # An event store with no timestamps is a separate problem, warned about downstream.
                if len(event_timestamps) == 0:
                    continue
                overlaps = (event_timestamps + nSecPost >= signal_start) & (event_timestamps + nSecPrev <= signal_end)
                if overlaps.any():
                    continue
                message = (
                    f"Event store '{event}' spans [{float(event_timestamps[0]):.6g}, "
                    f"{float(event_timestamps[-1]):.6g}]s, and with a PSTH window of "
                    f"[{nSecPrev:g}, {nSecPost:g}]s no trial overlaps the '{name_1}' signal, which spans "
                    f"[{signal_start:.6g}, {signal_end:.6g}]s in '{filepath}'. Every trial would be empty. "
                    "Events and signal must come from the same acquisition clock — re-run step 1 and "
                    "step 2 with the timestamp settings that put them on one timeline."
                )
                logger.error(message)
                raise ValueError(message)


def psthForEachStore(inputParameters: dict[str, object]) -> None:
    """Entry point for step-4 PSTH computation: validates parameters and dispatches to the appropriate sub-routine.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.
    """
    logger.info("Computing PSTH, Peak and Area for each event...")
    inputParameters = inputParameters

    _validate_psth_window_parameters(inputParameters)
    _validate_psth_significance_parameters(inputParameters)
    _validate_events_overlap_signal(inputParameters)

    combine_data = inputParameters["combine_data"]
    numProcesses = inputParameters["numberOfCores"]

    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration.
    save_parameters(inputParameters=inputParameters)
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            "Number of cores requested (%s) exceeds available cores (%s); using %s.",
            numProcesses,
            mp.cpu_count(),
            mp.cpu_count() - 1,
        )
        numProcesses = mp.cpu_count() - 1

    if combine_data == True:
        execute_psth_combined(inputParameters)
    else:
        orchestrate_psth(inputParameters)
    logger.info("PSTH, Area and Peak are computed for all events.")


def _start_step4_progress(input_parameters: dict[str, object]) -> None:
    """Declare the step-4 progress denominator for both halves of the step.

    Transient analysis and PSTH computation run under a single button and share one
    progress bar, so the denominator is declared once here rather than by whichever
    half happens to run first.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    # Two units per output directory: transient analysis, then PSTH. Significance
    # testing, when enabled, adds a third.
    run_folders = resolve_run_folders(input_parameters["session_folders"], input_parameters)
    units_per_run_folder = 3 if input_parameters["computePsthSignificance"] else 2
    progress.start(len(run_folders) * units_per_run_folder)


@step_error_handler
def run_psth_step(input_parameters: dict[str, object]) -> None:
    """Run step-4 transient analysis, then PSTH computation, with failure reporting attached.

    Transients run first because ``useTransientsAsEvents`` turns their timestamps into the
    event train the PSTH is computed against.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    _validate_psth_window_parameters(input_parameters)
    _validate_psth_significance_parameters(input_parameters)
    _validate_events_overlap_signal(input_parameters)
    _start_step4_progress(input_parameters)
    executeFindFreqAndAmp(input_parameters)
    psthForEachStore(input_parameters)
