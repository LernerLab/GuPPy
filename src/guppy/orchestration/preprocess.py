import glob
import logging
import os

import numpy as np

from .save_parameters import read_artifact_provenance, save_parameters
from ..analysis.artifact_removal import remove_artifacts
from ..analysis.combine_data import combine_data
from ..analysis.control_channel import add_control_channel, create_control_channel
from ..analysis.io_utils import (
    check_storeslistfile,
    check_TDT,
    decide_naming_convention,
    get_coords,
    read_hdf5,
    recording_site_from_channel_path,
    write_combined_stores_list,
)
from ..analysis.standard_io import (
    read_control_and_signal,
    read_coords_pairwise,
    read_corrected_data,
    read_corrected_data_dict,
    read_corrected_timestamps_pairwise,
    read_corrected_ttl_timestamps,
    read_data_for_combining_data,
    read_timestamps_for_combining_data,
    read_ttl,
    read_ttl_timestamps_for_combining_data,
    write_artifact_removal,
    write_combined_data,
    write_corrected_data,
    write_corrected_timestamps,
    write_corrected_ttl_timestamps,
    write_zscore,
)
from ..analysis.timestamp_correction import correct_timestamps
from ..analysis.z_score import compute_z_score
from ..utils import progress
from ..utils.progress import step_error_handler
from ..utils.utils import (
    get_all_stores_for_combining_data,
    resolve_run_folders,
    select_run_folders,
)
from ..utils.validation import validate_artifact_coords_present

logger = logging.getLogger(__name__)


def execute_timestamp_correction(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """
    Apply timestamp correction to all session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories to process.
    inputParameters : dict
        Pipeline configuration; must include ``'timeForLightsTurnOn'`` and
        ``'isosbestic_control'``.
    """

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    isosbestic_control = inputParameters["isosbestic_control"]

    selected_runs = inputParameters.get("selected_runs") or {}
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        run_folders = select_run_folders(filepath, selected_runs.get(filepath))
        mode = "tdt" if check_TDT(session_folders[i]) else "csv"
        logger.debug(f"Timestamps corrections started for {filepath}")
        for j in range(len(run_folders)):
            filepath = run_folders[j]
            store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(
                2, -1
            )

            if isosbestic_control == False:
                store_array = add_control_channel(filepath, store_array)

            control_and_signal_dicts = read_control_and_signal(filepath, store_array)
            store_label_to_data, store_label_to_timestamps, store_label_to_sampling_rate, store_label_to_npoints = (
                control_and_signal_dicts
            )
            store_label_to_timestamps_ttl = read_ttl(filepath, store_array)

            timestamps_dicts = correct_timestamps(
                timeForLightsTurnOn,
                store_array,
                store_label_to_timestamps,
                store_label_to_data,
                store_label_to_sampling_rate,
                store_label_to_npoints,
                store_label_to_timestamps_ttl,
                mode=mode,
            )
            (
                store_label_to_corrected_timestamps,
                store_label_to_correction_index,
                store_label_to_corrected_data,
                compound_name_to_corrected_ttl_timestamps,
            ) = timestamps_dicts

            write_corrected_timestamps(
                filepath,
                store_label_to_corrected_timestamps,
                store_label_to_timestamps,
                store_label_to_sampling_rate,
                store_label_to_correction_index,
                mode,
            )
            write_corrected_data(filepath, store_label_to_corrected_data)
            write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)

            # check if isosbestic control is false and also if new control channel is added
            if isosbestic_control == False:
                create_control_channel(filepath, store_array, window=101)

            progress.advance()
        logger.info(f"Timestamps corrections finished for {filepath}")


def execute_zscore(session_folders: list[str], inputParameters: dict[str, object], *, remove_artifacts: bool) -> None:
    """
    Compute z-score and dF/F for all channel pairs across session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories (or combined-output folder lists when ``combine_data`` is True).
    inputParameters : dict
        Pipeline configuration; must include ``'filter_window'``, ``'isosbestic_control'``,
        ``'zscore_method'``, ``'control_fit_method'``, ``'baselineWindowStart'``,
        ``'baselineWindowEnd'``, ``'photobleaching_detrend'``, and ``'combine_data'``.
    remove_artifacts : bool
        When True, chunk the signal on each run's saved artifact windows; when False,
        compute over the full recording.
    """

    combine_data = inputParameters["combine_data"]
    filter_window = inputParameters["filter_window"]
    isosbestic_control = inputParameters["isosbestic_control"]
    zscore_method = inputParameters["zscore_method"]
    control_fit_method = inputParameters["control_fit_method"]
    baseline_start, baseline_end = inputParameters["baselineWindowStart"], inputParameters["baselineWindowEnd"]
    control_fit_window_mode = inputParameters["controlFitWindowMode"]
    control_fit_window_start = inputParameters["controlFitWindowStart"]
    control_fit_window_end = inputParameters["controlFitWindowEnd"]
    photobleaching_detrend = inputParameters["photobleaching_detrend"]

    if control_fit_window_mode == "baseline epoch" and isosbestic_control == False:
        raise ValueError(
            "controlFitWindowMode='baseline epoch' requires an isosbestic control channel, but "
            "isosbestic_control is False. Enable the isosbestic control channel, or set the control fit "
            "window back to 'full trace'."
        )

    if photobleaching_detrend == True and isosbestic_control == False:
        raise ValueError(
            "photobleaching_detrend=True requires an isosbestic control channel, but isosbestic_control "
            "is False. Without one the synthetic control is itself an exponential fit to the signal, so "
            "the photobleaching trend has already been removed. Enable the isosbestic control channel, "
            "or set photobleaching detrending back to False."
        )

    if photobleaching_detrend == True and control_fit_method != "OLS":
        raise ValueError(
            f"photobleaching_detrend=True requires control_fit_method='OLS', but it is "
            f"'{control_fit_method}'. Adding the photobleaching term makes the control fit nonlinear, and "
            "the nonlinear solver has no robust variant. Set the control fit method to 'OLS', or set "
            "photobleaching detrending back to False."
        )

    run_folders = []
    for i in range(len(session_folders)):
        if combine_data == True:
            run_folders.append([session_folders[i][0]])
        else:
            filepath = session_folders[i]
            run_folders.append(select_run_folders(filepath, (inputParameters.get("selected_runs") or {}).get(filepath)))
    run_folders = np.concatenate(run_folders)

    for j in range(len(run_folders)):
        filepath = run_folders[j]
        logger.debug(f"Computing z-score for each of the data in {filepath}")
        path = decide_naming_convention(filepath)
        _, artifactsRemovalMethod = read_artifact_provenance(destination=filepath)

        for i in range(path.shape[1]):
            name = recording_site_from_channel_path(path[0, i])

            control, signal, tsNew = read_corrected_data(path[0, i], path[1, i], filepath, name)
            coords = get_coords(filepath, name, tsNew, remove_artifacts)
            z_score, dff, control_fit, control_array = compute_z_score(
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
                control_fit_method,
                control_fit_window_mode,
                control_fit_window_start,
                control_fit_window_end,
                photobleaching_detrend,
            )
            write_zscore(filepath, name, z_score, dff, control_fit, control_array)

        logger.info(f"z-score for the data in {filepath} computed.")
        progress.advance()

    logger.info("Z-score computation completed.")


def execute_artifact_removal(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """
    Apply artifact removal to all session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories to process.
    inputParameters : dict
        Pipeline configuration; must include ``'timeForLightsTurnOn'`` and ``'combine_data'``.
        The removal method is read from each run's recorded provenance.
    """

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    combine_data = inputParameters["combine_data"]

    run_folders = []
    for i in range(len(session_folders)):
        if combine_data == True:
            run_folders.append([session_folders[i][0]])
        else:
            filepath = session_folders[i]
            run_folders.append(select_run_folders(filepath, (inputParameters.get("selected_runs") or {}).get(filepath)))

    run_folders = np.concatenate(run_folders)

    for j in range(len(run_folders)):
        filepath = run_folders[j]
        store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
        _, artifactsRemovalMethod = read_artifact_provenance(destination=filepath)

        store_label_to_data = read_corrected_data_dict(filepath, store_array)
        pair_name_to_tsNew, pair_name_to_sampling_rate = read_corrected_timestamps_pairwise(filepath)
        pair_name_to_coords = read_coords_pairwise(filepath, pair_name_to_tsNew)
        compound_name_to_ttl_timestamps = read_corrected_ttl_timestamps(filepath, store_array)

        logger.debug("Removing artifacts from the data...")
        store_label_to_data, pair_name_to_timestamps, compound_name_to_ttl_timestamps = remove_artifacts(
            timeForLightsTurnOn,
            store_array,
            pair_name_to_tsNew,
            pair_name_to_sampling_rate,
            pair_name_to_coords,
            store_label_to_data,
            compound_name_to_ttl_timestamps,
            method=artifactsRemovalMethod,
        )

        write_artifact_removal(filepath, store_label_to_data, pair_name_to_timestamps, compound_name_to_ttl_timestamps)

        progress.advance()

    logger.info("Artifact removal completed.")


def execute_combine_data(
    session_folders: list[str], inputParameters: dict[str, object], store_array: np.ndarray
) -> list:
    """
    Concatenate data from multiple session files and save the result to the first output folder.

    Parameters
    ----------
    session_folders : list of str
        Session directories whose output subdirectories are to be combined.
    inputParameters : dict
        Pipeline configuration; must include ``'timeForLightsTurnOn'``.
    store_array : np.ndarray
        2-D store array with rows [store_id, store_label].

    Returns
    -------
    combined_output_groups : list
        List of ``[output_filepath, ...]`` entries for combined output directories.
    """
    logger.debug("Combining Data from different data files...")
    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders = []
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        run_folders.append(select_run_folders(filepath, selected_runs.get(filepath)))

    run_folders = list(np.concatenate(run_folders).flatten())
    sampling_rate_filepaths = []
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        session_run_folders = select_run_folders(filepath, selected_runs.get(filepath))
        for j in range(len(session_run_folders)):
            filepath = session_run_folders[j]
            storesList_new = np.genfromtxt(
                os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=","
            ).reshape(2, -1)
            sampling_rate_filepaths.append(glob.glob(os.path.join(filepath, "timeCorrection_*")))

    # check if sampling rate is same for both data
    sampling_rate_filepaths = np.concatenate(sampling_rate_filepaths)
    sampling_rate = []
    for i in range(sampling_rate_filepaths.shape[0]):
        sampling_rate.append(read_hdf5("", sampling_rate_filepaths[i], "sampling_rate"))

    all_sampling_rates_equal = all(i == sampling_rate[0] for i in sampling_rate)
    if all_sampling_rates_equal == False:
        rates = [float(np.asarray(rate).reshape(-1)[0]) for rate in sampling_rate]
        message = (
            f"Cannot combine data: sampling rates differ across files {sampling_rate_filepaths.tolist()} "
            f"(rates={rates}). All files being combined must share the same sampling rate."
        )
        logger.error(message)
        raise ValueError(message)

    # get the output folders informatinos
    combined_output_groups = get_all_stores_for_combining_data(run_folders)

    # processing timestamps for combining the data
    for filepaths_to_combine in combined_output_groups:
        pair_name_to_filepath_to_timestamps = read_timestamps_for_combining_data(filepaths_to_combine)
        store_label_to_filepath_to_data = read_data_for_combining_data(filepaths_to_combine, store_array)
        compound_name_to_filepath_to_ttl_timestamps = read_ttl_timestamps_for_combining_data(
            filepaths_to_combine, store_array
        )
        pair_name_to_tsNew, store_label_to_data, compound_name_to_ttl_timestamps = combine_data(
            filepaths_to_combine,
            pair_name_to_filepath_to_timestamps,
            store_label_to_filepath_to_data,
            compound_name_to_filepath_to_ttl_timestamps,
            timeForLightsTurnOn,
            store_array,
            sampling_rate[0],
        )
        output_filepath = filepaths_to_combine[0]
        write_combined_data(output_filepath, pair_name_to_tsNew, store_label_to_data, compound_name_to_ttl_timestamps)
    logger.info("Data is combined from different data files.")

    return combined_output_groups


def _start_progress(inputParameters: dict[str, object], *, passes_per_folder: int) -> None:
    """
    Size the progress bar for a preprocessing run.

    Each pass reports one unit per folder it visits, so the denominator is the folder
    count times the number of passes. Combining collapses the run folders into one
    output folder per run name, so only timestamp correction visits every folder.

    Parameters
    ----------
    inputParameters : dict
        Pipeline configuration; must include ``'session_folders'`` and ``'combine_data'``.
    passes_per_folder : int
        Number of passes that will run over each folder.
    """
    session_folders = inputParameters["session_folders"]
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders = []
    for i in range(len(session_folders)):
        run_folders.append(select_run_folders(session_folders[i], selected_runs.get(session_folders[i])))
    run_folders = np.concatenate(run_folders)

    if inputParameters["combine_data"] == False:
        progress.start(run_folders.shape[0] * passes_per_folder)
    else:
        combined_group_count = len(get_all_stores_for_combining_data(list(run_folders.flatten())))
        progress.start(run_folders.shape[0] + combined_group_count * (passes_per_folder - 1))


def _correct_and_maybe_combine(session_folders: list[str], inputParameters: dict[str, object]) -> list:
    """
    Run timestamp correction, then combining when enabled.

    Parameters
    ----------
    session_folders : list of str
        Session directories to process.
    inputParameters : dict
        Pipeline configuration; must include ``'combine_data'``.

    Returns
    -------
    list
        The folder structure the z-score and artifact-removal passes consume: the
        session folders, or the combined-output folder groups when combining.
    """
    execute_timestamp_correction(session_folders, inputParameters)
    if inputParameters["combine_data"] == False:
        return session_folders

    store_array = check_storeslistfile(session_folders)
    combined_output_folders = execute_combine_data(session_folders, inputParameters, store_array)
    write_combined_stores_list(combined_output_folders, store_array)
    return combined_output_folders


def extractTsAndSignal(inputParameters: dict[str, object]) -> None:
    """
    Orchestrate step-3 preprocessing (timestamp correction and z-score over the full recording).

    Parameters
    ----------
    inputParameters : dict
        Full pipeline configuration, including ``'session_folders'``,
        ``'timeForLightsTurnOn'``, ``'isosbestic_control'``, and ``'combine_data'``.
    """

    logger.debug("Extracting signal data and event timestamps...")

    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration.
    save_parameters(inputParameters=inputParameters, remove_artifacts=False)

    session_folders = inputParameters["session_folders"]
    logger.info(f"Combine Data : {inputParameters['combine_data']}")
    logger.info(f"Isosbestic Control Channel : {inputParameters['isosbestic_control']}")

    _start_progress(inputParameters, passes_per_folder=2)
    folders = _correct_and_maybe_combine(session_folders, inputParameters)
    execute_zscore(folders, inputParameters, remove_artifacts=False)


def removeArtifactsFromSignal(inputParameters: dict[str, object]) -> None:
    """
    Orchestrate the optional Remove Artifacts step.

    Re-runs timestamp correction and z-score against the artifact windows saved by the
    Select Artifact Windows step, then excises the artifacts from the corrected data,
    timestamps, and TTLs.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline configuration, including ``'session_folders'``,
        ``'timeForLightsTurnOn'``, ``'isosbestic_control'``, and ``'combine_data'``.
    """

    logger.debug("Removing artifacts from signal data...")

    session_folders = inputParameters["session_folders"]
    validate_artifact_coords_present(run_folders=resolve_run_folders(session_folders, inputParameters))

    save_parameters(inputParameters=inputParameters, remove_artifacts=True)

    logger.info(f"Combine Data : {inputParameters['combine_data']}")
    logger.info(f"Isosbestic Control Channel : {inputParameters['isosbestic_control']}")

    _start_progress(inputParameters, passes_per_folder=3)
    folders = _correct_and_maybe_combine(session_folders, inputParameters)
    execute_zscore(folders, inputParameters, remove_artifacts=True)
    execute_artifact_removal(folders, inputParameters)


@step_error_handler
def run_preprocess_step(input_parameters: dict[str, object]) -> None:
    """Run the step-3 preprocessing step with failure reporting attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    extractTsAndSignal(input_parameters)


@step_error_handler
def run_remove_artifacts_step(input_parameters: dict[str, object]) -> None:
    """Run the optional Remove Artifacts step with failure reporting attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    removeArtifactsFromSignal(input_parameters)
