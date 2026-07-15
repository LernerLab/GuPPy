import glob
import json
import logging
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .save_parameters import save_parameters
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
from ..frontend.artifact_removal import ArtifactRemovalWidget
from ..frontend.progress import PB_STEPS_FILE, subprocess_main_handler, writeToFile
from ..utils.utils import (
    get_all_stores_for_combining_data,
    is_headless,
    select_run_folders,
)
from ..visualization.preprocessing import visualize_preprocessing

logger = logging.getLogger(__name__)

# Only set matplotlib backend if not in CI or headless (GUPPY_BASE_DIR) environment
if not os.getenv("CI") and not is_headless():
    plt.switch_backend("TKAgg")


def execute_preprocessing_visualization(filepath: str, visualization_type: Literal["z_score", "dff"]) -> None:
    """
    Plot z-score or dF/F signals for all channel pairs in a session output directory.

    Parameters
    ----------
    filepath : str
        Session output directory containing ``timeCorrection_*`` and signal HDF5 files.
    visualization_type : {'z_score', 'dff'}
        Which preprocessed signal to visualize.
    """
    name = os.path.basename(filepath)

    path = glob.glob(os.path.join(filepath, f"{visualization_type}_*"))

    path = sorted(path)

    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        # Strip the fixed "z_score_"/"dff_" prefix so recording-site names may contain underscores.
        name_1 = basename[len(visualization_type) + 1 :]
        x = read_hdf5("timeCorrection_" + name_1, filepath, "timestampNew")
        y = read_hdf5("", path[i], "data")
        fig, ax = visualize_preprocessing(suptitle=name, title=basename, x=x, y=y)


def visualizeControlAndSignal(filepath: str, removeArtifacts: bool) -> list:
    """
    Build artifact-removal widgets for each control/signal pair in a session directory.

    Parameters
    ----------
    filepath : str
        Session output directory containing ``control_*`` and ``signal_*`` HDF5 files.
    removeArtifacts : bool
        When True, the widget is shown in artifact-review mode; when False it allows
        the user to draw new artifact boundaries.

    Returns
    -------
    widgets : list of ArtifactRemovalWidget
        One widget per channel pair.
    """
    path = decide_naming_convention(filepath)

    widgets = []
    for i in range(path.shape[1]):

        name_1 = recording_site_from_channel_path(path[0, i])

        ts_path = os.path.join(filepath, "timeCorrection_" + name_1 + ".hdf5")
        cntrl_sig_fit_path = os.path.join(filepath, "cntrl_sig_fit_" + name_1 + ".hdf5")
        timestamps = read_hdf5("", ts_path, "timestampNew")

        control = read_hdf5("", path[0, i], "data").reshape(-1)
        signal = read_hdf5("", path[1, i], "data").reshape(-1)
        cntrl_sig_fit = read_hdf5("", cntrl_sig_fit_path, "data").reshape(-1)

        plot_name = [
            (os.path.basename(path[0, i])).split(".")[0],
            (os.path.basename(path[1, i])).split(".")[0],
            (os.path.basename(cntrl_sig_fit_path)).split(".")[0],
        ]
        widget = ArtifactRemovalWidget(filepath, timestamps, control, signal, cntrl_sig_fit, plot_name, removeArtifacts)
        widgets.append(widget)
    return widgets


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
            )
            write_corrected_data(filepath, store_label_to_corrected_data)
            write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)

            # check if isosbestic control is false and also if new control channel is added
            if isosbestic_control == False:
                create_control_channel(filepath, store_array, window=101)

            writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
            inputParameters["step"] += 1
        logger.info(f"Timestamps corrections finished for {filepath}")


def execute_zscore(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """
    Compute z-score and dF/F for all channel pairs across session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories (or combined-output folder lists when ``combine_data`` is True).
    inputParameters : dict
        Pipeline configuration; must include ``'filter_window'``, ``'isosbestic_control'``,
        ``'zscore_method'``, ``'control_fit_method'``, ``'baselineWindowStart'``,
        ``'baselineWindowEnd'``, ``'removeArtifacts'``, ``'artifactsRemovalMethod'``,
        and ``'combine_data'``.
    """

    plot_zScore_dff = inputParameters["plot_zScore_dff"]
    combine_data = inputParameters["combine_data"]
    remove_artifacts = inputParameters["removeArtifacts"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    filter_window = inputParameters["filter_window"]
    isosbestic_control = inputParameters["isosbestic_control"]
    zscore_method = inputParameters["zscore_method"]
    control_fit_method = inputParameters["control_fit_method"]
    baseline_start, baseline_end = inputParameters["baselineWindowStart"], inputParameters["baselineWindowEnd"]

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
            )
            write_zscore(filepath, name, z_score, dff, control_fit, control_array)

        logger.info(f"z-score for the data in {filepath} computed.")
        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1

    if not is_headless():
        plt.show()
    logger.info("Z-score computation completed.")


def visualize_z_score(inputParameters: dict[str, object], session_folders: list[str]) -> None:
    """
    Display control/signal plots and z-score/dF/F visualizations for all sessions.

    Parameters
    ----------
    inputParameters : dict
        Pipeline configuration; must include ``'plot_zScore_dff'``, ``'combine_data'``,
        and ``'removeArtifacts'``.
    session_folders : list of str
        Session directories to visualize.
    """
    plot_zScore_dff = inputParameters["plot_zScore_dff"]
    combine_data = inputParameters["combine_data"]
    remove_artifacts = inputParameters["removeArtifacts"]

    run_folders = []
    for i in range(len(session_folders)):
        if combine_data == True:
            run_folders.append([session_folders[i][0]])
        else:
            filepath = session_folders[i]
            run_folders.append(select_run_folders(filepath, (inputParameters.get("selected_runs") or {}).get(filepath)))
    run_folders = np.concatenate(run_folders)

    widgets = []
    for j in range(len(run_folders)):
        filepath = run_folders[j]

        if not remove_artifacts:
            # a reference to widgets has to persist in the same scope as plt.show() is called
            widgets.extend(visualizeControlAndSignal(filepath, removeArtifacts=remove_artifacts))

        if plot_zScore_dff == "z_score":
            execute_preprocessing_visualization(filepath, visualization_type="z_score")
        if plot_zScore_dff == "dff":
            execute_preprocessing_visualization(filepath, visualization_type="dff")
        if plot_zScore_dff == "Both":
            execute_preprocessing_visualization(filepath, visualization_type="z_score")
            execute_preprocessing_visualization(filepath, visualization_type="dff")

    plt.show()
    logger.info("Visualization of z-score and dF/F completed.")


def execute_artifact_removal(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """
    Apply artifact removal to all session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories to process.
    inputParameters : dict
        Pipeline configuration; must include ``'timeForLightsTurnOn'``,
        ``'artifactsRemovalMethod'``, and ``'combine_data'``.
    """

    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
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

        writeToFile(str(10 + ((inputParameters["step"] + 1) * 10)) + "\n", file_path=PB_STEPS_FILE)
        inputParameters["step"] += 1

    if not is_headless():
        visualize_artifact_removal(session_folders, inputParameters)
    logger.info("Artifact removal completed.")


def visualize_artifact_removal(session_folders: list[str], inputParameters: dict[str, object]) -> None:
    """
    Display control/signal plots after artifact removal for all sessions.

    Parameters
    ----------
    session_folders : list of str
        Session directories to visualize.
    inputParameters : dict
        Pipeline configuration; must include ``'combine_data'``.
    """
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
        visualizeControlAndSignal(filepath, removeArtifacts=True)
    plt.show()
    logger.info("Visualization of artifact removal completed.")


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


def extractTsAndSignal(inputParameters: dict[str, object]) -> None:
    """
    Orchestrate the full preprocessing pipeline (timestamp correction, z-score, artifact removal).

    Parameters
    ----------
    inputParameters : dict
        Full pipeline configuration, including ``'session_folders'``, ``'timeForLightsTurnOn'``,
        ``'isosbestic_control'``, ``'removeArtifacts'``, and ``'combine_data'``.
    """

    logger.debug("Extracting signal data and event timestamps...")
    inputParameters = inputParameters

    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration.
    save_parameters(inputParameters=inputParameters)

    session_folders = inputParameters["session_folders"]
    timeForLightsTurnOn = inputParameters["timeForLightsTurnOn"]
    isosbestic_control = inputParameters["isosbestic_control"]
    remove_artifacts = inputParameters["removeArtifacts"]
    combine_data = inputParameters["combine_data"]

    inputParameters["step"] = 0
    logger.info(f"Remove Artifacts : {remove_artifacts}")
    logger.info(f"Combine Data : {combine_data}")
    logger.info(f"Isosbestic Control Channel : {isosbestic_control}")
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders = []
    for i in range(len(session_folders)):
        run_folders.append(select_run_folders(session_folders[i], selected_runs.get(session_folders[i])))
    run_folders = np.concatenate(run_folders)
    if combine_data == False:
        pbMaxValue = run_folders.shape[0] + len(session_folders)
        writeToFile(str((pbMaxValue + 1) * 10) + "\n" + str(10) + "\n", file_path=PB_STEPS_FILE)
        execute_timestamp_correction(session_folders, inputParameters)
        execute_zscore(session_folders, inputParameters)
        if not is_headless():
            visualize_z_score(inputParameters, session_folders)
        if remove_artifacts == True:
            execute_artifact_removal(session_folders, inputParameters)
    else:
        pbMaxValue = 1 + len(session_folders)
        writeToFile(str((pbMaxValue) * 10) + "\n" + str(10) + "\n", file_path=PB_STEPS_FILE)
        execute_timestamp_correction(session_folders, inputParameters)
        store_array = check_storeslistfile(session_folders)
        combined_output_folders = execute_combine_data(session_folders, inputParameters, store_array)
        write_combined_stores_list(combined_output_folders, store_array)
        execute_zscore(combined_output_folders, inputParameters)
        if not is_headless():
            visualize_z_score(inputParameters, combined_output_folders)
        if remove_artifacts == True:
            execute_artifact_removal(combined_output_folders, inputParameters)


@subprocess_main_handler
def main(input_parameters: dict[str, object]) -> None:
    """Subprocess entry point for the preprocessing step.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters deserialized from the subprocess argument.
    """
    extractTsAndSignal(input_parameters)


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
