import glob
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .save_parameters import save_parameters
from ..analysis.io_utils import (
    is_channel_label,
    is_continuous_label,
    recording_site_from_channel_label,
    recording_site_from_preprocessed_label,
)
from ..frontend.parameterized_plotter import (
    ParameterizedPlotter,
    overview_y_options,
    remove_cols,
)
from ..frontend.visualization_dashboard import VisualizationDashboard
from ..utils.utils import (
    event_labels_for_analysis,
    get_all_stores_for_combining_data,
    read_Df,
    select_run_folders,
)
from ..utils.validation import validate_group_definitions

logger = logging.getLogger(__name__)


def helper_plots(filepath: str, event: list[str], name: list[str], inputParameters: dict[str, object]) -> None:
    """Build and display the interactive PSTH visualization dashboard for one output directory.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : list of str
        Event names.
    name : list of str
        z-score/dff file basenames paired with ``event``.
    inputParameters : dict
        Full pipeline input parameters.
    """
    basename = os.path.basename(filepath)
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]

    # note when there are no behavior event TTLs
    if len(event) == 0:
        logger.warning("There are no behavior event TTLs present to visualize.")
        return 0

    if os.path.exists(os.path.join(filepath, "cross_correlation_output")):
        event_corr, frames = [], []
        if visualize_zscore_or_dff == "z_score":
            corr_fp = glob.glob(os.path.join(filepath, "cross_correlation_output", "*_z_score_*"))
        elif visualize_zscore_or_dff == "dff":
            corr_fp = glob.glob(os.path.join(filepath, "cross_correlation_output", "*_dff_*"))
        for i in range(len(corr_fp)):
            filename = os.path.basename(corr_fp[i]).split(".")[0]
            event_corr.append(filename)
            df = pd.read_hdf(corr_fp[i], key="df", mode="r")
            frames.append(df)
        if len(frames) > 0:
            df_corr = pd.concat(frames, keys=event_corr, axis=1)
        else:
            event_corr = []
            df_corr = []
    else:
        event_corr = []
        df_corr = None

    # combine all the event PSTH so that it can be viewed together
    event_name = event
    new_event, frames, bins = [], [], {}
    for i in range(len(event_name)):

        for j in range(len(name)):
            new_event.append(event_name[i] + "_" + recording_site_from_preprocessed_label(name[j]))
            new_name = name[j]
            event_df = read_Df(filepath, new_event[-1], new_name)
            columns = list(event_df.columns)
            regex = re.compile("bin_[(]")
            bins[new_event[-1]] = [columns[i] for i in range(len(columns)) if regex.match(columns[i])]
            frames.append(event_df)

    df = pd.concat(frames, keys=new_event, axis=1)

    if isinstance(df_corr, pd.DataFrame):
        new_event.extend(event_corr)
        df = pd.concat([df, df_corr], axis=1, sort=False).reset_index()

    columns_dict = dict()
    for i in range(len(new_event)):
        df_1 = df[new_event[i]]
        columns = list(df_1.columns)
        columns.append("All")
        columns_dict[new_event[i]] = columns

    # make options array for different selectors
    multiple_plots_options = []
    heatmap_options = new_event
    bins_keys = list(bins.keys())
    if len(bins_keys) > 0:
        bins_new = bins
        for i in range(len(bins_keys)):
            bin_columns = bins[bins_keys[i]]
            if len(bin_columns) > 0:
                for j in bin_columns:
                    multiple_plots_options.append("{}_{}".format(bins_keys[i], j))

        multiple_plots_options = new_event + multiple_plots_options
    else:
        multiple_plots_options = new_event
    # Default the x-axis to the actual PSTH window (nSecPrev is negative by
    # convention) so the traces fill the plot; users can still type/zoom beyond it.
    x_min = float(inputParameters["nSecPrev"])
    x_max = float(inputParameters["nSecPost"])
    colormaps = plt.colormaps()
    new_colormaps = ["plasma", "plasma_r", "magma", "magma_r", "inferno", "inferno_r", "viridis", "viridis_r"]
    all_colormaps_set = set(colormaps)
    preferred_colormaps_set = set(new_colormaps)
    colormaps = new_colormaps + list(all_colormaps_set.difference(preferred_colormaps_set))
    x = [columns_dict[new_event[0]][-4]]
    y = overview_y_options(columns_dict[new_event[0]])
    trial_no = range(1, len(remove_cols(columns_dict[heatmap_options[0]])[:-2]) + 1)
    trial_ts = [
        "{} - {}".format(i, j) for i, j in zip(trial_no, remove_cols(columns_dict[heatmap_options[0]])[:-2])
    ] + ["All"]

    plotter = ParameterizedPlotter(
        event_selector_objects=new_event,
        event_selector_heatmap_objects=heatmap_options,
        selector_for_multipe_events_plot_objects=multiple_plots_options,
        columns_dict=columns_dict,
        df_new=df,
        x_min=x_min,
        x_max=x_max,
        color_map_objects=colormaps,
        filepath=filepath,
        x_objects=x,
        y_objects=y,
        heatmap_y_objects=trial_ts,
        psth_y_objects=trial_ts[:-1],
    )
    dashboard = VisualizationDashboard(plotter=plotter, basename=basename)
    dashboard.show()


def createPlots(filepath: str, event: list[str], inputParameters: dict[str, object]) -> None:
    """Assemble PSTH data from an output directory and delegate to ``helper_plots``.

    Parameters
    ----------
    filepath : str
        Path to an output directory: a session run folder or a group folder.
    event : list of str
        Store labels (row 1 of store_array) to include in the visualization.
    inputParameters : dict
        Full pipeline input parameters.
    """
    for i in range(len(event)):
        event[i] = event[i].replace("\\", "_")
        event[i] = event[i].replace("/", "_")

    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]

    index = []
    for i in range(len(event)):
        if is_continuous_label(event[i]):
            index.append(i)

    event = np.delete(event, index)

    names = [f"{visualize_zscore_or_dff}_{site}" for site in _recording_sites(filepath)]

    helper_plots(filepath, event, names, inputParameters)


def _recording_sites(filepath: str) -> list[str]:
    """Return the recording-site names an output directory holds results for.

    Read from ``storesList.csv`` rather than from the preprocessed trace filenames, so a
    directory that holds averaged results but no traces of its own (a group) needs no
    stand-in files to name its sites.

    Parameters
    ----------
    filepath : str
        Path to an output directory: a session run folder or a group folder.

    Returns
    -------
    list of str
        Recording-site names, in the order their channels appear in ``storesList.csv``.
    """
    store_array = np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)
    sites = []
    for label in store_array[1, :]:
        if not is_channel_label(label):
            continue
        site = recording_site_from_channel_label(label)
        if site not in sites:
            sites.append(site)
    return sites


def _validate_metric_against_step4_outputs(inputParameters: dict[str, object]) -> None:
    """Cross-check the visualization metric selection against step-4 PSTH outputs on disk.

    Step 4 only writes PSTH ``.h5`` files for the metric(s) selected via
    ``selectForComputePsth``.  If the user later requests a different metric in
    step 5 the downstream ``read_Df`` call will fail with an opaque
    ``FileNotFoundError``.  This function detects that mismatch early and raises
    a :class:`ValueError` that names the offending sessions and tells the user
    exactly how to fix the problem.

    Parameters
    ----------
    inputParameters : dict
        The full input-parameters dict passed to :func:`visualizeResults`.

    Raises
    ------
    ValueError
        When one or more output directories are missing PSTH ``.h5`` files for
        the requested visualization metric.
    """
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]
    session_folders = inputParameters["session_folders"]

    # Collect every output directory that will be visualised: the selected session runs
    # plus the selected groups, which are visualised the same way.
    run_folders = list(inputParameters.get("selected_group_folders") or [])
    selected_runs = inputParameters.get("selected_runs") or {}
    for filepath in session_folders:
        runs = selected_runs.get(filepath)
        if not runs:
            # Session not in selected_runs (e.g. it has no _output_* dirs yet, which the
            # homepage gate `validate_selected_runs_for_consumers` skips). Nothing to validate.
            continue
        run_folders.extend(select_run_folders(filepath, runs))

    if not run_folders:
        return  # Nothing to check; the main function will handle the empty case.

    # PSTH output files use the ".h5" extension (pandas HDF5) and embed the
    # metric name, e.g. "<event>_z_score_<recording_site>.h5" or "<event>_dff_<recording_site>.h5".
    # Step-3 z-score/dff files use ".hdf5" and are therefore never false-positives.
    if visualize_zscore_or_dff == "z_score":
        pattern = "*_z_score_*.h5"
    else:
        pattern = "*_dff_*.h5"

    missing_sessions = [run_folder for run_folder in run_folders if not glob.glob(os.path.join(run_folder, pattern))]

    if missing_sessions:
        other_metric = "dff" if visualize_zscore_or_dff == "z_score" else "z_score"
        session_lines = "\n  - ".join(missing_sessions)
        raise ValueError(
            f"The visualization metric '{visualize_zscore_or_dff}' was not computed "
            f"for {len(missing_sessions)} output director(ies):\n"
            f"  - {session_lines}\n\n"
            f"To fix this, either:\n"
            f"  1. Change the visualization selection to '{other_metric}', or\n"
            f"  2. Re-run step 4 (or, for a '_group' directory, the Group Analysis step) "
            f"with '{visualize_zscore_or_dff}' (or 'Both') enabled."
        )


def visualizeResults(inputParameters: dict[str, object]) -> None:
    """Entry point for step-5 visualization: validate preconditions and open dashboards.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Raises
    ------
    ValueError
        When a selected group directory is not usable, or when the visualization
        metric was not computed in step 4.
    """
    inputParameters = inputParameters

    _validate_metric_against_step4_outputs(inputParameters)
    group_folders = list(inputParameters.get("selected_group_folders") or [])
    validate_group_definitions(group_folders=group_folders)

    combine_data = inputParameters["combine_data"]
    selected_runs = inputParameters.get("selected_runs") or {}
    # A session with no selected run is skipped rather than fatal: visualizing a group on
    # its own is a legitimate request that leaves the individual selection empty.
    session_folders = [session for session in inputParameters["session_folders"] if selected_runs.get(session)]

    if not session_folders and not group_folders:
        message = (
            "Nothing is selected to visualize. Pick at least one output directory in the Output "
            "Folder Selection panel, or at least one group in the Group Output Folder Selection panel."
        )
        logger.error(message)
        raise ValueError(message)

    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration. This
    # iterates the individual sessions only, so a group's own snapshot keeps recording
    # how it was averaged.
    if session_folders:
        save_parameters(inputParameters={**inputParameters, "session_folders": session_folders})
    if combine_data == True:
        run_folders = []
        for i in range(len(session_folders)):
            filepath = session_folders[i]
            run_folders.append(select_run_folders(filepath, selected_runs.get(filepath)))
        run_folders = list(np.concatenate(run_folders).flatten())
        combined_output_groups = get_all_stores_for_combining_data(run_folders)
        for i in range(len(combined_output_groups)):
            store_array = np.asarray([[], []])
            for j in range(len(combined_output_groups[i])):
                store_array = np.concatenate(
                    (
                        store_array,
                        np.genfromtxt(
                            os.path.join(combined_output_groups[i][j], "storesList.csv"),
                            dtype="str",
                            delimiter=",",
                        ).reshape(2, -1),
                    ),
                    axis=1,
                )
            store_array = np.unique(store_array, axis=1)
            filepath = combined_output_groups[i][0]
            createPlots(
                filepath,
                event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
                inputParameters,
            )
    else:
        for i in range(len(session_folders)):
            filepath = session_folders[i]
            run_folders = select_run_folders(filepath, selected_runs.get(filepath))
            for j in range(len(run_folders)):
                filepath = run_folders[j]
                store_array = np.genfromtxt(
                    os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=","
                ).reshape(2, -1)

                createPlots(
                    filepath,
                    event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
                    inputParameters,
                )

    # Groups are ordinary output directories to the visualizer: one dashboard each,
    # opened alongside any selected session runs rather than instead of them.
    for group_folder in group_folders:
        store_array = np.genfromtxt(os.path.join(group_folder, "storesList.csv"), dtype="str", delimiter=",").reshape(
            2, -1
        )
        createPlots(
            group_folder,
            event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
            inputParameters,
        )
