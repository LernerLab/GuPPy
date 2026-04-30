import glob
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..frontend.parameterized_plotter import ParameterizedPlotter, remove_cols
from ..frontend.visualization_dashboard import VisualizationDashboard
from ..utils.utils import get_all_stores_for_combining_data, read_Df, takeOnlyDirs

logger = logging.getLogger(__name__)


def helper_plots(filepath, event, name, inputParameters):
    """Build and display the interactive PSTH visualization dashboard for one output directory.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : list of str
        Event names (or z-score/dff file basenames when ``name`` is empty).
    name : list of str or str
        z-score/dff file basenames paired with ``event``; pass an empty string
        for the average-results code path.
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
    if name:
        event_name, name = event, name
        new_event, frames, bins = [], [], {}
        for i in range(len(event_name)):

            for j in range(len(name)):
                new_event.append(event_name[i] + "_" + name[j].split("_")[-1])
                new_name = name[j]
                temp_df = read_Df(filepath, new_event[-1], new_name)
                cols = list(temp_df.columns)
                regex = re.compile("bin_[(]")
                bins[new_event[-1]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
                # bins.append(keep_cols)
                frames.append(temp_df)

        df = pd.concat(frames, keys=new_event, axis=1)
    else:
        new_event = list(np.unique(np.array(event)))
        frames, bins = [], {}
        for i in range(len(new_event)):
            temp_df = read_Df(filepath, new_event[i], "")
            cols = list(temp_df.columns)
            regex = re.compile("bin_[(]")
            bins[new_event[i]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
            frames.append(temp_df)

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
            arr = bins[bins_keys[i]]
            if len(arr) > 0:
                # heatmap_options.append('{}_bin'.format(bins_keys[i]))
                for j in arr:
                    multiple_plots_options.append("{}_{}".format(bins_keys[i], j))

        multiple_plots_options = new_event + multiple_plots_options
    else:
        multiple_plots_options = new_event
    x_min = float(inputParameters["nSecPrev"]) - 20
    x_max = float(inputParameters["nSecPost"]) + 20
    colormaps = plt.colormaps()
    new_colormaps = ["plasma", "plasma_r", "magma", "magma_r", "inferno", "inferno_r", "viridis", "viridis_r"]
    set_a = set(colormaps)
    set_b = set(new_colormaps)
    colormaps = new_colormaps + list(set_a.difference(set_b))
    x = [columns_dict[new_event[0]][-4]]
    y = remove_cols(columns_dict[new_event[0]])
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


def createPlots(filepath, event, inputParameters):
    """Assemble PSTH data from an output directory and delegate to ``helper_plots``.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : list of str
        Storenames (row 1 of storesList) to include in the visualization.
    inputParameters : dict
        Full pipeline input parameters.
    """
    for i in range(len(event)):
        event[i] = event[i].replace("\\", "_")
        event[i] = event[i].replace("/", "_")

    average = inputParameters["visualizeAverageResults"]
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]

    if average == True:
        path = []
        for i in range(len(event)):
            if visualize_zscore_or_dff == "z_score":
                path.append(glob.glob(os.path.join(filepath, event[i] + "*_z_score_*")))
            elif visualize_zscore_or_dff == "dff":
                path.append(glob.glob(os.path.join(filepath, event[i] + "*_dff_*")))

        path = np.concatenate(path)
    else:
        if visualize_zscore_or_dff == "z_score":
            path = glob.glob(os.path.join(filepath, "z_score_*"))
        elif visualize_zscore_or_dff == "dff":
            path = glob.glob(os.path.join(filepath, "dff_*"))

    name_arr = []
    event_arr = []

    index = []
    for i in range(len(event)):
        if "control" in event[i].lower() or "signal" in event[i].lower():
            index.append(i)

    event = np.delete(event, index)

    for i in range(len(path)):
        name = (os.path.basename(path[i])).split(".")
        name = name[0]
        name_arr.append(name)

    if average == True:
        logger.info("average")
        helper_plots(filepath, name_arr, "", inputParameters)
    else:
        helper_plots(filepath, event, name_arr, inputParameters)


def _validate_metric_against_step5_outputs(inputParameters):
    """Cross-check the visualization metric selection against step-5 PSTH outputs on disk.

    Step 5 only writes PSTH ``.h5`` files for the metric(s) selected via
    ``selectForComputePsth``.  If the user later requests a different metric in
    step 6 the downstream ``read_Df`` call will fail with an opaque
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
    average = inputParameters["visualizeAverageResults"]
    folderNames = inputParameters["folderNames"]
    folderNamesForAvg = inputParameters["folderNamesForAvg"]

    # Collect all output directories that will be visualised
    output_dirs = []
    source_folders = folderNamesForAvg if (average and len(folderNamesForAvg) > 0) else folderNames
    for filepath in source_folders:
        output_dirs.extend(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))

    if not output_dirs:
        return  # Nothing to check; the main function will handle the empty case.

    # PSTH output files use the ".h5" extension (pandas HDF5) and embed the
    # metric name, e.g. "ttl_region_z_score_region.h5" or "ttl_region_dff_region.h5".
    # Step-4 z-score/dff files use ".hdf5" and are therefore never false-positives.
    if visualize_zscore_or_dff == "z_score":
        pattern = "*_z_score_*.h5"
    else:
        pattern = "*_dff_*.h5"

    missing_sessions = [od for od in output_dirs if not glob.glob(os.path.join(od, pattern))]

    if missing_sessions:
        other_metric = "dff" if visualize_zscore_or_dff == "z_score" else "z_score"
        session_lines = "\n  - ".join(missing_sessions)
        raise ValueError(
            f"The visualization metric '{visualize_zscore_or_dff}' was not computed "
            f"in step 5 for {len(missing_sessions)} session(s):\n"
            f"  - {session_lines}\n\n"
            f"To fix this, either:\n"
            f"  1. Change the visualization selection to '{other_metric}', or\n"
            f"  2. Re-run step 5 with '{visualize_zscore_or_dff}' (or 'Both') enabled."
        )


def _validate_average_visualization_preconditions(inputParameters):
    """Ensure the prerequisites for 'Visualize Average Results' are satisfied.

    Catches the three user-facing failure modes documented in issue #274:

    1. ``visualizeAverageResults`` is True, but no folders are selected in the
       group-analysis folder picker — previously the visualization silently
       fell through to individual mode.
    2. ``visualizeAverageResults`` is True, but step 5 was never run with
       ``averageForGroup`` = True, so no ``average/`` directory exists —
       previously only a terminal warning was logged.
    3. ``visualizeAverageResults`` is True and an ``average/`` folder exists,
       but the folders selected for averaging are not reflected in the saved
       averaged outputs (e.g. the user deselected them after running step 5).

    Raises
    ------
    ValueError
        With a message pointing the user at the specific corrective action.
    """
    if not inputParameters["visualizeAverageResults"]:
        return

    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    if len(folderNamesForAvg) == 0:
        raise ValueError(
            "'Visualize Average Results?' is set to True, but no folders are "
            "selected in the Group Analysis folder picker. Please either "
            "select the folders to visualize in the group-analysis selector, "
            "or set 'Visualize Average Results?' to False for individual "
            "visualization."
        )

    average_folder = os.path.join(inputParameters["abspath"], "average")
    if not os.path.isdir(average_folder):
        raise ValueError(
            "'Visualize Average Results?' is set to True, but no 'average' "
            f"directory was found at {average_folder}. Please re-run step 5 "
            "('PSTH Computation') with 'Average Group? (bool)' = True before "
            "visualizing the averaged results."
        )

    # Ensure the average folder contains PSTH outputs; otherwise step 5 was
    # run without averageForGroup=True even though the folder exists from some
    # earlier run.
    visualize_zscore_or_dff = inputParameters["visualize_zscore_or_dff"]
    if visualize_zscore_or_dff == "z_score":
        pattern = "*_z_score_*.h5"
    else:
        pattern = "*_dff_*.h5"
    if not glob.glob(os.path.join(average_folder, pattern)):
        raise ValueError(
            f"'Visualize Average Results?' is set to True and an 'average' "
            f"directory exists at {average_folder}, but it contains no PSTH "
            f"outputs for the '{visualize_zscore_or_dff}' metric. Please "
            "re-run step 5 ('PSTH Computation') with 'Average Group? (bool)' "
            "= True and the appropriate 'z_score and/or ΔF/F? (psth)' "
            "selection before visualizing the averaged results."
        )


def visualizeResults(inputParameters):
    """Entry point for step-6 visualization: validate preconditions and open dashboards.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Raises
    ------
    ValueError
        When average visualization is requested but prerequisites are not met,
        or when the visualization metric was not computed in step 5.
    """
    inputParameters = inputParameters

    _validate_average_visualization_preconditions(inputParameters)
    _validate_metric_against_step5_outputs(inputParameters)

    average = inputParameters["visualizeAverageResults"]
    logger.info(average)

    folderNames = inputParameters["folderNames"]
    folderNamesForAvg = inputParameters["folderNamesForAvg"]
    combine_data = inputParameters["combine_data"]

    if average == True and len(folderNamesForAvg) > 0:
        # folderNames = folderNamesForAvg
        filepath_avg = os.path.join(inputParameters["abspath"], "average")
        # filepath = os.path.join(inputParameters['abspath'], folderNames[0])
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
                    np.genfromtxt(
                        os.path.join(storesListPath[i], "storesList.csv"), dtype="str", delimiter=","
                    ).reshape(2, -1),
                ),
                axis=1,
            )
        storesList = np.unique(storesList, axis=1)

        createPlots(filepath_avg, np.unique(storesList[1, :]), inputParameters)

    else:
        if combine_data == True:
            storesListPath = []
            for i in range(len(folderNames)):
                filepath = folderNames[i]
                storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
            storesListPath = list(np.concatenate(storesListPath).flatten())
            op = get_all_stores_for_combining_data(storesListPath)
            for i in range(len(op)):
                storesList = np.asarray([[], []])
                for j in range(len(op[i])):
                    storesList = np.concatenate(
                        (
                            storesList,
                            np.genfromtxt(os.path.join(op[i][j], "storesList.csv"), dtype="str", delimiter=",").reshape(
                                2, -1
                            ),
                        ),
                        axis=1,
                    )
                storesList = np.unique(storesList, axis=1)
                filepath = op[i][0]
                createPlots(filepath, storesList[1, :], inputParameters)
        else:
            for i in range(len(folderNames)):

                filepath = folderNames[i]
                storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
                for j in range(len(storesListPath)):
                    filepath = storesListPath[j]
                    storesList = np.genfromtxt(
                        os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=","
                    ).reshape(2, -1)

                    createPlots(filepath, storesList[1, :], inputParameters)


# logger.info(sys.argv[1:])
# visualizeResults(sys.argv[1:][0])
