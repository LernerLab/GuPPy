import logging
from pathlib import Path

import numpy as np

from .save_parameters import save_parameters
from ..analysis.io_utils import is_continuous_label
from ..frontend.parameterized_plotter import available_psth_metrics, build_plotter
from ..frontend.visualization_dashboard import (
    VisualizationDashboard,
    whole_session_result_tabs,
)
from ..utils import page_links
from ..utils.stores_list import read_stores_list
from ..utils.utils import (
    event_labels_for_analysis,
    get_all_stores_for_combining_data,
    relative_output_labels,
    select_run_folders,
)
from ..utils.validation import validate_group_definitions

logger = logging.getLogger(__name__)

# Glob patterns matching the PSTH result files step 4 writes, one per metric.
PSTH_FILE_PATTERNS = ("*_z_score_*.h5", "*_dff_*.h5")

# What step 5 can show, and the steps that produce it, for the error raised when a
# selected directory holds none of it.
NOTHING_TO_VISUALIZE_HINT = (
    "Step 5 shows PSTH results, which need behavioral events and come from step 4 (or, for a "
    "'_group' directory, from the Group Analysis step); binned metrics and covariate correlations "
    "from step 4; and results from the Tonic Analysis step."
)


def helper_plots(filepath: str, event: list[str], inputParameters: dict[str, object], *, label: str) -> None:
    """Build and display the interactive visualization dashboard for one output directory.

    A directory without PSTH results, for example a session with no behavioral events,
    still gets a dashboard when it holds Tonic Analysis results or binned metrics.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : list of str
        Event names.
    inputParameters : dict
        Full pipeline input parameters.
    """
    available_metrics = available_psth_metrics(filepath=filepath, events=list(event)) if len(event) > 0 else []
    if not available_metrics:
        if not whole_session_result_tabs(filepath=filepath):
            logger.warning(
                "Nothing to visualize was found in %s, so no dashboard is opened for it. %s",
                filepath,
                NOTHING_TO_VISUALIZE_HINT,
            )
            return
        dashboard = VisualizationDashboard(
            filepath=filepath,
            plotter=None,
            basename=label,
            events=[],
            metric=None,
            available_metrics=[],
        )
        page_links.offer_template(name=label, template=dashboard.build_template())
        return

    metric = available_metrics[0]
    plotter = build_plotter(
        filepath=filepath,
        events=event,
        metric=metric,
        # Default the x-axis to the actual PSTH window (nSecPrev is negative by
        # convention) so the traces fill the plot; users can still type/zoom beyond it.
        x_min=float(inputParameters["nSecPrev"]),
        x_max=float(inputParameters["nSecPost"]),
    )
    dashboard = VisualizationDashboard(
        filepath=filepath,
        plotter=plotter,
        basename=label,
        events=list(event),
        metric=metric,
        available_metrics=available_metrics,
    )
    page_links.offer_template(name=label, template=dashboard.build_template())


def createPlots(filepath: str, event: list[str], inputParameters: dict[str, object], *, label: str) -> None:
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

    index = []
    for i in range(len(event)):
        if is_continuous_label(event[i]):
            index.append(i)

    event = np.delete(event, index)

    helper_plots(filepath, event, inputParameters, label=label)


def _output_directories(inputParameters: dict[str, object]) -> list[str]:
    """Return every output directory step 5 will open a dashboard for.

    The selected session runs plus the selected groups, which are visualized the
    same way.

    Parameters
    ----------
    inputParameters : dict
        The full input-parameters dict passed to :func:`visualizeResults`.

    Returns
    -------
    list of str
        Paths to the run folders and group folders being visualized.
    """
    run_folders = list(inputParameters.get("selected_group_folders") or [])
    selected_runs = inputParameters.get("selected_runs") or {}
    for filepath in inputParameters["session_folders"]:
        if not selected_runs.get(filepath):
            # Session not in selected_runs (e.g. it holds no runs yet, which the homepage
            # gate `validate_selected_runs_for_consumers` skips). Nothing to visualize.
            continue
        run_folders.extend(select_run_folders(filepath, inputParameters=inputParameters))
    return run_folders


def _has_results_to_visualize(*, output_directory: str) -> bool:
    """Return whether ``output_directory`` holds anything a step-5 dashboard can show.

    Parameters
    ----------
    output_directory : str
        A session run folder or a group folder.

    Returns
    -------
    bool
        True when the directory holds PSTH results, Tonic Analysis results, or binned
        metrics.
    """
    # PSTH output files use the ".h5" extension (pandas HDF5) and embed the metric name,
    # e.g. "<event>_<site>_z_score_<site>.h5". Step-3 z-score/dff files use ".hdf5" and
    # are therefore never false-positives. The transient frequency/amplitude tables
    # ("freqAndAmp_z_score_<site>.h5") also carry the metric name, and step 4 writes them
    # whether or not the run has events, so they are excluded.
    has_psth_results = any(
        not path.name.startswith("freqAndAmp_")
        for pattern in PSTH_FILE_PATTERNS
        for path in Path(output_directory).glob(pattern)
    )
    return has_psth_results or bool(whole_session_result_tabs(filepath=output_directory))


def _validate_results_exist(inputParameters: dict[str, object]) -> None:
    """Check that at least one selected output directory holds results step 5 can show.

    Parameters
    ----------
    inputParameters : dict
        The full input-parameters dict passed to :func:`visualizeResults`.

    Raises
    ------
    ValueError
        When none of the selected output directories hold PSTH results, Tonic Analysis
        results, or binned metrics.
    """
    run_folders = _output_directories(inputParameters)

    if not run_folders:
        return  # Nothing to check; the main function will handle the empty case.

    if any(_has_results_to_visualize(output_directory=run_folder) for run_folder in run_folders):
        return

    directory_lines = "\n  - ".join(run_folders)
    raise ValueError(
        f"Nothing to visualize was found in any of the {len(run_folders)} selected output "
        f"director(ies):\n"
        f"  - {directory_lines}\n\n"
        f"{NOTHING_TO_VISUALIZE_HINT}"
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
        When a selected group directory is not usable, or when no selected output
        directory holds PSTH results, Tonic Analysis results, or binned metrics.
    """
    inputParameters = inputParameters

    _validate_results_exist(inputParameters)
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

    # One label per dashboard, built over every directory being opened, so each names
    # its run by the path that distinguishes it from the others on screen.
    labels = relative_output_labels(_output_directories(inputParameters))

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
            run_folders.append(select_run_folders(filepath, inputParameters=inputParameters))
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
            filepath = combined_output_groups[i][0]
            createPlots(
                filepath,
                event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
                inputParameters,
                label=labels[str(filepath)],
            )
    else:
        for i in range(len(session_folders)):
            filepath = session_folders[i]
            run_folders = select_run_folders(filepath, inputParameters=inputParameters)
            for j in range(len(run_folders)):
                filepath = run_folders[j]
                store_array = read_stores_list(run_folder=filepath)

                createPlots(
                    filepath,
                    event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
                    inputParameters,
                    label=labels[str(filepath)],
                )

    # Groups are ordinary output directories to the visualizer: one dashboard each,
    # opened alongside any selected session runs rather than instead of them.
    for group_folder in group_folders:
        store_array = read_stores_list(run_folder=group_folder)
        createPlots(
            group_folder,
            event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters),
            inputParameters,
            label=labels[str(group_folder)],
        )
