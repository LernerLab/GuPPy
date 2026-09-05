"""Group Analysis step: average an event's results across a named set of output runs.

A group is a named output directory, ``<destination>/<group_name>_group``, holding the
same filename patterns as a run folder with one column per member run in place of one
column per trial. It is produced from run folders that already carry Step-4 results — the
step averages what its members have and computes nothing from raw traces.

The group directory is run-folder-shaped for the consumers that read PSTH outputs
(Step 5), but not for the steps that need raw traces; the ``_group`` marker contains no
``_output_``, so ``discover_run_folders`` can never return one.
"""

import logging
import shutil
from pathlib import Path

import numpy as np

from .psth_significance import execute_compute_psth_significance
from .save_parameters import build_analysis_parameters, write_analysis_parameters
from ..analysis.io_utils import is_channel_label, is_continuous_label
from ..analysis.psth_average import average_psth_for_group
from ..analysis.transients_average import average_transients_for_group
from ..utils import progress
from ..utils.progress import step_error_handler
from ..utils.stores_list import read_stores_list, write_stores_list
from ..utils.utils import (
    GROUP_MEMBERS_FILENAME,
    event_labels_for_analysis,
    parse_group_name,
    read_group_members,
)
from ..utils.validation import (
    validate_group_folders_selected,
    validate_group_member_run_folders,
)

logger = logging.getLogger(__name__)


def _validate_fiber_recording_sites_consistent_for_group(*, member_run_folders: list[str]) -> None:
    """Check that every member run shares the same fiber (control/signal) store_ids.

    Group averaging buckets each member's data by its fiber recording-site basename
    (``z_score_<recording_site>`` / ``dff_<recording_site>``) and averages each
    behavioral event independently, skipping members that lack a given event. Members
    may therefore differ in their *event* store_ids — that is the intended
    cross-condition workflow (e.g. ``novelobject`` runs averaged alongside
    ``novelfemale1`` runs). What must agree is the set of *fiber* store_ids:
    averaging across different recording sites produces meaningless per-recording-site
    single-member "averages".

    Fiber store_ids follow the codebase-wide convention that their names contain
    ``control`` or ``signal``; every other store_id is treated as a behavioral
    event and ignored here.

    Parameters
    ----------
    member_run_folders : list of str
        Output (run) directories selected as the group's members.

    Raises
    ------
    ValueError
        When the members disagree on the set of fiber (control/signal) store_ids.
    """
    per_member_fibers = {}
    for run_folder in member_run_folders:
        member_stores_list = read_stores_list(run_folder=run_folder)
        fiber_stores = tuple(sorted(name for name in set(member_stores_list[1, :]) if is_channel_label(name)))
        per_member_fibers[run_folder] = fiber_stores

    unique_fiber_sets = set(per_member_fibers.values())
    if len(unique_fiber_sets) <= 1:
        return

    member_lines = "\n".join(
        f"  - {Path(run_folder).parent.name}: " f"{', '.join(stores) if stores else '(no control/signal store_ids)'}"
        for run_folder, stores in per_member_fibers.items()
    )
    raise ValueError(
        "Group averaging requires every member run to share the same fiber "
        "recording sites, but the selected members have mismatched control/signal "
        "store_ids:\n"
        f"{member_lines}\n"
        "Event store_ids may differ across members, but the control/signal "
        "store_ids must match. Fix the store_id labels in step 1, or remove the "
        "mismatched run from the group's member selection."
    )


def _merge_group_stores_list(*, member_run_folders: list[str]) -> np.ndarray:
    """Return the union of every member run's storesList as a single store array.

    Parameters
    ----------
    member_run_folders : list of str
        Output (run) directories of every member in the group.

    Returns
    -------
    np.ndarray
        2-D array with rows [store_id, store_label], deduplicated column-wise.
    """
    store_array = np.asarray([[], []])
    for run_folder in member_run_folders:
        store_array = np.concatenate(
            (
                store_array,
                read_stores_list(run_folder=run_folder),
            ),
            axis=1,
        )
    return np.unique(store_array, axis=1)


def _group_event_labels(*, store_array: np.ndarray, inputParameters: dict[str, object]) -> list[str]:
    """Return the event labels group averaging computes a PSTH average for.

    Parameters
    ----------
    store_array : np.ndarray
        Merged store array of every member run in the group.
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    list of str
        Store labels with the continuously sampled streams dropped.
    """
    return [
        label
        for label in event_labels_for_analysis(store_array=store_array, inputParameters=inputParameters)
        if not is_continuous_label(label)
    ]


def _clear_group_results(*, group_folder: str) -> None:
    """Delete a group's previously averaged results, keeping its definition.

    A group is fully recomputed from its members each run, so stale results are removed
    first: otherwise dropping a member would leave that member's columns behind in files
    the new run does not rewrite. ``group_members.json`` is preserved — it is the
    definition the averaging is driven from, not a result.

    Parameters
    ----------
    group_folder : str
        Group output directory to clear.
    """
    for path in Path(group_folder).iterdir():
        if path.name == GROUP_MEMBERS_FILENAME:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _filter_stores_list_to_averaged_events(*, store_array: np.ndarray, averaged_events: list[str]) -> np.ndarray:
    """Return the store array reduced to the group's continuous streams and averaged events.

    The group's ``storesList.csv`` is what Step 5 enumerates its plots from, so it must
    list only the events the group actually holds a PSTH for.

    Parameters
    ----------
    store_array : np.ndarray
        Merged store array of every member run in the group.
    averaged_events : list of str
        Event labels a PSTH average was written for.

    Returns
    -------
    np.ndarray
        2-D array with rows [store_id, store_label].
    """
    keep = [
        index
        for index in range(store_array.shape[1])
        if is_continuous_label(store_array[1, index]) or store_array[1, index] in averaged_events
    ]
    return store_array[:, keep]


def average_one_group(*, group_folder: str, inputParameters: dict[str, object]) -> None:
    """Average a single group's member runs into its directory.

    Parameters
    ----------
    group_folder : str
        Group output directory holding a ``group_members.json`` manifest.
    inputParameters : dict
        Full pipeline input parameters.

    Raises
    ------
    ValueError
        When the group holds no manifest, its members are unusable, or the members
        disagree on their fiber recording sites.
    """
    member_run_folders = read_group_members(group_folder=group_folder)
    validate_group_member_run_folders(member_run_folders=member_run_folders)
    _validate_fiber_recording_sites_consistent_for_group(member_run_folders=member_run_folders)

    group_name = parse_group_name(group_folder)
    logger.info("Averaging %s member run(s) into '%s'...", len(member_run_folders), group_folder)

    _clear_group_results(group_folder=group_folder)
    write_analysis_parameters(
        destination=group_folder,
        analysis_parameters=build_analysis_parameters(inputParameters=inputParameters),
    )

    average_transients_for_group(
        member_run_folders=member_run_folders, group_folder=group_folder, inputParameters=inputParameters
    )
    progress.advance()

    store_array = _merge_group_stores_list(member_run_folders=member_run_folders)
    averaged_events = []
    for event in _group_event_labels(store_array=store_array, inputParameters=inputParameters):
        if average_psth_for_group(
            member_run_folders=member_run_folders,
            event=event,
            group_folder=group_folder,
            inputParameters=inputParameters,
        ):
            averaged_events.append(event)
        progress.advance()

    write_stores_list(
        run_folder=group_folder,
        store_array=_filter_stores_list_to_averaged_events(store_array=store_array, averaged_events=averaged_events),
    )
    logger.info(
        "Group '%s' averaged %s event(s) from %s run(s).", group_name, len(averaged_events), len(member_run_folders)
    )

    # After the filtered storesList.csv is written: comparison planning reads it to learn
    # which events the group actually holds averaged results for.
    execute_compute_psth_significance(group_folder, inputParameters)
    if inputParameters["computePsthSignificance"]:
        progress.advance()


def orchestrate_group_analysis(inputParameters: dict[str, object]) -> None:
    """Average every selected group's member runs into its own directory.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``selected_group_folders``.

    Raises
    ------
    ValueError
        When no groups are selected, or when a group's manifest or members are unusable.
    """
    group_folders = list(inputParameters["selected_group_folders"])
    validate_group_folders_selected(group_folders=group_folders)
    for group_folder in group_folders:
        average_one_group(group_folder=group_folder, inputParameters=inputParameters)


@step_error_handler
def run_group_analysis_step(input_parameters: dict[str, object]) -> None:
    """Run the Group Analysis step with progress reporting and failure handling attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    group_folders = list(input_parameters["selected_group_folders"])
    validate_group_folders_selected(group_folders=group_folders)

    # One unit per event store per group, plus one per group for its transient average.
    total = 0
    for group_folder in group_folders:
        member_run_folders = read_group_members(group_folder=group_folder)
        validate_group_member_run_folders(member_run_folders=member_run_folders)
        store_array = _merge_group_stores_list(member_run_folders=member_run_folders)
        total += len(_group_event_labels(store_array=store_array, inputParameters=input_parameters)) + 1
        if input_parameters["computePsthSignificance"]:
            total += 1
    progress.start(total)

    orchestrate_group_analysis(input_parameters)
