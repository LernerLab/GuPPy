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
import os
import shutil

import numpy as np

from .save_parameters import build_analysis_parameters, write_analysis_parameters
from ..analysis.io_utils import is_channel_label, is_continuous_label
from ..analysis.psth_average import average_psth_for_group
from ..analysis.transients_average import average_transients_for_group
from ..utils import progress
from ..utils.progress import step_error_handler
from ..utils.utils import (
    GROUP_MEMBERS_FILENAME,
    event_labels_for_analysis,
    group_folder_for_group,
    validate_group_name,
    write_group_members,
)
from ..utils.validation import (
    validate_group_destination,
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
        member_stores_list = np.genfromtxt(
            os.path.join(run_folder, "storesList.csv"), dtype="str", delimiter=","
        ).reshape(2, -1)
        fiber_stores = tuple(sorted(name for name in set(member_stores_list[1, :]) if is_channel_label(name)))
        per_member_fibers[run_folder] = fiber_stores

    unique_fiber_sets = set(per_member_fibers.values())
    if len(unique_fiber_sets) <= 1:
        return

    member_lines = "\n".join(
        f"  - {os.path.basename(os.path.dirname(run_folder))}: "
        f"{', '.join(stores) if stores else '(no control/signal store_ids)'}"
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
                np.genfromtxt(os.path.join(run_folder, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
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


def _create_group_folder(*, group_folder: str) -> None:
    """Create the group output directory, rebuilding it from scratch if it already exists.

    A group is fully recomputed from its members each run, so a pre-existing group
    directory is cleared first: otherwise dropping a member would leave that member's
    files behind. A directory that carries no manifest was not written by this step and
    is never deleted.

    Parameters
    ----------
    group_folder : str
        Path of the group output directory.

    Raises
    ------
    ValueError
        If the path exists but holds no ``group_members.json``.
    """
    if os.path.isdir(group_folder):
        if not os.path.exists(os.path.join(group_folder, GROUP_MEMBERS_FILENAME)):
            message = (
                f"'{group_folder}' already exists but was not created by the Group Analysis step "
                f"(it holds no {GROUP_MEMBERS_FILENAME}). Choose a different group name or destination "
                "directory."
            )
            logger.error(message)
            raise ValueError(message)
        shutil.rmtree(group_folder)
    os.makedirs(group_folder)


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


def orchestrate_group_analysis(inputParameters: dict[str, object]) -> None:
    """Average the selected member runs into a named group output directory.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``group_member_run_folders``,
        ``group_destination_directories`` and ``group_name``.

    Raises
    ------
    ValueError
        When the group name, destination or member selection is invalid, when the
        members disagree on their fiber recording sites, or when the target directory
        exists but was not created by this step.
    """
    member_run_folders = list(inputParameters["group_member_run_folders"])
    group_name = inputParameters["group_name"]

    validate_group_name(group_name)
    destination_directory = validate_group_destination(
        destination_directories=inputParameters["group_destination_directories"]
    )
    validate_group_member_run_folders(member_run_folders=member_run_folders)
    _validate_fiber_recording_sites_consistent_for_group(member_run_folders=member_run_folders)

    group_folder = group_folder_for_group(destination_directory=destination_directory, group_name=group_name)
    logger.info(f"Averaging {len(member_run_folders)} member run(s) into '{group_folder}'...")

    _create_group_folder(group_folder=group_folder)
    write_group_members(group_folder=group_folder, member_run_folders=member_run_folders)
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

    np.savetxt(
        os.path.join(group_folder, "storesList.csv"),
        _filter_stores_list_to_averaged_events(store_array=store_array, averaged_events=averaged_events),
        delimiter=",",
        fmt="%s",
    )
    logger.info(f"Group '{group_name}' averaged {len(averaged_events)} event(s) from {len(member_run_folders)} run(s).")


@step_error_handler
def run_group_analysis_step(input_parameters: dict[str, object]) -> None:
    """Run the Group Analysis step with progress reporting and failure handling attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    member_run_folders = list(input_parameters["group_member_run_folders"])
    validate_group_member_run_folders(member_run_folders=member_run_folders)
    store_array = _merge_group_stores_list(member_run_folders=member_run_folders)
    event_labels = _group_event_labels(store_array=store_array, inputParameters=input_parameters)
    # One unit per event store, plus the single unit the transient average reports.
    progress.start(len(event_labels) + 1)
    orchestrate_group_analysis(input_parameters)
