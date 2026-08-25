"""Shared input-validation helpers for the GuPPy pipeline.

This module is the canonical home for reusable validation logic that several
layers of the pipeline (frontend, orchestration, analysis) need. Validation
that is genuinely one-off — i.e. coupled to a specific extractor or step —
should stay where it is used; only patterns that already repeat (or that this
module's introduction makes repeatable without straining the abstraction) live
here.

Conventions
-----------
- **Exception type**: validation helpers raise ``ValueError``. The Panel UI in
  ``orchestration/home.py`` catches exceptions from input-parameter parsing and
  step orchestration and surfaces them as persistent notifications via
  ``pn.state.notifications.error(str(e), duration=0)``. Using ``ValueError``
  consistently — instead of the generic ``Exception`` left over from older code
  — lets callers distinguish input problems from genuine bugs.
- **Layer responsibilities**:

  * Frontend (``src/guppy/frontend/``): required-field and format checks that
    can be evaluated from the form alone (folder selected, DANDI URI present).
  * Orchestration (``src/guppy/orchestration/``): pre-execution prerequisite
    checks that depend on the cross-product of multiple parameters or on
    on-disk state (store_ids consistency, peak-window ordering, metric
    availability against step-4 outputs).
  * Analysis (``src/guppy/analysis/``): parameter-vs-data checks that need a
    loaded signal (baseline window inside signal timespan).

- **Error message style**: name the offending value, state the rule, and tell
  the user the valid range or fix. See PR #283 for the established template.
"""

import glob
import logging
import os
from typing import Sequence

import numpy as np

from .utils import _RUN_NAME_MARKER, GROUP_MEMBERS_FILENAME, is_group_folder

logger = logging.getLogger(__name__)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return not np.isnan(float(value))


def validate_window_bounds(
    *,
    start: float,
    end: float,
    ts_min: float,
    ts_max: float,
    start_name: str,
    end_name: str,
    range_label: str = "signal timespan",
) -> None:
    """Validate a ``[start, end]`` window against an outer ``[ts_min, ts_max]`` range.

    Parameters
    ----------
    start, end : numeric
        The candidate window bounds, in the same units as ``ts_min`` / ``ts_max``.
    ts_min, ts_max : numeric
        The valid outer range the window must lie inside.
    start_name, end_name : str
        Parameter names used in error messages (e.g. ``"baselineWindowStart"``).
    range_label : str, optional
        Short label for the outer range used in error messages
        (e.g. ``"signal timespan"`` or ``"PSTH window"``).

    Raises
    ------
    ValueError
        If either bound is non-numeric / NaN, if ``start >= end``, or if either
        bound falls outside ``[ts_min, ts_max]``.
    """
    for name, value in ((start_name, start), (end_name, end)):
        if not _is_finite_number(value):
            message = f"{name}={value!r} is not a valid number; provide a numeric value in seconds."
            logger.error(message)
            raise ValueError(message)

    if start >= end:
        message = f"{start_name}={start} must be strictly less than {end_name}={end}; " f"choose start < end."
        logger.error(message)
        raise ValueError(message)

    if start < ts_min or end > ts_max:
        offending = []
        if start < ts_min:
            offending.append(f"{start_name}={start} is before the signal start {ts_min:.4g}s")
        if end > ts_max:
            offending.append(f"{end_name}={end} exceeds signal duration {ts_max:.4g}s")
        message = (
            f"{'; '.join(offending)}; "
            f"{range_label} is [{ts_min:.4g}, {ts_max:.4g}]s — "
            f"choose values within this range."
        )
        logger.error(message)
        raise ValueError(message)


def validate_positive(*, value: float, name: str) -> None:
    """Validate that ``value`` is a positive number (strictly greater than 0).

    Parameters
    ----------
    value : numeric
        The candidate value.
    name : str
        Parameter name used in error messages (e.g. ``"moving_window"``).

    Raises
    ------
    ValueError
        If ``value`` is non-numeric / NaN, or is less than or equal to 0.
    """
    if not _is_finite_number(value):
        message = f"{name}={value!r} is not a valid number; provide a positive numeric value."
        logger.error(message)
        raise ValueError(message)
    if value <= 0:
        message = f"{name}={value} must be greater than 0; choose a positive value."
        logger.error(message)
        raise ValueError(message)


def validate_non_negative(*, value: float, name: str) -> None:
    """Validate that ``value`` is a non-negative number (0 or greater).

    Parameters
    ----------
    value : numeric
        The candidate value.
    name : str
        Parameter name used in error messages (e.g. ``"filter_window"``).

    Raises
    ------
    ValueError
        If ``value`` is non-numeric / NaN, or is less than 0.
    """
    if not _is_finite_number(value):
        message = f"{name}={value!r} is not a valid number; provide a non-negative numeric value."
        logger.error(message)
        raise ValueError(message)
    if value < 0:
        message = f"{name}={value} must be 0 or greater; choose a non-negative value."
        logger.error(message)
        raise ValueError(message)


def validate_peak_windows(*, peak_starts: Sequence[float], peak_ends: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired peak-window arrays and return them with NaN padding stripped.

    The GUI exposes ten peak-window slots, each padded with ``NaN`` when unused,
    so valid input has equal numbers of non-NaN start and end values.

    Parameters
    ----------
    peak_starts, peak_ends : sequence of float
        Per-pair start and end times (in seconds, relative to the PSTH event).

    Returns
    -------
    starts, ends : np.ndarray
        Cleaned arrays with NaN slots removed.

    Raises
    ------
    ValueError
        If the number of non-NaN starts and ends differ, or if any
        ``end <= start`` after stripping.
    """
    starts = np.asarray(peak_starts, dtype=float)
    ends = np.asarray(peak_ends, dtype=float)
    starts = starts[~np.isnan(starts)]
    ends = ends[~np.isnan(ends)]

    if starts.shape[0] != ends.shape[0]:
        message = (
            f"Number of Peak Start Time and Peak End Time are unequal "
            f"(start: {starts.shape[0]}, end: {ends.shape[0]}). "
            f"Each peak window needs both a start and an end value."
        )
        logger.error(message)
        raise ValueError(message)

    if starts.shape[0] > 0 and np.less_equal(ends, starts).any():
        offending = [f"(start={starts[i]}, end={ends[i]})" for i in range(starts.shape[0]) if ends[i] <= starts[i]]
        message = (
            "Peak End Time is less than or equal to Peak Start Time for "
            f"{len(offending)} window(s): {', '.join(offending)}. "
            "Please check the Peak parameters window."
        )
        logger.error(message)
        raise ValueError(message)

    return starts, ends


def validate_required_folder_selection(*, file_selectors: Sequence) -> None:
    """Validate that at least one folder is selected across the given file selectors.

    Parameters
    ----------
    file_selectors : sequence
        Iterable of Panel file-selector widgets (or any object with a ``.value``
        attribute that returns a list of selected paths).

    Raises
    ------
    ValueError
        If every selector is empty.
    """
    if all(len(selector.value) == 0 for selector in file_selectors):
        message = (
            "No folder is selected for analysis. Pick at least one session folder in the "
            "file selector(s) before running this step."
        )
        logger.error(message)
        raise ValueError(message)


def validate_same_parent_directory(*, paths: Sequence[str]) -> np.ndarray:
    """Validate that every path shares the same parent directory.

    Parameters
    ----------
    paths : sequence of str
        Absolute paths to selected session folders.

    Returns
    -------
    np.ndarray
        A length-1 array containing the shared parent directory.

    Raises
    ------
    ValueError
        If the paths span more than one parent directory.
    """
    parents = np.unique(np.asarray([os.path.dirname(path) for path in paths]))
    if len(parents) > 1:
        path_to_parent = "\n".join(f"  - {path} (parent: {os.path.dirname(path)})" for path in paths)
        message = (
            "All the folders selected should be at the same location, but the selected folders "
            f"span {len(parents)} parent directories:\n{path_to_parent}"
        )
        logger.error(message)
        raise ValueError(message)
    return parents


def validate_artifact_coords_present(*, run_folders: Sequence[str]) -> None:
    """Validate that artifact windows have been selected for every run folder.

    Parameters
    ----------
    run_folders : sequence of str
        Session output (run) directories the Remove Artifacts step will process.

    Raises
    ------
    ValueError
        If any run folder has no ``coordsForPreProcessing_<recording_site>.npy`` file.
    """
    for run_folder in run_folders:
        if not glob.glob(os.path.join(run_folder, "coordsForPreProcessing_*.npy")):
            message = (
                f"No artifact windows have been selected for '{run_folder}'. Run Select Artifact Windows "
                "and save at least one window before running Remove Artifacts."
            )
            logger.error(message)
            raise ValueError(message)


def validate_preprocessing_outputs_present(
    *, run_folders: Sequence[str], action: str = "selecting artifact windows"
) -> None:
    """Validate that every run folder holds the preprocessing outputs the step-3 result pages read.

    Parameters
    ----------
    run_folders : sequence of str
        Session output (run) directories to check.
    action : str
        Phrase naming what the caller is about to do, used to close the error message.

    Raises
    ------
    ValueError
        If any run folder is missing its ``cntrl_sig_fit_<recording_site>.hdf5`` files.
    """
    for run_folder in run_folders:
        if not glob.glob(os.path.join(run_folder, "cntrl_sig_fit_*.hdf5")):
            message = f"No preprocessing outputs found in '{run_folder}'. Run Step 3 (Preprocess) before {action}."
            logger.error(message)
            raise ValueError(message)


def validate_group_member_run_folders(*, member_run_folders: Sequence[str]) -> None:
    """Validate the run folders selected as a group's members.

    Parameters
    ----------
    member_run_folders : sequence of str
        Output (run) directories selected to be averaged into a group.

    Raises
    ------
    ValueError
        If the selection is empty, or if any path is missing, is not an output
        directory, or holds no ``storesList.csv``.
    """
    if not member_run_folders:
        message = (
            "No member runs selected for group averaging. Pick at least one "
            "'<session>_output_<run>' directory in the Group Analysis card before running the step."
        )
        logger.error(message)
        raise ValueError(message)

    not_output_directories = [path for path in member_run_folders if _RUN_NAME_MARKER not in os.path.basename(path)]
    if not_output_directories:
        message = (
            f"Group members must be output directories, but these are not: {not_output_directories!r}. "
            "Select the '<session>_output_<run>' directories inside each session, not the session folders "
            "themselves."
        )
        logger.error(message)
        raise ValueError(message)

    missing = [path for path in member_run_folders if not os.path.isdir(path)]
    if missing:
        message = f"Group member run folders do not exist: {missing!r}. Re-select the group's members."
        logger.error(message)
        raise ValueError(message)

    missing_stores = [path for path in member_run_folders if not os.path.exists(os.path.join(path, "storesList.csv"))]
    if missing_stores:
        message = (
            f"Group member run folders are missing storesList.csv: {missing_stores!r}. "
            "Re-run Step 1 (Label Stores) for these runs before adding them to a group."
        )
        logger.error(message)
        raise ValueError(message)


def validate_group_definitions(*, group_folders: Sequence[str]) -> None:
    """Validate a selection of group output directories as group *definitions*.

    Checks only what the Label Groups step writes; a group holds no averaged results
    until the Group Analysis step runs against it.

    Parameters
    ----------
    group_folders : sequence of str
        Group output directories to check.

    Raises
    ------
    ValueError
        If any path is missing, is not a ``<group_name>_group`` directory, or holds no
        ``group_members.json``.
    """
    not_group_directories = [path for path in group_folders if not is_group_folder(path)]
    if not_group_directories:
        message = (
            f"These are not group output directories: {not_group_directories!r}. "
            "A group directory is named '<group_name>_group' and is created by the Label Groups step."
        )
        logger.error(message)
        raise ValueError(message)

    missing = [path for path in group_folders if not os.path.isdir(path)]
    if missing:
        message = f"Group output directories do not exist: {missing!r}. Re-create them with the Label Groups step."
        logger.error(message)
        raise ValueError(message)

    undefined = [path for path in group_folders if not os.path.exists(os.path.join(path, GROUP_MEMBERS_FILENAME))]
    if undefined:
        message = (
            f"Group output directories hold no {GROUP_MEMBERS_FILENAME}: {undefined!r}. "
            "Define their members with the Label Groups step."
        )
        logger.error(message)
        raise ValueError(message)


def validate_group_folders_selected(*, group_folders: Sequence[str]) -> None:
    """Validate that at least one usable group directory is selected.

    Parameters
    ----------
    group_folders : sequence of str
        Group output directories selected on the homepage.

    Raises
    ------
    ValueError
        If nothing is selected, or if any selection is not a usable group directory.
    """
    if not group_folders:
        message = (
            "No groups selected. Pick at least one '<name>_group' directory in the Group Output "
            "Folder Selection panel, or define a group first with the Label Groups step."
        )
        logger.error(message)
        raise ValueError(message)
    validate_group_definitions(group_folders=group_folders)


def validate_data_not_combined(*, combine_data: bool) -> None:
    """Validate that the selected sessions were not analyzed with combining enabled.

    Parameters
    ----------
    combine_data : bool
        The pipeline's ``combine_data`` setting.

    Raises
    ------
    ValueError
        If ``combine_data`` is True.
    """
    if combine_data:
        message = (
            "NWB export does not support combine_data=True. Combining collapses a run group into a "
            "single output directory, while the export writes one NWB file per selected session from "
            "that session's own raw folder, so there is no session the combined outputs belong to. "
            "Re-run the pipeline with 'Combine Data?' set to False to export."
        )
        logger.error(message)
        raise ValueError(message)
