import glob
import json
import logging
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAISE_ISSUE_URL = "https://github.com/LernerLab/GuPPy/issues/new"

_RUN_NAME_MARKER = "_output_"
_FORBIDDEN_RUN_NAME_CHARACTERS = ("/", "\\", ":", "\0")

# Group output directories are named "<group_name>_group". The marker contains no
# "_output_", so discover_run_folders can never return a group folder.
_GROUP_NAME_MARKER = "_group"

# Records which run folders a group averaged, so the group can be reopened and rebuilt
# and so column N of a group PSTH can be traced back to member N.
GROUP_MEMBERS_FILENAME = "group_members.json"

# NPM decomposition parameters chosen interactively in Step 1 are not part of the
# saved analysis parameters, so they are persisted next to storesList.csv for Step 2.
NPM_PARAMS_FILENAME = ".npm_params.json"
NPM_PARAM_KEYS = ("npm_split_events", "npm_time_unit", "npm_timestamp_column_name")

# Event-label prefix for the transient trains that stand in for external TTLs when
# useTransientsAsEvents is on. Prepended to a preprocessed basename it yields the
# event file name, e.g. "transients_" + "z_score_DMS" -> transients_z_score_DMS.hdf5.
TRANSIENT_EVENT_PREFIX = "transients_"


def write_npm_params(*, run_folder: str, npm_params: dict[str, object]) -> None:
    """Persist the NPM decomposition parameters for one output directory.

    The NPM choices made during Step 1 (event splitting, the session's timestamp
    unit and timestamp column) determine how :class:`NpmRecordingExtractor`
    demultiplexes the raw files in memory. They are written next to
    ``storesList.csv`` so Step 2 can reproduce the identical decomposition.

    Parameters
    ----------
    run_folder : str
        Output directory where ``storesList.csv`` is written.
    npm_params : dict
        The NPM parameters (keys in :data:`NPM_PARAM_KEYS`) to persist.
    """
    with open(os.path.join(run_folder, NPM_PARAMS_FILENAME), "w") as file:
        json.dump(npm_params, file, indent=4)


def load_npm_params(run_folder: str) -> dict[str, object]:
    """Load persisted NPM decomposition parameters from an output directory.

    Parameters
    ----------
    run_folder : str
        Output directory possibly containing the NPM parameters file.

    Returns
    -------
    dict
        The persisted NPM parameters, or an empty dict if none were written.

    Raises
    ------
    ValueError
        If the file predates the session-wide timestamp unit and so records no
        unit that can be trusted to match the one its data was read with.
    """
    npm_params_path = os.path.join(run_folder, NPM_PARAMS_FILENAME)
    if not os.path.exists(npm_params_path):
        return {}
    with open(npm_params_path) as file:
        npm_params = json.load(file)

    if "npm_time_unit" not in npm_params:
        message = (
            f"'{npm_params_path}' records no 'npm_time_unit' and was written by a GuPPy version whose "
            "recorded timestamp unit did not always match the one applied. Re-run Step 1 (Label Stores) "
            f"for '{run_folder}' to record the unit this session's timestamps are in."
        )
        logger.error(message)
        raise ValueError(message)

    return npm_params


def write_group_members(*, group_folder: str, member_run_folders: list[str]) -> None:
    """Persist the run folders a group was averaged from.

    Parameters
    ----------
    group_folder : str
        Group output directory receiving the manifest.
    member_run_folders : list of str
        Absolute paths of the member run folders, in averaging order.
    """
    with open(os.path.join(group_folder, GROUP_MEMBERS_FILENAME), "w") as file:
        json.dump({"member_run_folders": list(member_run_folders)}, file, indent=4)


def read_group_members(*, group_folder: str) -> list[str]:
    """Return the run folders recorded in a group's manifest.

    Parameters
    ----------
    group_folder : str
        Group output directory holding the manifest.

    Returns
    -------
    list of str
        Absolute paths of the member run folders, in averaging order.

    Raises
    ------
    ValueError
        If the group directory holds no manifest.
    """
    manifest_path = os.path.join(group_folder, GROUP_MEMBERS_FILENAME)
    if not os.path.exists(manifest_path):
        message = (
            f"{group_folder!r} holds no {GROUP_MEMBERS_FILENAME}, so it was not created by GuPPy's "
            "Group Analysis step. Re-create the group from the Group Analysis card."
        )
        logger.error(message)
        raise ValueError(message)
    with open(manifest_path) as file:
        return json.load(file)["member_run_folders"]


def takeOnlyDirs(paths: list[str]) -> list[str]:
    """Filter a list of paths to include only directories.

    Parameters
    ----------
    paths : list of str
        Mixed list of file and directory paths.

    Returns
    -------
    list of str
        Subset of ``paths`` containing only directories.
    """
    removePaths = []
    for path in paths:
        if os.path.isfile(path):
            removePaths.append(path)
    return list(set(paths) - set(removePaths))


def parse_run_name(run_folder: str) -> str:
    """Return the run-name suffix of an output directory.

    Splits the directory's basename on the last occurrence of ``_output_`` and
    returns everything after it.  Legacy ``mySession_output_1`` directories
    yield ``"1"``.

    Parameters
    ----------
    run_folder : str
        Path to an ``<session_basename>_output_<run_name>`` directory.

    Returns
    -------
    str
        The run-name suffix.

    Raises
    ------
    ValueError
        If the basename does not match the expected pattern.
    """
    # Strip both separators so trailing forward slashes are tolerated on Windows
    # (where os.sep is "\\" but paths can still use "/").
    basename = os.path.basename(run_folder.rstrip("/\\"))
    index = basename.rfind(_RUN_NAME_MARKER)
    if index < 0:
        raise ValueError(
            f"Cannot parse run name from {run_folder!r}: basename {basename!r} does not match "
            f"'<session_basename>_output_<run_name>' pattern."
        )
    return basename[index + len(_RUN_NAME_MARKER) :]


def discover_run_folders(session_path: str) -> list[str]:
    """Return all output directories within a session, sorted by run name.

    Parameters
    ----------
    session_path : str
        Path to a session folder.

    Returns
    -------
    list of str
        Absolute paths of every ``<basename>_output_*`` subdirectory, sorted
        deterministically: numeric run names first (sorted numerically), then
        non-numeric run names (sorted case-insensitively).
    """
    candidates = takeOnlyDirs(glob.glob(os.path.join(session_path, "*" + _RUN_NAME_MARKER + "*")))
    return sorted(candidates, key=_run_name_sort_key_for_path)


def run_folder_for_run(session_path: str, run_name: str) -> str:
    """Build the path of the output directory for a given run name.

    Does not check whether the directory exists.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    run_name : str
        Run-name suffix to append after ``_output_``.

    Returns
    -------
    str
        Path of the form ``<session_path>/<basename>_output_<run_name>``.
    """
    basename = os.path.basename(session_path.rstrip(os.sep))
    return os.path.join(session_path, basename + _RUN_NAME_MARKER + run_name)


def selected_session_runs(*, inputParameters: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten ``selected_runs`` into ``(session_path, run_name)`` pairs.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    list of (str, str)
        One pair per selected run, in the order the sessions and runs were selected.
    """
    selected_runs: dict[str, list[str]] = inputParameters["selected_runs"]
    return [(session_path, run_name) for session_path, run_names in selected_runs.items() for run_name in run_names]


def select_run_folders(session_path: str, selected_runs: list[str]) -> list[str]:
    """Filter a session's output directories to those matching ``selected_runs``.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    selected_runs : list of str
        Run-name suffixes to keep. Must be a non-empty list.

    Returns
    -------
    list of str
        Absolute paths of the selected output directories.

    Raises
    ------
    ValueError
        When ``selected_runs`` is empty/``None``, when a requested run name has
        no matching directory, or when a selected directory is missing
        ``storesList.csv``. The error message lists the available run names so
        the user can correct their input.
    """
    if not selected_runs:
        raise ValueError(
            f"select_run_folders requires an explicit non-empty list of run names for session "
            f"{session_path!r}; got {selected_runs!r}. Pick at least one existing _output_<run> "
            "directory in the Output Folder Selection panel."
        )
    available = discover_run_folders(session_path)
    available_by_name = {parse_run_name(directory): directory for directory in available}
    missing = [run for run in selected_runs if run not in available_by_name]
    if missing:
        raise ValueError(
            f"Output directory not found in {session_path!r} for run name(s) {missing!r}. "
            f"Available runs: {sorted(available_by_name.keys())!r}. "
            "Either run step 1 with the requested run name first, or update the selected_runs filter."
        )

    selected = [available_by_name[run] for run in selected_runs]
    missing_stores = [
        run_folder for run_folder in selected if not os.path.exists(os.path.join(run_folder, "storesList.csv"))
    ]
    if missing_stores:
        raise ValueError(
            f"Selected output directories are missing storesList.csv: {missing_stores!r}. "
            "Re-run step 1 (Label Stores) for these run names before continuing."
        )
    return sorted(selected, key=_run_name_sort_key_for_path)


def validate_run_name(run_name: str) -> None:
    """Validate that ``run_name`` is a legal run-name suffix.

    Rejects empty strings, whitespace-only strings, path separators, ``..``,
    null bytes, and any string that contains the literal substring
    ``_output_`` (which would break round-tripping through
    :func:`parse_run_name`).

    Parameters
    ----------
    run_name : str
        Candidate run-name suffix.

    Raises
    ------
    ValueError
        If ``run_name`` is invalid.
    """
    if not isinstance(run_name, str):
        raise ValueError(f"run_name must be a string; got {type(run_name).__name__}.")
    if not run_name:
        raise ValueError("run_name must be a non-empty string.")
    if run_name.strip() != run_name or not run_name.strip():
        raise ValueError(f"run_name {run_name!r} must not contain leading/trailing whitespace or be all whitespace.")
    for character in _FORBIDDEN_RUN_NAME_CHARACTERS:
        if character in run_name:
            raise ValueError(
                f"run_name {run_name!r} contains forbidden character {character!r}. "
                f"Path separators and null bytes are not allowed."
            )
    if ".." in run_name:
        raise ValueError(f"run_name {run_name!r} must not contain '..' (path traversal).")
    if _RUN_NAME_MARKER in run_name:
        raise ValueError(
            f"run_name {run_name!r} must not contain the substring {_RUN_NAME_MARKER!r}; "
            "this would break parsing of the output directory name."
        )


def parse_group_name(group_folder: str) -> str:
    """Return the group name of a group output directory.

    Parameters
    ----------
    group_folder : str
        Path to a ``<group_name>_group`` directory.

    Returns
    -------
    str
        The group name.

    Raises
    ------
    ValueError
        If the basename does not match the expected pattern.
    """
    basename = os.path.basename(group_folder.rstrip("/\\"))
    if not basename.endswith(_GROUP_NAME_MARKER) or basename == _GROUP_NAME_MARKER:
        raise ValueError(
            f"Cannot parse group name from {group_folder!r}: basename {basename!r} does not match "
            f"'<group_name>_group' pattern."
        )
    return basename[: -len(_GROUP_NAME_MARKER)]


def common_parent_directory(*, paths: Sequence[str]) -> str:
    """Return the deepest directory that contains every one of ``paths``.

    Parameters
    ----------
    paths : sequence of str
        Absolute paths to selected session folders.

    Returns
    -------
    str
        The parent directory shared by all ``paths`` when they sit side by side,
        or their nearest common ancestor when they do not.
    """
    parent_directories = {os.path.dirname(path) for path in paths}
    return os.path.commonpath(sorted(parent_directories))


def is_group_folder(path: str) -> bool:
    """Report whether a path names a group output directory.

    Parameters
    ----------
    path : str
        Path to test.

    Returns
    -------
    bool
        ``True`` when the basename ends with ``_group`` and is not itself a run
        folder (a run named ``group`` would otherwise match both).
    """
    basename = os.path.basename(path.rstrip("/\\"))
    if _RUN_NAME_MARKER in basename:
        return False
    return basename.endswith(_GROUP_NAME_MARKER) and basename != _GROUP_NAME_MARKER


def discover_group_folders(destination_directory: str) -> list[str]:
    """Return all group output directories within a destination directory.

    Parameters
    ----------
    destination_directory : str
        Directory that group output directories are written into.

    Returns
    -------
    list of str
        Absolute paths of every ``<group_name>_group`` subdirectory, sorted
        case-insensitively by group name.
    """
    candidates = takeOnlyDirs(glob.glob(os.path.join(destination_directory, "*" + _GROUP_NAME_MARKER)))
    group_folders = [path for path in candidates if is_group_folder(path)]
    return sorted(group_folders, key=lambda path: parse_group_name(path).casefold())


def group_folder_for_group(*, destination_directory: str, group_name: str) -> str:
    """Build the path of the output directory for a given group name.

    Does not check whether the directory exists.

    Parameters
    ----------
    destination_directory : str
        Directory the group output directory is written into.
    group_name : str
        Name of the group.

    Returns
    -------
    str
        Path of the group output directory.
    """
    return os.path.join(destination_directory, group_name + _GROUP_NAME_MARKER)


def validate_group_name(group_name: str) -> None:
    """Validate that ``group_name`` is a legal group name.

    Rejects empty strings, whitespace-only strings, path separators, ``..``,
    null bytes, and any string containing ``_output_`` or ``_group`` (either of
    which would make the resulting directory indistinguishable from a run
    folder or from a session folder that merely ends in ``_group``).

    Parameters
    ----------
    group_name : str
        Candidate group name.

    Raises
    ------
    ValueError
        If ``group_name`` is invalid.
    """
    if not isinstance(group_name, str):
        raise ValueError(f"group_name must be a string; got {type(group_name).__name__}.")
    if not group_name:
        raise ValueError("group_name must be a non-empty string. Type a name in the Group Analysis card.")
    if group_name.strip() != group_name or not group_name.strip():
        raise ValueError(
            f"group_name {group_name!r} must not contain leading/trailing whitespace or be all whitespace."
        )
    for character in _FORBIDDEN_RUN_NAME_CHARACTERS:
        if character in group_name:
            raise ValueError(
                f"group_name {group_name!r} contains forbidden character {character!r}. "
                f"Path separators and null bytes are not allowed."
            )
    if ".." in group_name:
        raise ValueError(f"group_name {group_name!r} must not contain '..' (path traversal).")
    for marker in (_RUN_NAME_MARKER, _GROUP_NAME_MARKER):
        if marker in group_name:
            raise ValueError(
                f"group_name {group_name!r} must not contain the substring {marker!r}; "
                "this would break parsing of the group directory name."
            )


def _run_name_sort_key(run_name: str) -> tuple[int, int, str]:
    """Sort key that orders numeric run names ahead of alphanumeric ones."""
    try:
        return (0, int(run_name), "")
    except ValueError:
        return (1, 0, run_name.casefold())


def _run_name_sort_key_for_path(path: str) -> tuple[int, int, str]:
    """Sort key that orders output-directory paths by their run-name suffix."""
    try:
        run_name = parse_run_name(path)
    except ValueError:
        return (2, 0, os.path.basename(path).casefold())
    return _run_name_sort_key(run_name)


def get_all_stores_for_combining_data(run_folders: list[str]) -> list[list[str]]:
    """Group output directories by run-name suffix for cross-session combining.

    Parameters
    ----------
    run_folders : list of str
        Paths to ``<basename>_output_<run_name>`` directories across all sessions.

    Returns
    -------
    list of list of str
        One inner list per distinct run name.  Inner lists are sorted
        case-insensitively by path; outer ordering puts numeric run names
        first (numerically) and then alphanumeric run names (case-insensitive).
    """
    run_name_to_paths = {}
    for path in run_folders:
        try:
            run_name = parse_run_name(path)
        except ValueError:
            continue
        run_name_to_paths.setdefault(run_name, []).append(path)

    ordered_run_names = sorted(run_name_to_paths.keys(), key=_run_name_sort_key)
    return [sorted(run_name_to_paths[name], key=str.casefold) for name in ordered_run_names]


def transient_event_labels(*, inputParameters: dict[str, object]) -> list[str]:
    """Return the event labels contributed by the detected transients.

    The labels are derived from the parameters rather than discovered on disk, so
    transient event files left behind by an earlier run cannot re-enter the analysis
    once the toggle is switched off.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    list of str
        ``[]`` when ``useTransientsAsEvents`` is off, otherwise one label per metric
        the transient detector runs on, e.g. ``["transients_z_score"]``.
    """
    if inputParameters["useTransientsAsEvents"] == False:
        return []

    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    if selectForTransientsComputation == "z_score":
        metrics = ["z_score"]
    elif selectForTransientsComputation == "dff":
        metrics = ["dff"]
    else:
        metrics = ["z_score", "dff"]

    return [TRANSIENT_EVENT_PREFIX + metric for metric in metrics]


def event_labels_for_analysis(*, store_array: np.ndarray, inputParameters: dict[str, object]) -> list[str]:
    """Return every store label the PSTH and visualization steps should fan out over.

    Parameters
    ----------
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].
    inputParameters : dict
        Full pipeline input parameters.

    Returns
    -------
    list of str
        The storesList labels, followed by the transient event labels when
        ``useTransientsAsEvents`` is on. Labels are deduplicated in first-seen order,
        since two stores may share a label across the merged storesList files.
    """
    labels = list(store_array[1, :]) + transient_event_labels(inputParameters=inputParameters)
    return list(dict.fromkeys(labels))


def read_Df(filepath: str, event: str, name: str) -> pd.DataFrame:
    """Read a PSTH HDF5 file and return it as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Event name used in the filename.
    name : str
        z-score/dff basename; when non-empty the filename is
        ``<event>_<name>.h5``, otherwise ``<event>.h5``.

    Returns
    -------
    pandas.DataFrame
        PSTH data loaded from the HDF5 file.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        hdf5_path = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        hdf5_path = os.path.join(filepath, event + ".h5")
    df = pd.read_hdf(hdf5_path, key="df", mode="r")

    return df


def resolve_run_folders(session_folders: list, inputParameters: dict) -> list[str]:
    """Return the output (run) folders a compute job wrote for the given sessions.

    Mirrors the folder selection the step workers use: per-session run folders normally,
    or the first folder of each combine-group when ``combine_data`` is set.

    Parameters
    ----------
    session_folders : list
        Session directories to resolve.
    inputParameters : dict
        Pipeline configuration; must include ``'combine_data'``.

    Returns
    -------
    list of str
        The resolved run folders.
    """
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders: list[str] = []
    for session in session_folders:
        run_folders.append(select_run_folders(session, selected_runs.get(session)))
    run_folders = list(np.concatenate(run_folders).flatten())

    if inputParameters["combine_data"] == True:
        return [group[0] for group in get_all_stores_for_combining_data(run_folders)]
    return run_folders
