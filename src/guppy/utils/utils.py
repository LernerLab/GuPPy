import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAISE_ISSUE_URL = "https://github.com/LernerLab/GuPPy/issues/new"

_RUN_FOLDER_PREFIX = "output_"
_RUN_NAME_MARKER = "_" + _RUN_FOLDER_PREFIX
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
NPM_PARAM_KEYS = ("npm_split_events", "npm_time_unit", "npm_timestamp_column_name", "noChannels")

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
    with (Path(run_folder) / NPM_PARAMS_FILENAME).open("w") as file:
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
    npm_params_path = Path(run_folder) / NPM_PARAMS_FILENAME
    if not npm_params_path.exists():
        return {}
    with npm_params_path.open() as file:
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
    with (Path(group_folder) / GROUP_MEMBERS_FILENAME).open("w") as file:
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
    manifest_path = Path(group_folder) / GROUP_MEMBERS_FILENAME
    if not manifest_path.exists():
        message = (
            f"{group_folder!r} holds no {GROUP_MEMBERS_FILENAME}, so it was not created by GuPPy's "
            "Group Analysis step. Re-create the group from the Group Analysis card."
        )
        logger.error(message)
        raise ValueError(message)
    with manifest_path.open() as file:
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
    return [path for path in paths if not Path(path).is_file()]


def is_run_folder(path: str) -> bool:
    """Return whether a directory's name is that of an output directory.

    Recognises both the ``output_<run name>`` directories written under an output base
    directory and the ``<session basename>_output_<run name>`` directories of the
    inside-the-session layout.

    Parameters
    ----------
    path : str
        Path to check.

    Returns
    -------
    bool
        ``True`` when the basename carries the run-folder marker.
    """
    return _RUN_FOLDER_PREFIX in Path(str(path).rstrip("/\\")).name


def parse_run_name(run_folder: str) -> str:
    """Return the run name of an output directory.

    Reads everything after the last ``output_`` in the basename, which covers the
    ``output_<run name>`` directories written under an output base directory and the
    ``<session basename>_output_<run name>`` directories of the inside-the-session
    layout.

    Parameters
    ----------
    run_folder : str
        Path to an output directory.

    Returns
    -------
    str
        The run name.

    Raises
    ------
    ValueError
        If the basename does not match the expected pattern.
    """
    # Strip both separators so trailing forward slashes are tolerated on Windows
    # (where os.sep is "\\" but paths can still use "/").
    basename = Path(str(run_folder).rstrip("/\\")).name
    index = basename.rfind(_RUN_FOLDER_PREFIX)
    if index < 0:
        raise ValueError(
            f"Cannot parse run name from {run_folder!r}: basename {basename!r} does not match "
            f"'output_<run_name>' or '<session_basename>_output_<run_name>' pattern."
        )
    return basename[index + len(_RUN_FOLDER_PREFIX) :]


def parse_session_basename(run_folder: str) -> str:
    """Return the name of the session folder an output directory belongs to.

    An output directory always sits in a directory named for its session: the mirror
    of the session under the output base directory, or the session folder itself in
    the inside-the-session layout.

    Parameters
    ----------
    run_folder : str
        Path to an output directory.

    Returns
    -------
    str
        The session folder's basename.
    """
    return Path(_normalize(run_folder)).parent.name


def _normalize(path: str) -> str:
    """Return ``path`` as an absolute, symlink-resolved path.

    Session folders reach GuPPy from a file browser and data roots can be typed, so the
    two sides of a containment check are resolved before they are compared.
    """
    return str(Path(str(path).rstrip("/\\")).resolve())


def session_is_under_data_root(*, session_path: str, data_root: str | None) -> bool:
    """Return whether a session folder sits inside the data root.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    data_root : str or None
        Directory the session folders are selected inside.

    Returns
    -------
    bool
        ``True`` when the session sits strictly below ``data_root``.
    """
    if data_root is None:
        return False
    return Path(_normalize(data_root)) in Path(_normalize(session_path)).parents


def session_relative_path(*, session_path: str, data_root: str | None) -> str:
    """Return a session folder's path relative to the data root it was selected under.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    data_root : str or None
        Directory the session folders are selected inside.

    Returns
    -------
    str
        ``session_path`` relative to ``data_root``.

    Raises
    ------
    ValueError
        If ``data_root`` is ``None`` or ``session_path`` does not sit under it.
    """
    if data_root is None:
        raise ValueError(
            f"No data root given for session {str(session_path)!r}, so its path cannot be mirrored "
            f"into the output base directory. Pick a data root in the Input Folder Selection card."
        )
    session = Path(_normalize(session_path))
    root = Path(_normalize(data_root))
    if session == root or root not in session.parents:
        raise ValueError(
            f"Session folder {str(session)!r} is not inside the data root {str(root)!r}, so its "
            f"output directory has no place in the mirrored output tree."
        )
    return str(session.relative_to(root))


def run_folder_label(run_folder: str) -> str:
    """Return a label naming both the session and the run an output directory holds.

    Group results carry one column per member run, so the label has to stay distinct
    across sessions even though the directories themselves are named for the run alone.

    Parameters
    ----------
    run_folder : str
        Path to an output directory.

    Returns
    -------
    str
        ``<session basename>_output_<run name>``.
    """
    return parse_session_basename(run_folder) + _RUN_NAME_MARKER + parse_run_name(run_folder)


def run_directory_root(*, session_path: str, output_base_directory: str | None, data_root: str | None = None) -> str:
    """Return the directory a session's output directories are written into.

    With an output base directory the session's path relative to ``data_root`` is
    mirrored underneath it, so ``<data_root>/subject1/session1`` writes its runs into
    ``<output base>/subject1/session1``.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    output_base_directory : str or None
        Directory the mirrored output tree is written into, or ``None`` to write the
        output directories inside the session folder itself.
    data_root : str or None, optional
        Directory the session folders are selected inside.  Required whenever
        ``output_base_directory`` is given.

    Returns
    -------
    str
        The directory that holds this session's output directories.
    """
    if output_base_directory is None:
        return session_path
    relative = session_relative_path(session_path=session_path, data_root=data_root)
    return str(Path(output_base_directory) / relative)


def run_folder_basename(*, session_path: str, run_name: str, output_base_directory: str | None) -> str:
    """Return the basename an output directory carries for a given run name.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    run_name : str
        Run name the output directory is for.
    output_base_directory : str or None
        Directory the mirrored output tree is written into, or ``None`` for the
        inside-the-session layout, where the basename is prefixed with the session
        name to keep the session folder readable.

    Returns
    -------
    str
        The output directory's basename.
    """
    if output_base_directory is None:
        return Path(str(session_path).rstrip(os.sep)).name + _RUN_NAME_MARKER + run_name
    return _RUN_FOLDER_PREFIX + run_name


def discover_run_folders(
    session_path: str, *, output_base_directory: str | None = None, data_root: str | None = None
) -> list[str]:
    """Return all output directories belonging to a session, sorted by run name.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    output_base_directory : str or None, optional
        Directory the mirrored output tree is written into, or ``None`` (the default)
        to look inside the session folder.
    data_root : str or None, optional
        Directory the session folders are selected inside.  Required whenever
        ``output_base_directory`` is given.

    Returns
    -------
    list of str
        Absolute paths of the session's output directories, sorted deterministically:
        numeric run names first (sorted numerically), then non-numeric run names
        (sorted case-insensitively).
    """
    root = run_directory_root(
        session_path=session_path, output_base_directory=output_base_directory, data_root=data_root
    )
    if output_base_directory is None:
        # Inside a session folder every "*_output_*" child is one of its own runs, whatever
        # prefix it carries — so a session folder renamed after an analysis keeps its runs.
        pattern = "*" + _RUN_NAME_MARKER + "*"
    else:
        # The mirror gives each session a directory of its own, so every run inside it is
        # named for the run alone.
        pattern = _RUN_FOLDER_PREFIX + "*"
    candidates = [str(path) for path in Path(root).glob(pattern) if path.is_dir()]
    return sorted(candidates, key=_run_name_sort_key_for_path)


def sibling_run_folders(run_folder: str) -> list[str]:
    """Return every output directory written beside ``run_folder`` for the same session.

    Parameters
    ----------
    run_folder : str
        Path to an output directory.

    Returns
    -------
    list of str
        Absolute paths of the session's output directories, ``run_folder``
        included, sorted by run name.
    """
    root = str(Path(_normalize(run_folder)).parent)
    candidates = [str(path) for path in Path(root).glob("*" + _RUN_FOLDER_PREFIX + "*") if path.is_dir()]
    return sorted(candidates, key=_run_name_sort_key_for_path)


def run_folder_for_run(
    session_path: str, run_name: str, *, output_base_directory: str | None = None, data_root: str | None = None
) -> str:
    """Build the path of the output directory for a given run name.

    Does not check whether the directory exists.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    run_name : str
        Run name the output directory is for.
    output_base_directory : str or None, optional
        Directory the mirrored output tree is written into, or ``None`` (the default)
        to write the output directory inside the session folder.
    data_root : str or None, optional
        Directory the session folders are selected inside.  Required whenever
        ``output_base_directory`` is given.

    Returns
    -------
    str
        Path of the output directory.
    """
    root = run_directory_root(
        session_path=session_path, output_base_directory=output_base_directory, data_root=data_root
    )
    basename = run_folder_basename(
        session_path=session_path, run_name=run_name, output_base_directory=output_base_directory
    )
    return str(Path(root) / basename)


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


def select_run_folders(session_path: str, *, inputParameters: dict[str, object]) -> list[str]:
    """Return the output directories the run selection names for one session.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    inputParameters : dict
        Full pipeline input parameters; supplies ``selected_runs`` (the run names
        to keep, per session) and ``output_base_directory`` / ``data_root`` (where
        the session's output directories live).

    Returns
    -------
    list of str
        Absolute paths of the selected output directories.

    Raises
    ------
    ValueError
        When the session has no selected run names, when a requested run name has
        no matching directory, or when a selected directory is missing
        ``storesList.csv``. The error message lists the available run names so
        the user can correct their input.
    """
    output_base_directory = inputParameters.get("output_base_directory")
    data_root = inputParameters.get("data_root")
    selected_runs = (inputParameters.get("selected_runs") or {}).get(session_path)
    if not selected_runs:
        raise ValueError(
            f"select_run_folders requires an explicit non-empty list of run names for session "
            f"{session_path!r}; got {selected_runs!r}. Pick at least one existing run "
            "directory in the Output Folder Selection panel."
        )
    available = discover_run_folders(session_path, output_base_directory=output_base_directory, data_root=data_root)
    available_by_name = {parse_run_name(directory): directory for directory in available}
    missing = [run for run in selected_runs if run not in available_by_name]
    if missing:
        raise ValueError(
            f"Output directory not found in {session_path!r} for run name(s) {missing!r}. "
            f"Available runs: {sorted(available_by_name.keys())!r}. "
            "Either run step 1 with the requested run name first, or update the selected_runs filter."
        )

    selected = [available_by_name[run] for run in selected_runs]
    missing_stores = [run_folder for run_folder in selected if not (Path(run_folder) / "storesList.csv").exists()]
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
    basename = Path(str(group_folder).rstrip("/\\")).name
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
    parent_directories = {str(Path(path).parent) for path in paths}
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
    basename = Path(str(path).rstrip("/\\")).name
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
    candidates = [str(path) for path in Path(destination_directory).glob("*" + _GROUP_NAME_MARKER) if path.is_dir()]
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
    return str(Path(destination_directory) / (group_name + _GROUP_NAME_MARKER))


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
        return (2, 0, Path(path).name.casefold())
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
        hdf5_path = Path(filepath) / (event + f"_{name}.h5")
    else:
        hdf5_path = Path(filepath) / (event + ".h5")
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
        Pipeline configuration; must include ``'combine_data'``, ``'selected_runs'``,
        ``'output_base_directory'`` and ``'data_root'``.

    Returns
    -------
    list of str
        The resolved run folders.
    """
    run_folders: list[str] = []
    for session in session_folders:
        run_folders.append(select_run_folders(session, inputParameters=inputParameters))
    run_folders = list(np.concatenate(run_folders).flatten())

    if inputParameters["combine_data"] == True:
        return [group[0] for group in get_all_stores_for_combining_data(run_folders)]
    return run_folders
