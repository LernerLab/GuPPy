import glob
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_RUN_NAME_MARKER = "_output_"
_FORBIDDEN_RUN_NAME_CHARACTERS = ("/", "\\", ":", "\0")

# NPM decomposition parameters chosen interactively in Step 1 are not part of the
# saved analysis parameters, so they are persisted next to storesList.csv for Step 2.
NPM_PARAMS_FILENAME = ".npm_params.json"
NPM_PARAM_KEYS = ("npm_split_events", "npm_time_units", "npm_timestamp_column_names")


def is_headless() -> bool:
    """Report whether GuPPy is running in headless/test mode.

    Headless mode is signalled by the ``GUPPY_BASE_DIR`` environment variable,
    which the testing API sets to bypass the folder dialog. Code paths that
    open GUI dialogs or interactive matplotlib backends should be skipped when
    this returns ``True``.

    Returns
    -------
    bool
        ``True`` when ``GUPPY_BASE_DIR`` is set, ``False`` otherwise.
    """
    return bool(os.environ.get("GUPPY_BASE_DIR"))


def write_npm_params(*, run_folder: str, npm_params: dict[str, object]) -> None:
    """Persist the NPM decomposition parameters for one output directory.

    The interactive NPM choices made during Step 1 (event splitting, timestamp
    units, timestamp column names) determine how :class:`NpmRecordingExtractor`
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
    """
    npm_params_path = os.path.join(run_folder, NPM_PARAMS_FILENAME)
    if not os.path.exists(npm_params_path):
        return {}
    with open(npm_params_path) as file:
        return json.load(file)


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
