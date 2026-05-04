import glob
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_RUN_NAME_MARKER = "_output_"
_FORBIDDEN_RUN_NAME_CHARACTERS = ("/", "\\", ":", "\0")


def takeOnlyDirs(paths):
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
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


def parse_run_name(output_dir):
    """Return the run-name suffix of an output directory.

    Splits the directory's basename on the last occurrence of ``_output_`` and
    returns everything after it.  Legacy ``mySession_output_1`` directories
    yield ``"1"``.

    Parameters
    ----------
    output_dir : str
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
    basename = os.path.basename(output_dir.rstrip("/\\"))
    index = basename.rfind(_RUN_NAME_MARKER)
    if index < 0:
        raise ValueError(
            f"Cannot parse run name from {output_dir!r}: basename {basename!r} does not match "
            f"'<session_basename>_output_<run_name>' pattern."
        )
    return basename[index + len(_RUN_NAME_MARKER) :]


def discover_output_dirs(session_path):
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


def output_dir_for_run(session_path, run_name):
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


def select_output_dirs(session_path, selected_runs):
    """Filter a session's output directories to those matching ``selected_runs``.

    Parameters
    ----------
    session_path : str
        Path to a session folder.
    selected_runs : list of str or None
        Run-name suffixes to keep.  When ``None`` (or empty) all output dirs
        are returned (back-compat with the pre-parametrized pipeline).

    Returns
    -------
    list of str
        Absolute paths of the selected output directories.

    Raises
    ------
    ValueError
        When a requested run name has no matching directory or when a
        selected directory is missing ``storesList.csv``.  The error message
        lists the available run names so the user can correct their input.
    """
    available = discover_output_dirs(session_path)
    if not selected_runs:
        return available

    available_by_name = {parse_run_name(directory): directory for directory in available}
    missing = [run for run in selected_runs if run not in available_by_name]
    if missing:
        raise ValueError(
            f"Output directory not found in {session_path!r} for run name(s) {missing!r}. "
            f"Available runs: {sorted(available_by_name.keys())!r}. "
            "Either run step 2 with the requested run name first, or update the selectedOutputs filter."
        )

    selected = [available_by_name[run] for run in selected_runs]
    missing_stores_list = [d for d in selected if not os.path.exists(os.path.join(d, "storesList.csv"))]
    if missing_stores_list:
        raise ValueError(
            f"Selected output directories are missing storesList.csv: {missing_stores_list!r}. "
            "Re-run step 2 (Save Storenames) for these run names before continuing."
        )
    return sorted(selected, key=_run_name_sort_key_for_path)


def validate_run_name(run_name):
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
        raise ValueError(f"runName must be a string; got {type(run_name).__name__}.")
    if not run_name:
        raise ValueError("runName must be a non-empty string.")
    if run_name.strip() != run_name or not run_name.strip():
        raise ValueError(f"runName {run_name!r} must not contain leading/trailing whitespace or be all whitespace.")
    for character in _FORBIDDEN_RUN_NAME_CHARACTERS:
        if character in run_name:
            raise ValueError(
                f"runName {run_name!r} contains forbidden character {character!r}. "
                f"Path separators and null bytes are not allowed."
            )
    if ".." in run_name:
        raise ValueError(f"runName {run_name!r} must not contain '..' (path traversal).")
    if _RUN_NAME_MARKER in run_name:
        raise ValueError(
            f"runName {run_name!r} must not contain the substring {_RUN_NAME_MARKER!r}; "
            "this would break parsing of the output directory name."
        )


def _run_name_sort_key(run_name):
    """Sort key that orders numeric run names ahead of alphanumeric ones."""
    try:
        return (0, int(run_name), "")
    except ValueError:
        return (1, 0, run_name.casefold())


def _run_name_sort_key_for_path(path):
    """Sort key that orders output-directory paths by their run-name suffix."""
    try:
        run_name = parse_run_name(path)
    except ValueError:
        return (2, 0, os.path.basename(path).casefold())
    return _run_name_sort_key(run_name)


def get_all_stores_for_combining_data(folderNames):
    """Group output directories by run-name suffix for cross-session combining.

    Parameters
    ----------
    folderNames : list of str
        Paths to ``<basename>_output_<run_name>`` directories across all sessions.

    Returns
    -------
    list of list of str
        One inner list per distinct run name.  Inner lists are sorted
        case-insensitively by path; outer ordering puts numeric run names
        first (numerically) and then alphanumeric run names (case-insensitive).
    """
    run_name_to_paths = {}
    for path in folderNames:
        try:
            run_name = parse_run_name(path)
        except ValueError:
            continue
        run_name_to_paths.setdefault(run_name, []).append(path)

    ordered_run_names = sorted(run_name_to_paths.keys(), key=_run_name_sort_key)
    return [sorted(run_name_to_paths[name], key=str.casefold) for name in ordered_run_names]


def read_Df(filepath, event, name):
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
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df
