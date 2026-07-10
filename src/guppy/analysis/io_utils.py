import fnmatch
import glob
import logging
import os
import re

import numpy as np

from ..utils._hdf5_io import read_hdf5, write_hdf5  # noqa: F401  (re-exported)
from ..utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)


def find_files(path: str, glob_path: str, ignore_case: bool = False) -> list[str]:
    """
    List files in ``path`` matching a glob pattern, optionally case-insensitively.

    Parameters
    ----------
    path : str
        Directory to search.
    glob_path : str
        Glob-style pattern (e.g. ``'control_*'``).
    ignore_case : bool, optional
        When True, match the pattern case-insensitively. Default is False.

    Returns
    -------
    list of str
        Absolute paths of matching files.
    """
    rule = (
        re.compile(fnmatch.translate(glob_path), re.IGNORECASE)
        if ignore_case
        else re.compile(fnmatch.translate(glob_path))
    )

    no_bytes_path = os.listdir(os.path.expanduser(path))
    decoded_names = []

    # converting byte object to string
    for raw_name in no_bytes_path:
        try:
            decoded_names.append(raw_name.decode("utf-8"))
        except:
            decoded_names.append(raw_name)
    return [os.path.join(path, name) for name in decoded_names if rule.match(name)]


def check_TDT(filepath: str) -> bool:
    """
    Return True if ``filepath`` contains TDT ``.tsq`` files.

    Parameters
    ----------
    filepath : str
        Directory to check.

    Returns
    -------
    bool
        True if at least one ``.tsq`` file exists in the directory.
    """
    path = glob.glob(os.path.join(filepath, "*.tsq"))
    if len(path) > 0:
        return True
    else:
        return False


def decide_naming_convention(filepath: str) -> np.ndarray:
    """
    Find and pair control/signal HDF5 files in ``filepath``.

    Parameters
    ----------
    filepath : str
        Session output directory containing ``control_*`` and ``signal_*`` files.

    Returns
    -------
    path : np.ndarray
        Shape ``(2, N)`` array where row 0 contains control file paths and
        row 1 contains the matching signal file paths.
    """
    path_1 = find_files(filepath, "control_*", ignore_case=True)

    path_2 = find_files(filepath, "signal_*", ignore_case=True)

    path = sorted(path_1 + path_2, key=str.casefold)
    if len(path) % 2 != 0:
        message = (
            f"Unequal number of control and signal files in '{filepath}': "
            f"found {len(path_1)} control and {len(path_2)} signal file(s). "
            "Each signal must be paired with a control; re-run step 1 to fix the entries."
        )
        logger.error(message)
        raise ValueError(message)

    path = np.asarray(path).reshape(2, -1)

    return path


def fetchCoords(filepath: str, naming: str, data: np.ndarray) -> np.ndarray:
    """
    Load artifact-removal boundary coordinates for a channel pair.

    Parameters
    ----------
    filepath : str
        Session output directory to search for the coordinates ``.npy`` file.
    naming : str
        Channel pair suffix used to build the filename.
    data : np.ndarray
        Timestamp array; its last element is used as a fallback end coordinate.

    Returns
    -------
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.
    """

    path = os.path.join(filepath, "coordsForPreProcessing_" + naming + ".npy")

    if not os.path.exists(path):
        coords = np.array([0, data[-1]])
    else:
        coords = np.load(os.path.join(filepath, "coordsForPreProcessing_" + naming + ".npy"))[:, 0]

    if coords.shape[0] % 2 != 0:
        coords_path = os.path.join(filepath, "coordsForPreProcessing_" + naming + ".npy")
        message = (
            f"Coordinates file '{coords_path}' contains {coords.shape[0]} values, but artifact-removal "
            "coordinates must come in pairs (start, end) — i.e. an even count."
        )
        logger.error(message)
        raise ValueError(message)

    coords = coords.reshape(-1, 2)

    return coords


def get_coords(
    filepath: str, name: str, tsNew: np.ndarray, removeArtifacts: bool
) -> np.ndarray:  # TODO: Make less redundant with fetchCoords
    """
    Return artifact-removal boundary coordinates, or a single full-span window.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name : str
        Channel pair suffix used to locate the coordinates file.
    tsNew : np.ndarray
        Corrected timestamp array; defines the full-span fallback window.
    removeArtifacts : bool
        When True, load saved coordinates; when False, return a single window
        spanning the full recording.

    Returns
    -------
    coords : np.ndarray
        Shape ``(N, 2)`` array of ``[start, end]`` bounds for good chunks.
    """
    if removeArtifacts == True:
        coords = fetchCoords(filepath, name, tsNew)
    else:
        dt = tsNew[1] - tsNew[0]
        coords = np.array([[tsNew[0] - dt, tsNew[-1] + dt]])
    return coords


def check_storeslistfile(session_folders: list[str]) -> np.ndarray:
    """
    Merge storesList CSVs from all session output directories.

    Parameters
    ----------
    session_folders : list of str
        Session directories whose output subdirectories contain ``storesList.csv`` files.

    Returns
    -------
    store_array : np.ndarray
        2-D array with rows [store_id, store_label] merged across all sessions.
    """
    store_array = np.array([[], []])
    for i in range(len(session_folders)):
        filepath = session_folders[i]
        run_folders = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(run_folders)):
            filepath = run_folders[j]
            store_array = np.concatenate(
                (
                    store_array,
                    np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
                ),
                axis=1,
            )

    store_array = np.unique(store_array, axis=1)

    return store_array


def write_combined_stores_list(run_folders: list[object], store_array: np.ndarray) -> None:
    """
    Write a combined storesList CSV to each output directory.

    Parameters
    ----------
    run_folders : list
        Sequence of ``[filepath, ...]`` entries; ``filepath`` is the output directory.
    store_array : np.ndarray
        2-D store array with rows [store_id, store_label].
    """
    for k in range(len(run_folders)):
        filepath = run_folders[k][0]
        np.savetxt(os.path.join(filepath, "combine_storesList.csv"), store_array, fmt="%s", delimiter=",")


def get_control_and_signal_channel_names(store_array: np.ndarray) -> np.ndarray:
    """
    Extract and pair control/signal display names from a store_array array.

    Parameters
    ----------
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    control_signal_names : np.ndarray
        Shape ``(2, N)`` array where row 0 is control display names and
        row 1 is the matching signal display names.
    """
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]

    control_signal_names = []
    for i in range(store_labels.shape[0]):
        if "control" in store_labels[i].lower() or "signal" in store_labels[i].lower():
            control_signal_names.append(store_labels[i])

    control_signal_names = sorted(control_signal_names, key=str.casefold)

    signal_regions = {name[len("signal_") :] for name in control_signal_names if name.lower().startswith("signal_")}
    control_regions = {name[len("control_") :] for name in control_signal_names if name.lower().startswith("control_")}
    # Only enforce region pairing when both signal and control channels are present
    # (signal-only / control-only configurations are valid when isosbestic control is disabled).
    if signal_regions and control_regions:
        signal_without_control = sorted(signal_regions - control_regions)
        control_without_signal = sorted(control_regions - signal_regions)
        if signal_without_control or control_without_signal:
            parts = []
            if signal_without_control:
                parts.append(f"signal region(s) without a matching control: {', '.join(signal_without_control)}")
            if control_without_signal:
                parts.append(f"control region(s) without a matching signal: {', '.join(control_without_signal)}")
            message = (
                "Mismatched signal/control region pairs in storesList — "
                + "; ".join(parts)
                + ". Every 'signal_<region>' must have a matching 'control_<region>' when "
                "isosbestic control is enabled. Re-run step 1 (Label Stores) to fix the region names."
            )
            logger.error(message)
            raise ValueError(message)

    try:
        control_signal_names = np.asarray(control_signal_names).reshape(2, -1)
    except ValueError:
        message = (
            f"Cannot pair control and signal channels: found {len(control_regions)} control and "
            f"{len(signal_regions)} signal entries in storesList. Each signal must be paired with a control "
            "when isosbestic control is enabled; re-run step 1 (Label Stores) to correct the entries."
        )
        logger.error(message)
        raise ValueError(message)

    return control_signal_names


def make_dir_for_cross_correlation(filepath: str) -> str:
    """
    Create and return the cross-correlation output subdirectory.

    Parameters
    ----------
    filepath : str
        Parent directory inside which ``cross_correlation_output/`` is created.

    Returns
    -------
    run_folder : str
        Path to the cross-correlation output directory.
    """
    run_folder = os.path.join(filepath, "cross_correlation_output")
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
    return run_folder


def makeAverageDir(filepath: str) -> str:
    """
    Create and return the group-average output subdirectory.

    Parameters
    ----------
    filepath : str
        Parent directory inside which ``average/`` is created.

    Returns
    -------
    run_folder : str
        Path to the average output directory.
    """

    run_folder = os.path.join(filepath, "average")
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    return run_folder
