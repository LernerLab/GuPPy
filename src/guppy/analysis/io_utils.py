import fnmatch
import glob
import logging
import os
import re

import numpy as np

from ..utils._hdf5_io import read_hdf5, write_hdf5  # noqa: F401  (re-exported)
from ..utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)


def find_files(path, glob_path, ignore_case=False):
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
    str_path = []

    # converting byte object to string
    for x in no_bytes_path:
        try:
            str_path.append(x.decode("utf-8"))
        except:
            str_path.append(x)
    return [os.path.join(path, n) for n in str_path if rule.match(n)]


def check_TDT(filepath):
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


def decide_naming_convention(filepath):
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
    path_1 = find_files(filepath, "control_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'control*'))

    path_2 = find_files(filepath, "signal_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'signal*'))

    path = sorted(path_1 + path_2, key=str.casefold)
    if len(path) % 2 != 0:
        message = (
            f"Unequal number of control and signal files in '{filepath}': "
            f"found {len(path_1)} control and {len(path_2)} signal file(s). "
            "Each signal must be paired with a control; re-run step 2 to fix the entries."
        )
        logger.error(message)
        raise ValueError(message)

    path = np.asarray(path).reshape(2, -1)

    return path


def fetchCoords(filepath, naming, data):
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


def get_coords(filepath, name, tsNew, removeArtifacts):  # TODO: Make less redundant with fetchCoords
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


def check_storeslistfile(folderNames):
    """
    Merge storesList CSVs from all session output directories.

    Parameters
    ----------
    folderNames : list of str
        Session directories whose output subdirectories contain ``storesList.csv`` files.

    Returns
    -------
    storesList : np.ndarray
        2-D array with rows [storenames, display_names] merged across all sessions.
    """
    storesList = np.array([[], []])
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))
        for j in range(len(storesListPath)):
            filepath = storesListPath[j]
            storesList = np.concatenate(
                (
                    storesList,
                    np.genfromtxt(os.path.join(filepath, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1),
                ),
                axis=1,
            )

    storesList = np.unique(storesList, axis=1)

    return storesList


def write_combined_stores_list(op, storesList):
    """
    Write a combined storesList CSV to each output directory.

    Parameters
    ----------
    op : list
        Sequence of ``[filepath, ...]`` entries; ``filepath`` is the output directory.
    storesList : np.ndarray
        2-D storesList array with rows [storenames, display_names].
    """
    for k in range(len(op)):
        filepath = op[k][0]
        np.savetxt(os.path.join(filepath, "combine_storesList.csv"), storesList, fmt="%s", delimiter=",")


def get_control_and_signal_channel_names(storesList):
    """
    Extract and pair control/signal display names from a storesList array.

    Parameters
    ----------
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    channels_arr : np.ndarray
        Shape ``(2, N)`` array where row 0 is control display names and
        row 1 is the matching signal display names.
    """
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    channels_arr = []
    for i in range(names_for_storenames.shape[0]):
        if "control" in names_for_storenames[i].lower() or "signal" in names_for_storenames[i].lower():
            channels_arr.append(names_for_storenames[i])

    channels_arr = sorted(channels_arr, key=str.casefold)

    signal_regions = {name[len("signal_") :] for name in channels_arr if name.lower().startswith("signal_")}
    control_regions = {name[len("control_") :] for name in channels_arr if name.lower().startswith("control_")}
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
                "isosbestic control is enabled. Re-run step 2 (Storenames) to fix the region names."
            )
            logger.error(message)
            raise ValueError(message)

    try:
        channels_arr = np.asarray(channels_arr).reshape(2, -1)
    except ValueError:
        message = (
            f"Cannot pair control and signal channels: found {len(control_regions)} control and "
            f"{len(signal_regions)} signal entries in storesList. Each signal must be paired with a control "
            "when isosbestic control is enabled; re-run step 2 (Storenames) to correct the entries."
        )
        logger.error(message)
        raise ValueError(message)

    return channels_arr


def make_dir_for_cross_correlation(filepath):
    """
    Create and return the cross-correlation output subdirectory.

    Parameters
    ----------
    filepath : str
        Parent directory inside which ``cross_correlation_output/`` is created.

    Returns
    -------
    op : str
        Path to the cross-correlation output directory.
    """
    op = os.path.join(filepath, "cross_correlation_output")
    if not os.path.exists(op):
        os.mkdir(op)
    return op


def makeAverageDir(filepath):
    """
    Create and return the group-average output subdirectory.

    Parameters
    ----------
    filepath : str
        Parent directory inside which ``average/`` is created.

    Returns
    -------
    op : str
        Path to the average output directory.
    """

    op = os.path.join(filepath, "average")
    if not os.path.exists(op):
        os.mkdir(op)

    return op
