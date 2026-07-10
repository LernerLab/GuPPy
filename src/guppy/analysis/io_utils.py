import fnmatch
import glob
import logging
import os
import re

import numpy as np

from ..utils._hdf5_io import read_hdf5, write_hdf5  # noqa: F401  (re-exported)
from ..utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)

SIGNAL_PREFIX = "signal_"
CONTROL_PREFIX = "control_"
ZSCORE_PREFIX = "z_score_"
DFF_PREFIX = "dff_"


def region_from_channel_label(label: str) -> str:
    """
    Return the region/pair name of a control or signal channel label.

    The role is encoded as a fixed leading prefix (``signal_`` / ``control_``), so the
    region is recovered by stripping that prefix rather than by splitting on the last
    underscore. This keeps region names that themselves contain underscores intact.

    Parameters
    ----------
    label : str
        Channel label, e.g. ``"signal_DMS"`` or ``"control_left_hemisphere"``.

    Returns
    -------
    str
        The region/pair name with its original case preserved, e.g. ``"DMS"`` or
        ``"left_hemisphere"``. Labels without a control/signal prefix are returned
        unchanged.
    """
    lowered = label.lower()
    if lowered.startswith(SIGNAL_PREFIX):
        return label[len(SIGNAL_PREFIX) :]
    if lowered.startswith(CONTROL_PREFIX):
        return label[len(CONTROL_PREFIX) :]
    return label


def region_from_preprocessed_label(label: str) -> str:
    """
    Return the region/pair name of a ``z_score_*`` / ``dff_*`` label or basename.

    The prefix is stripped rather than split on the last underscore, so region names
    containing underscores are preserved.

    Parameters
    ----------
    label : str
        Label or extension-stripped basename, e.g. ``"z_score_left_hemisphere"``.

    Returns
    -------
    str
        The region/pair name, e.g. ``"left_hemisphere"``.
    """
    if label.startswith(ZSCORE_PREFIX):
        return label[len(ZSCORE_PREFIX) :]
    if label.startswith(DFF_PREFIX):
        return label[len(DFF_PREFIX) :]
    return label


def region_from_channel_path(path: str) -> str:
    """
    Return the region/pair name of a ``control_*`` / ``signal_*`` HDF5 file path.

    Parameters
    ----------
    path : str
        File path such as ``".../signal_left_hemisphere.hdf5"``.

    Returns
    -------
    str
        The region/pair name, e.g. ``"left_hemisphere"``.
    """
    return region_from_channel_label(os.path.splitext(os.path.basename(path))[0])


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
    control_paths = find_files(filepath, "control_*", ignore_case=True)
    signal_paths = find_files(filepath, "signal_*", ignore_case=True)

    # Pair by region name (fixed-prefix strip) rather than by sort position so that
    # region names containing underscores are handled correctly.
    control_by_region = {region_from_channel_label(os.path.splitext(os.path.basename(p))[0]): p for p in control_paths}
    signal_by_region = {region_from_channel_label(os.path.splitext(os.path.basename(p))[0]): p for p in signal_paths}

    if set(control_by_region) != set(signal_by_region):
        control_without_signal = sorted(set(control_by_region) - set(signal_by_region))
        signal_without_control = sorted(set(signal_by_region) - set(control_by_region))
        parts = []
        if control_without_signal:
            parts.append(f"control file(s) without a matching signal: {', '.join(control_without_signal)}")
        if signal_without_control:
            parts.append(f"signal file(s) without a matching control: {', '.join(signal_without_control)}")
        message = (
            f"Mismatched control/signal files in '{filepath}' — "
            + "; ".join(parts)
            + ". Each signal must be paired with a control; re-run step 1 to fix the entries."
        )
        logger.error(message)
        raise ValueError(message)

    regions = sorted(control_by_region, key=str.casefold)
    path = np.asarray(
        [[control_by_region[region] for region in regions], [signal_by_region[region] for region in regions]]
    )

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
    store_labels = store_array[1, :]

    # Group control and signal labels by their region/pair name (fixed-prefix strip)
    # so pairing is explicit rather than dependent on sort order. This keeps region
    # names that contain underscores intact.
    control_by_region = {}
    signal_by_region = {}
    for label in store_labels:
        lowered = label.lower()
        if lowered.startswith(CONTROL_PREFIX):
            control_by_region[region_from_channel_label(label)] = label
        elif lowered.startswith(SIGNAL_PREFIX):
            signal_by_region[region_from_channel_label(label)] = label

    signal_without_control = sorted(set(signal_by_region) - set(control_by_region), key=str.casefold)
    control_without_signal = sorted(set(control_by_region) - set(signal_by_region), key=str.casefold)
    if signal_without_control or control_without_signal:
        parts = []
        if signal_without_control:
            parts.append(f"signal region(s) without a matching control: {', '.join(signal_without_control)}")
        if control_without_signal:
            parts.append(f"control region(s) without a matching signal: {', '.join(control_without_signal)}")
        message = (
            "Mismatched signal/control region pairs in storesList — "
            + "; ".join(parts)
            + ". Every 'signal_<region>' must have a matching 'control_<region>'. "
            "Re-run step 1 (Label Stores) to fix the region names."
        )
        logger.error(message)
        raise ValueError(message)

    regions = sorted(control_by_region, key=str.casefold)
    control_signal_names = np.asarray(
        [[control_by_region[region] for region in regions], [signal_by_region[region] for region in regions]]
    )

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
