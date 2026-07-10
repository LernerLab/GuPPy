import logging
import os

import numpy as np
import pandas as pd

from .io_utils import (
    decide_naming_convention,
    fetchCoords,
    get_control_and_signal_channel_names,
    read_hdf5,
    region_from_channel_label,
    region_from_channel_path,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def read_control_and_signal(
    filepath: str, store_array: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    """
    Load control and signal channel arrays from HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    store_label_to_data : dict
        Store label → data array.
    store_label_to_timestamps : dict
        Store label → timestamp array.
    store_label_to_sampling_rate : dict
        Store label → sampling-rate array.
    store_label_to_npoints : dict
        Store label → npoints array (or None for CSV datasets).
    """
    control_signal_names = get_control_and_signal_channel_names(store_array)
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]

    store_label_to_data = {}
    store_label_to_timestamps = {}
    store_label_to_sampling_rate = {}
    store_label_to_npoints = {}

    for i in range(control_signal_names.shape[1]):
        control_name = control_signal_names[0, i]
        signal_name = control_signal_names[1, i]
        control_index = np.where(store_labels == control_name)[0]
        signal_index = np.where(store_labels == signal_name)[0]
        control_store_id = store_ids[control_index[0]]
        signal_store_id = store_ids[signal_index[0]]

        control_data = read_hdf5(control_store_id, filepath, "data")
        signal_data = read_hdf5(signal_store_id, filepath, "data")
        control_timestamps = read_hdf5(control_store_id, filepath, "timestamps")
        signal_timestamps = read_hdf5(signal_store_id, filepath, "timestamps")
        control_sampling_rate = read_hdf5(control_store_id, filepath, "sampling_rate")
        signal_sampling_rate = read_hdf5(signal_store_id, filepath, "sampling_rate")
        try:  # TODO: define npoints for csv datasets
            control_npoints = read_hdf5(control_store_id, filepath, "npoints")
            signal_npoints = read_hdf5(signal_store_id, filepath, "npoints")
        except KeyError:  # npoints is not defined for csv datasets
            control_npoints = None
            signal_npoints = None

        store_label_to_data[control_name] = control_data
        store_label_to_data[signal_name] = signal_data
        store_label_to_timestamps[control_name] = control_timestamps
        store_label_to_timestamps[signal_name] = signal_timestamps
        store_label_to_sampling_rate[control_name] = control_sampling_rate
        store_label_to_sampling_rate[signal_name] = signal_sampling_rate
        store_label_to_npoints[control_name] = control_npoints
        store_label_to_npoints[signal_name] = signal_npoints

    return store_label_to_data, store_label_to_timestamps, store_label_to_sampling_rate, store_label_to_npoints


def read_ttl(filepath: str, store_array: np.ndarray) -> dict[str, np.ndarray]:
    """
    Load TTL event timestamps from HDF5 files, skipping control/signal channels.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    store_label_to_timestamps : dict
        Store label → TTL timestamp array for each non-channel store.
    """
    control_signal_names = get_control_and_signal_channel_names(store_array)
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]

    store_label_to_timestamps = {}
    for store_id, store_label in zip(store_ids, store_labels):
        if store_label in control_signal_names:
            continue
        timestamps = read_hdf5(store_id, filepath, "timestamps")
        store_label_to_timestamps[store_label] = timestamps

    return store_label_to_timestamps


def write_corrected_timestamps(
    filepath: str,
    corrected_name_to_timestamps: dict[str, np.ndarray],
    store_label_to_timestamps: dict[str, np.ndarray],
    store_label_to_sampling_rate: dict[str, np.ndarray],
    store_label_to_correction_index: dict[str, np.ndarray],
) -> None:
    """
    Write timestamp-correction HDF5 datasets for all channel pairs.

    Parameters
    ----------
    filepath : str
        Session output directory.
    corrected_name_to_timestamps : dict
        Store label → corrected timestamp array.
    store_label_to_timestamps : dict
        Store label → original timestamp array.
    store_label_to_sampling_rate : dict
        Store label → sampling-rate array.
    store_label_to_correction_index : dict
        Store label → index array used to slice the original timestamps.
    """
    for name, correctionIndex in store_label_to_correction_index.items():
        timestamps = store_label_to_timestamps[name]
        corrected_timestamps = corrected_name_to_timestamps[name]
        sampling_rate = store_label_to_sampling_rate[name]
        if sampling_rate.shape == ():  # numpy scalar
            sampling_rate = np.asarray([sampling_rate])
        name_1 = region_from_channel_label(name)
        write_hdf5(np.asarray([timestamps[0]]), "timeCorrection_" + name_1, filepath, "timeRecStart")
        write_hdf5(corrected_timestamps, "timeCorrection_" + name_1, filepath, "timestampNew")
        write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
        write_hdf5(sampling_rate, "timeCorrection_" + name_1, filepath, "sampling_rate")


def write_corrected_data(filepath: str, store_label_to_corrected_data: dict[str, np.ndarray]) -> None:
    """
    Write corrected data arrays to HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_label_to_corrected_data : dict
        Store label → corrected data array.
    """
    for name, data in store_label_to_corrected_data.items():
        write_hdf5(data, name, filepath, "data")


def write_corrected_ttl_timestamps(
    filepath: str,
    compound_name_to_corrected_ttl_timestamps: dict[str, np.ndarray],
) -> None:
    """
    Write corrected TTL timestamp arrays to HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    compound_name_to_corrected_ttl_timestamps : dict
        Compound TTL name → corrected TTL timestamp array.
    """
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    for compound_name, corrected_ttl_timestamps in compound_name_to_corrected_ttl_timestamps.items():
        write_hdf5(corrected_ttl_timestamps, compound_name, filepath, "ts")
    logger.info("Timestamps corrections applied to the data and event timestamps.")


def read_corrected_data(
    control_path: str, signal_path: str, filepath: str, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load corrected control, signal, and timestamp arrays for one channel pair.

    Parameters
    ----------
    control_path : str
        Path to the control HDF5 file.
    signal_path : str
        Path to the signal HDF5 file.
    filepath : str
        Session output directory (used to find the ``timeCorrection_<name>`` file).
    name : str
        Channel pair suffix used to locate the timestamp-correction file.

    Returns
    -------
    control : np.ndarray
        1-D corrected control data array.
    signal : np.ndarray
        1-D corrected signal data array.
    tsNew : np.ndarray
        Corrected timestamp array.
    """
    control = read_hdf5("", control_path, "data").reshape(-1)
    signal = read_hdf5("", signal_path, "data").reshape(-1)
    tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")

    return control, signal, tsNew


def write_zscore(
    filepath: str,
    name: str,
    z_score: np.ndarray,
    dff: np.ndarray,
    control_fit: np.ndarray,
    synthetic_control: np.ndarray | None,
) -> None:
    """
    Write z-score, dF/F, and fitted-control arrays to HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name : str
        Channel pair suffix used in HDF5 key names.
    z_score : np.ndarray
        Z-scored signal array.
    dff : np.ndarray
        Delta-F/F signal array.
    control_fit : np.ndarray
        Fitted control channel array.
    synthetic_control : np.ndarray or None
        Synthetic control array when no isosbestic control is present; None otherwise.
    """
    write_hdf5(z_score, "z_score_" + name, filepath, "data")
    write_hdf5(dff, "dff_" + name, filepath, "data")
    write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
    if synthetic_control is not None:
        write_hdf5(synthetic_control, "control_" + name, filepath, "data")


def read_corrected_timestamps_pairwise(
    filepath: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Load corrected timestamps and sampling rates for all channel pairs.

    Parameters
    ----------
    filepath : str
        Session output directory.

    Returns
    -------
    pair_name_to_tsNew : dict
        Pair name → corrected timestamp array.
    pair_name_to_sampling_rate : dict
        Pair name → sampling rate (Hz).
    """
    pair_name_to_tsNew = {}
    pair_name_to_sampling_rate = {}
    path = decide_naming_convention(filepath)
    for j in range(path.shape[1]):
        name_1 = region_from_channel_path(path[0, j])
        name_2 = region_from_channel_path(path[1, j])
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepath}': control file region '{name_1}' does not match "
                f"signal file region '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
        name = name_1

        tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
        sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]
        pair_name_to_tsNew[name] = tsNew
        pair_name_to_sampling_rate[name] = sampling_rate
    return pair_name_to_tsNew, pair_name_to_sampling_rate


def read_coords_pairwise(filepath: str, pair_name_to_tsNew: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Load artifact-removal boundary coordinates for all channel pairs.

    Parameters
    ----------
    filepath : str
        Session output directory.
    pair_name_to_tsNew : dict
        Pair name → corrected timestamp array (used as fallback bounds).

    Returns
    -------
    pair_name_to_coords : dict
        Pair name → shape ``(N, 2)`` array of good-chunk boundaries.
    """
    pair_name_to_coords = {}
    path = decide_naming_convention(filepath)
    for j in range(path.shape[1]):
        name_1 = region_from_channel_path(path[0, j])
        name_2 = region_from_channel_path(path[1, j])
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepath}': control file region '{name_1}' does not match "
                f"signal file region '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
        pair_name = name_1

        tsNew = pair_name_to_tsNew[pair_name]
        coords = fetchCoords(filepath, pair_name, tsNew)
        pair_name_to_coords[pair_name] = coords
    return pair_name_to_coords


def read_corrected_data_dict(
    filepath: str, store_array: np.ndarray
) -> dict[str, np.ndarray]:  # TODO: coordinate with read_corrected_data
    """
    Load corrected control and signal data arrays into a flat dict.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    store_label_to_corrected_data : dict
        Store label → 1-D corrected data array.
    """
    store_label_to_corrected_data = {}
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]
    control_and_signal_names = get_control_and_signal_channel_names(store_array)

    for store_id, store_label in zip(store_ids, store_labels):
        if store_label not in control_and_signal_names:
            continue
        data = read_hdf5(store_label, filepath, "data").reshape(-1)
        store_label_to_corrected_data[store_label] = data

    return store_label_to_corrected_data


def read_corrected_ttl_timestamps(filepath: str, store_array: np.ndarray) -> dict[str, np.ndarray]:
    """
    Load corrected TTL timestamps for all non-channel stores.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    compound_name_to_ttl_timestamps : dict
        Compound TTL name → TTL timestamp array.
    """
    compound_name_to_ttl_timestamps = {}
    store_ids = store_array[0, :]
    store_labels = store_array[1, :]
    control_signal_names = get_control_and_signal_channel_names(store_array)

    for store_id, store_label in zip(store_ids, store_labels):
        if store_label in control_signal_names:
            continue
        ttl_name = store_label
        for i in range(control_signal_names.shape[1]):
            name_1 = region_from_channel_label(control_signal_names[0, i])
            name_2 = region_from_channel_label(control_signal_names[1, i])
            if name_1 != name_2:
                message = (
                    f"Pair name mismatch in storesList: control channel '{control_signal_names[0, i]}' has region "
                    f"'{name_1}' but signal channel '{control_signal_names[1, i]}' has region '{name_2}'. Check the "
                    "naming convention of your files and the storesList file, then re-run step 1."
                )
                logger.error(message)
                raise ValueError(message)
            compound_name = ttl_name + "_" + name_1
            ttl_timestamps = read_hdf5(compound_name, filepath, "ts")
            compound_name_to_ttl_timestamps[compound_name] = ttl_timestamps

    return compound_name_to_ttl_timestamps


def write_artifact_corrected_timestamps(
    filepath: str, pair_name_to_corrected_timestamps: dict[str, np.ndarray]
) -> None:
    """
    Write artifact-corrected timestamp arrays to the ``timeCorrection_*`` HDF5 keys.

    Parameters
    ----------
    filepath : str
        Session output directory.
    pair_name_to_corrected_timestamps : dict
        Pair name → corrected timestamp array.
    """
    for pair_name, timestamps in pair_name_to_corrected_timestamps.items():
        write_hdf5(timestamps, "timeCorrection_" + pair_name, filepath, "timestampNew")


def write_artifact_removal(
    filepath: str,
    store_label_to_corrected_data: dict[str, np.ndarray],
    pair_name_to_corrected_timestamps: dict[str, np.ndarray] | None,
    compound_name_to_corrected_ttl_timestamps: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Write all artifact-removal outputs (data, TTL timestamps, and optionally timestamps) to HDF5.

    Parameters
    ----------
    filepath : str
        Session output directory.
    store_label_to_corrected_data : dict
        Store label → corrected data array.
    pair_name_to_corrected_timestamps : dict or None
        Pair name → corrected timestamp array; skipped when None.
    compound_name_to_corrected_ttl_timestamps : dict, optional
        Compound TTL name → corrected TTL timestamp array. Default is None.
    """
    write_corrected_data(filepath, store_label_to_corrected_data)
    write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)
    if pair_name_to_corrected_timestamps is not None:
        write_artifact_corrected_timestamps(filepath, pair_name_to_corrected_timestamps)


def read_timestamps_for_combining_data(
    filepaths_to_combine: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load corrected timestamps from all session files for the combine-data step.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories.

    Returns
    -------
    pair_name_to_filepath_to_timestamps : dict
        ``{pair_name: {filepath: timestamps_array}}``.
    """
    path = decide_naming_convention(filepaths_to_combine[0])
    pair_name_to_filepath_to_timestamps: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = region_from_channel_path(path[0, j])
        name_2 = region_from_channel_path(path[1, j])
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file region '{name_1}' does not match "
                f"signal file region '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
        pair_name = name_1
        pair_name_to_filepath_to_timestamps[pair_name] = {}
        for filepath in filepaths_to_combine:
            tsNew = read_hdf5("timeCorrection_" + pair_name, filepath, "timestampNew")
            pair_name_to_filepath_to_timestamps[pair_name][filepath] = tsNew

    return pair_name_to_filepath_to_timestamps


def read_data_for_combining_data(
    filepaths_to_combine: list[str], store_array: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load corrected channel data from all session files for the combine-data step.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    store_label_to_filepath_to_data : dict
        ``{display_name: {filepath: data_array}}``.
    """
    store_labels = store_array[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    store_label_to_filepath_to_data: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = region_from_channel_path(path[0, j])
        name_2 = region_from_channel_path(path[1, j])
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file region '{name_1}' does not match "
                f"signal file region '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
        pair_name = name_1
        for i in range(len(store_labels)):
            if not (
                "control_" + pair_name.lower() in store_labels[i].lower()
                or "signal_" + pair_name.lower() in store_labels[i].lower()
            ):
                continue
            display_name = store_labels[i]
            store_label_to_filepath_to_data[display_name] = {}
            for filepath in filepaths_to_combine:
                data = read_hdf5(display_name, filepath, "data").reshape(-1)
                store_label_to_filepath_to_data[display_name][filepath] = data

    return store_label_to_filepath_to_data


def read_ttl_timestamps_for_combining_data(
    filepaths_to_combine: list[str], store_array: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load corrected TTL timestamps from all session files for the combine-data step.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories.
    store_array : np.ndarray
        2-D array with rows [store_id, store_label].

    Returns
    -------
    compound_name_to_filepath_to_ttl_timestamps : dict
        ``{compound_name: {filepath: ttl_timestamps_array}}``.
    """
    store_labels = store_array[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    compound_name_to_filepath_to_ttl_timestamps: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = region_from_channel_path(path[0, j])
        name_2 = region_from_channel_path(path[1, j])
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file region '{name_1}' does "
                f"not match signal file region '{name_2}'. Check the naming convention of your files and "
                "the storesList file, then re-run step 1."
            )
            logger.error(message)
            raise ValueError(message)
        pair_name = name_1
        for i in range(len(store_labels)):
            if (
                "control_" + pair_name.lower() in store_labels[i].lower()
                or "signal_" + pair_name.lower() in store_labels[i].lower()
            ):
                continue
            compound_name = store_labels[i] + "_" + pair_name
            compound_name_to_filepath_to_ttl_timestamps[compound_name] = {}
            for filepath in filepaths_to_combine:
                if os.path.exists(os.path.join(filepath, store_labels[i] + "_" + pair_name + ".hdf5")):
                    ttl_timestamps = read_hdf5(store_labels[i] + "_" + pair_name, filepath, "ts").reshape(-1)
                else:
                    ttl_timestamps = np.array([])
                compound_name_to_filepath_to_ttl_timestamps[compound_name][filepath] = ttl_timestamps

    return compound_name_to_filepath_to_ttl_timestamps


def write_combined_data(
    output_filepath: str,
    pair_name_to_tsNew: dict[str, np.ndarray],
    store_label_to_data: dict[str, np.ndarray],
    compound_name_to_ttl_timestamps: dict[str, np.ndarray],
) -> None:
    """
    Write combined multi-session data (timestamps, channel data, TTLs) to HDF5.

    Parameters
    ----------
    output_filepath : str
        Destination session output directory.
    pair_name_to_tsNew : dict
        Pair name → combined timestamp array.
    store_label_to_data : dict
        Store label → combined data array.
    compound_name_to_ttl_timestamps : dict
        Compound TTL name → combined TTL timestamp array.
    """
    for pair_name, tsNew in pair_name_to_tsNew.items():
        write_hdf5(tsNew, "timeCorrection_" + pair_name, output_filepath, "timestampNew")
    for display_name, data in store_label_to_data.items():
        write_hdf5(data, display_name, output_filepath, "data")
    for compound_name, ttl_timestamps in compound_name_to_ttl_timestamps.items():
        write_hdf5(ttl_timestamps, compound_name, output_filepath, "ts")


def write_peak_and_area_to_hdf5(filepath: str, peak_and_area_data: object, name: str, index: list[object] = []) -> None:
    """
    Save peak and area-under-curve metrics to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory.
    peak_and_area_data : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``peak_AUC_<name>.h5``.
    index : list, optional
        Row index labels. Default is an empty list.
    """

    output_path = os.path.join(filepath, "peak_AUC_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(peak_and_area_data, index=index)

    df.to_hdf(output_path, key="df", mode="w")


def write_peak_and_area_to_csv(filepath: str, peak_and_area_data: object, name: str, index: list[object] = []) -> None:
    """
    Save peak and area-under-curve metrics to a CSV file.

    Parameters
    ----------
    filepath : str
        Output directory.
    peak_and_area_data : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``peak_AUC_<name>.csv``.
    index : list, optional
        Row index labels. Default is an empty list.
    """
    output_path = os.path.join(filepath, "peak_AUC_" + name + ".csv")
    df = pd.DataFrame(peak_and_area_data, index=index)

    df.to_csv(output_path)


def write_freq_and_amp_to_hdf5(
    filepath: str, freq_and_amp_data: object, name: str, index: list[object] = [], columns: list[object] = []
) -> None:
    """
    Save transient frequency and amplitude metrics to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory.
    freq_and_amp_data : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``freqAndAmp_<name>.h5``.
    index : list, optional
        Row index labels. Default is an empty list.
    columns : list, optional
        Column labels. Default is an empty list.
    """

    output_path = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(freq_and_amp_data, index=index, columns=columns)

    df.to_hdf(output_path, key="df", mode="w")


def write_freq_and_amp_to_csv(
    filepath: str, freq_and_amp_data: object, name: str, index: list[object] = [], columns: list[object] = []
) -> None:
    """
    Save transient frequency and amplitude metrics to a CSV file.

    Parameters
    ----------
    filepath : str
        Output directory.
    freq_and_amp_data : array-like
        Metrics data to store in the DataFrame.
    name : str
        Output filename (written directly inside ``filepath``).
    index : list, optional
        Row index labels. Default is an empty list.
    columns : list, optional
        Column labels. Default is an empty list.
    """
    output_path = os.path.join(filepath, name)
    df = pd.DataFrame(freq_and_amp_data, index=index, columns=columns)
    df.to_csv(output_path)


def read_freq_and_amp_from_hdf5(filepath: str, name: str) -> pd.DataFrame:
    """
    Load transient frequency and amplitude metrics from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Directory containing the ``freqAndAmp_<name>.h5`` file.
    name : str
        Filename stem (without the ``freqAndAmp_`` prefix or ``.h5`` suffix).

    Returns
    -------
    df : pd.DataFrame
        DataFrame of frequency and amplitude metrics.
    """
    output_path = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    df = pd.read_hdf(output_path, key="df", mode="r")

    return df


def write_transients_to_hdf5(
    filepath: str, name: str, z_score: np.ndarray, timestamps: np.ndarray, peaksInd: np.ndarray
) -> None:
    """
    Write transient detection outputs (z-score, timestamps, peak indices) to HDF5.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name : str
        Channel suffix used to build the HDF5 event key.
    z_score : np.ndarray
        Z-scored signal array (NaN-free).
    timestamps : np.ndarray
        Timestamp array corresponding to ``z_score``.
    peaksInd : np.ndarray
        Integer indices of detected transient peaks in ``z_score``.
    """
    event = f"transient_outputs_{name}"
    write_hdf5(z_score, event, filepath, "z_score")
    write_hdf5(timestamps, event, filepath, "timestamps")
    write_hdf5(peaksInd, event, filepath, "peaksInd")


def read_transients_from_hdf5(filepath: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load transient detection outputs from HDF5.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name : str
        Channel suffix used to build the HDF5 event key.

    Returns
    -------
    z_score : np.ndarray
        Z-scored signal array.
    timestamps : np.ndarray
        Timestamp array.
    peaksInd : np.ndarray
        Integer indices of detected transient peaks.
    """
    event = f"transient_outputs_{name}"
    z_score = read_hdf5(event, filepath, "z_score")
    timestamps = read_hdf5(event, filepath, "timestamps")
    peaksInd = read_hdf5(event, filepath, "peaksInd")
    return z_score, timestamps, peaksInd
