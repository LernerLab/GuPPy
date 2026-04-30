import logging
import os

import numpy as np
import pandas as pd

from .io_utils import (
    decide_naming_convention,
    fetchCoords,
    get_control_and_signal_channel_names,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


def read_control_and_signal(filepath, storesList):
    """
    Load control and signal channel arrays from HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    name_to_data : dict
        Display name → data array.
    name_to_timestamps : dict
        Display name → timestamp array.
    name_to_sampling_rate : dict
        Display name → sampling-rate array.
    name_to_npoints : dict
        Display name → npoints array (or None for CSV datasets).
    """
    channels_arr = get_control_and_signal_channel_names(storesList)
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    name_to_data = {}
    name_to_timestamps = {}
    name_to_sampling_rate = {}
    name_to_npoints = {}

    for i in range(channels_arr.shape[1]):
        control_name = channels_arr[0, i]
        signal_name = channels_arr[1, i]
        idx_c = np.where(names_for_storenames == control_name)[0]
        idx_s = np.where(names_for_storenames == signal_name)[0]
        control_storename = storenames[idx_c[0]]
        signal_storename = storenames[idx_s[0]]

        control_data = read_hdf5(control_storename, filepath, "data")
        signal_data = read_hdf5(signal_storename, filepath, "data")
        control_timestamps = read_hdf5(control_storename, filepath, "timestamps")
        signal_timestamps = read_hdf5(signal_storename, filepath, "timestamps")
        control_sampling_rate = read_hdf5(control_storename, filepath, "sampling_rate")
        signal_sampling_rate = read_hdf5(signal_storename, filepath, "sampling_rate")
        try:  # TODO: define npoints for csv datasets
            control_npoints = read_hdf5(control_storename, filepath, "npoints")
            signal_npoints = read_hdf5(signal_storename, filepath, "npoints")
        except KeyError:  # npoints is not defined for csv datasets
            control_npoints = None
            signal_npoints = None

        name_to_data[control_name] = control_data
        name_to_data[signal_name] = signal_data
        name_to_timestamps[control_name] = control_timestamps
        name_to_timestamps[signal_name] = signal_timestamps
        name_to_sampling_rate[control_name] = control_sampling_rate
        name_to_sampling_rate[signal_name] = signal_sampling_rate
        name_to_npoints[control_name] = control_npoints
        name_to_npoints[signal_name] = signal_npoints

    return name_to_data, name_to_timestamps, name_to_sampling_rate, name_to_npoints


def read_ttl(filepath, storesList):
    """
    Load TTL event timestamps from HDF5 files, skipping control/signal channels.

    Parameters
    ----------
    filepath : str
        Session output directory.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    name_to_timestamps : dict
        Display name → TTL timestamp array for each non-channel store.
    """
    channels_arr = get_control_and_signal_channel_names(storesList)
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]

    name_to_timestamps = {}
    for storename, name in zip(storenames, names_for_storenames):
        if name in channels_arr:
            continue
        timestamps = read_hdf5(storename, filepath, "timestamps")
        name_to_timestamps[name] = timestamps

    return name_to_timestamps


def write_corrected_timestamps(
    filepath, corrected_name_to_timestamps, name_to_timestamps, name_to_sampling_rate, name_to_correctionIndex
):
    """
    Write timestamp-correction HDF5 datasets for all channel pairs.

    Parameters
    ----------
    filepath : str
        Session output directory.
    corrected_name_to_timestamps : dict
        Display name → corrected timestamp array.
    name_to_timestamps : dict
        Display name → original timestamp array.
    name_to_sampling_rate : dict
        Display name → sampling-rate array.
    name_to_correctionIndex : dict
        Display name → index array used to slice the original timestamps.
    """
    for name, correctionIndex in name_to_correctionIndex.items():
        timestamps = name_to_timestamps[name]
        corrected_timestamps = corrected_name_to_timestamps[name]
        sampling_rate = name_to_sampling_rate[name]
        if sampling_rate.shape == ():  # numpy scalar
            sampling_rate = np.asarray([sampling_rate])
        name_1 = name.split("_")[-1]
        write_hdf5(np.asarray([timestamps[0]]), "timeCorrection_" + name_1, filepath, "timeRecStart")
        write_hdf5(corrected_timestamps, "timeCorrection_" + name_1, filepath, "timestampNew")
        write_hdf5(correctionIndex, "timeCorrection_" + name_1, filepath, "correctionIndex")
        write_hdf5(sampling_rate, "timeCorrection_" + name_1, filepath, "sampling_rate")


def write_corrected_data(filepath, name_to_corrected_data):
    """
    Write corrected data arrays to HDF5 files.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name_to_corrected_data : dict
        Display name → corrected data array.
    """
    for name, data in name_to_corrected_data.items():
        write_hdf5(data, name, filepath, "data")


def write_corrected_ttl_timestamps(
    filepath,
    compound_name_to_corrected_ttl_timestamps,
):
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


def read_corrected_data(control_path, signal_path, filepath, name):
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


def write_zscore(filepath, name, z_score, dff, control_fit, temp_control_arr):
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
    temp_control_arr : np.ndarray or None
        Synthetic control array when no isosbestic control is present; None otherwise.
    """
    write_hdf5(z_score, "z_score_" + name, filepath, "data")
    write_hdf5(dff, "dff_" + name, filepath, "data")
    write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
    if temp_control_arr is not None:
        write_hdf5(temp_control_arr, "control_" + name, filepath, "data")


def read_corrected_timestamps_pairwise(filepath):
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
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        if name_1[-1] != name_2[-1]:
            msg = (
                f"Pair name mismatch in '{filepath}': control file suffix '{name_1[-1]}' does not match "
                f"signal file suffix '{name_2[-1]}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 2."
            )
            logger.error(msg)
            raise ValueError(msg)
        name = name_1[-1]

        tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
        sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]
        pair_name_to_tsNew[name] = tsNew
        pair_name_to_sampling_rate[name] = sampling_rate
    return pair_name_to_tsNew, pair_name_to_sampling_rate


def read_coords_pairwise(filepath, pair_name_to_tsNew):
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
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        if name_1[-1] != name_2[-1]:
            msg = (
                f"Pair name mismatch in '{filepath}': control file suffix '{name_1[-1]}' does not match "
                f"signal file suffix '{name_2[-1]}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 2."
            )
            logger.error(msg)
            raise ValueError(msg)
        pair_name = name_1[-1]

        tsNew = pair_name_to_tsNew[pair_name]
        coords = fetchCoords(filepath, pair_name, tsNew)
        pair_name_to_coords[pair_name] = coords
    return pair_name_to_coords


def read_corrected_data_dict(filepath, storesList):  # TODO: coordinate with read_corrected_data
    """
    Load corrected control and signal data arrays into a flat dict.

    Parameters
    ----------
    filepath : str
        Session output directory.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    name_to_corrected_data : dict
        Display name → 1-D corrected data array.
    """
    name_to_corrected_data = {}
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    control_and_signal_names = get_control_and_signal_channel_names(storesList)

    for storename, name in zip(storenames, names_for_storenames):
        if name not in control_and_signal_names:
            continue
        data = read_hdf5(name, filepath, "data").reshape(-1)
        name_to_corrected_data[name] = data

    return name_to_corrected_data


def read_corrected_ttl_timestamps(filepath, storesList):
    """
    Load corrected TTL timestamps for all non-channel stores.

    Parameters
    ----------
    filepath : str
        Session output directory.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    compound_name_to_ttl_timestamps : dict
        Compound TTL name → TTL timestamp array.
    """
    compound_name_to_ttl_timestamps = {}
    storenames = storesList[0, :]
    names_for_storenames = storesList[1, :]
    arr = get_control_and_signal_channel_names(storesList)

    for storename, name in zip(storenames, names_for_storenames):
        if name in arr:
            continue
        ttl_name = name
        for i in range(arr.shape[1]):
            name_1 = arr[0, i].split("_")[-1]
            name_2 = arr[1, i].split("_")[-1]
            if name_1 != name_2:
                message = (
                    f"Pair name mismatch in storesList: control channel '{arr[0, i]}' has suffix "
                    f"'{name_1}' but signal channel '{arr[1, i]}' has suffix '{name_2}'. Check the "
                    "naming convention of your files and the storesList file, then re-run step 2."
                )
                logger.error(message)
                raise ValueError(message)
            compound_name = ttl_name + "_" + name_1
            ts = read_hdf5(compound_name, filepath, "ts")
            compound_name_to_ttl_timestamps[compound_name] = ts

    return compound_name_to_ttl_timestamps


def write_artifact_corrected_timestamps(filepath, pair_name_to_corrected_timestamps):
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
    filepath,
    name_to_corrected_data,
    pair_name_to_corrected_timestamps,
    compound_name_to_corrected_ttl_timestamps=None,
):
    """
    Write all artifact-removal outputs (data, TTL timestamps, and optionally timestamps) to HDF5.

    Parameters
    ----------
    filepath : str
        Session output directory.
    name_to_corrected_data : dict
        Display name → corrected data array.
    pair_name_to_corrected_timestamps : dict or None
        Pair name → corrected timestamp array; skipped when None.
    compound_name_to_corrected_ttl_timestamps : dict, optional
        Compound TTL name → corrected TTL timestamp array. Default is None.
    """
    write_corrected_data(filepath, name_to_corrected_data)
    write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)
    if pair_name_to_corrected_timestamps is not None:
        write_artifact_corrected_timestamps(filepath, pair_name_to_corrected_timestamps)


def read_timestamps_for_combining_data(filepaths_to_combine):
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
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            msg = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file suffix '{name_1}' does not match "
                f"signal file suffix '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 2."
            )
            logger.error(msg)
            raise ValueError(msg)
        pair_name = name_1
        pair_name_to_filepath_to_timestamps[pair_name] = {}
        for filepath in filepaths_to_combine:
            tsNew = read_hdf5("timeCorrection_" + pair_name, filepath, "timestampNew")
            pair_name_to_filepath_to_timestamps[pair_name][filepath] = tsNew

    return pair_name_to_filepath_to_timestamps


def read_data_for_combining_data(filepaths_to_combine, storesList):
    """
    Load corrected channel data from all session files for the combine-data step.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    display_name_to_filepath_to_data : dict
        ``{display_name: {filepath: data_array}}``.
    """
    names_for_storenames = storesList[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    display_name_to_filepath_to_data: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            msg = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file suffix '{name_1}' does not match "
                f"signal file suffix '{name_2}'. Check the naming convention of your files and the "
                f"storesList file, then re-run step 2."
            )
            logger.error(msg)
            raise ValueError(msg)
        pair_name = name_1
        for i in range(len(names_for_storenames)):
            if not (
                "control_" + pair_name.lower() in names_for_storenames[i].lower()
                or "signal_" + pair_name.lower() in names_for_storenames[i].lower()
            ):
                continue
            display_name = names_for_storenames[i]
            display_name_to_filepath_to_data[display_name] = {}
            for filepath in filepaths_to_combine:
                data = read_hdf5(display_name, filepath, "data").reshape(-1)
                display_name_to_filepath_to_data[display_name][filepath] = data

    return display_name_to_filepath_to_data


def read_ttl_timestamps_for_combining_data(filepaths_to_combine, storesList):
    """
    Load corrected TTL timestamps from all session files for the combine-data step.

    Parameters
    ----------
    filepaths_to_combine : list of str
        Ordered list of session output directories.
    storesList : np.ndarray
        2-D array with rows [storenames, display_names].

    Returns
    -------
    compound_name_to_filepath_to_ttl_timestamps : dict
        ``{compound_name: {filepath: ttl_timestamps_array}}``.
    """
    names_for_storenames = storesList[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    compound_name_to_filepath_to_ttl_timestamps: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            message = (
                f"Pair name mismatch in '{filepaths_to_combine[0]}': control file suffix '{name_1}' does "
                f"not match signal file suffix '{name_2}'. Check the naming convention of your files and "
                "the storesList file, then re-run step 2."
            )
            logger.error(message)
            raise ValueError(message)
        pair_name = name_1
        for i in range(len(names_for_storenames)):
            if (
                "control_" + pair_name.lower() in names_for_storenames[i].lower()
                or "signal_" + pair_name.lower() in names_for_storenames[i].lower()
            ):
                continue
            compound_name = names_for_storenames[i] + "_" + pair_name
            compound_name_to_filepath_to_ttl_timestamps[compound_name] = {}
            for filepath in filepaths_to_combine:
                if os.path.exists(os.path.join(filepath, names_for_storenames[i] + "_" + pair_name + ".hdf5")):
                    ts = read_hdf5(names_for_storenames[i] + "_" + pair_name, filepath, "ts").reshape(-1)
                else:
                    ts = np.array([])
                compound_name_to_filepath_to_ttl_timestamps[compound_name][filepath] = ts

    return compound_name_to_filepath_to_ttl_timestamps


def write_combined_data(output_filepath, pair_name_to_tsNew, display_name_to_data, compound_name_to_ttl_timestamps):
    """
    Write combined multi-session data (timestamps, channel data, TTLs) to HDF5.

    Parameters
    ----------
    output_filepath : str
        Destination session output directory.
    pair_name_to_tsNew : dict
        Pair name → combined timestamp array.
    display_name_to_data : dict
        Display name → combined data array.
    compound_name_to_ttl_timestamps : dict
        Compound TTL name → combined TTL timestamp array.
    """
    for pair_name, tsNew in pair_name_to_tsNew.items():
        write_hdf5(tsNew, "timeCorrection_" + pair_name, output_filepath, "timestampNew")
    for display_name, data in display_name_to_data.items():
        write_hdf5(data, display_name, output_filepath, "data")
    for compound_name, ts in compound_name_to_ttl_timestamps.items():
        write_hdf5(ts, compound_name, output_filepath, "ts")


def write_peak_and_area_to_hdf5(filepath, arr, name, index=[]):
    """
    Save peak and area-under-curve metrics to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory.
    arr : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``peak_AUC_<name>.h5``.
    index : list, optional
        Row index labels. Default is an empty list.
    """

    op = os.path.join(filepath, "peak_AUC_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index)

    df.to_hdf(op, key="df", mode="w")


def write_peak_and_area_to_csv(filepath, arr, name, index=[]):
    """
    Save peak and area-under-curve metrics to a CSV file.

    Parameters
    ----------
    filepath : str
        Output directory.
    arr : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``peak_AUC_<name>.csv``.
    index : list, optional
        Row index labels. Default is an empty list.
    """
    op = os.path.join(filepath, "peak_AUC_" + name + ".csv")
    df = pd.DataFrame(arr, index=index)

    df.to_csv(op)


def write_freq_and_amp_to_hdf5(filepath, arr, name, index=[], columns=[]):
    """
    Save transient frequency and amplitude metrics to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory.
    arr : array-like
        Metrics data to store in the DataFrame.
    name : str
        Filename stem; the file is written as ``freqAndAmp_<name>.h5``.
    index : list, optional
        Row index labels. Default is an empty list.
    columns : list, optional
        Column labels. Default is an empty list.
    """

    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index, columns=columns)

    df.to_hdf(op, key="df", mode="w")


def write_freq_and_amp_to_csv(filepath, arr, name, index=[], columns=[]):
    """
    Save transient frequency and amplitude metrics to a CSV file.

    Parameters
    ----------
    filepath : str
        Output directory.
    arr : array-like
        Metrics data to store in the DataFrame.
    name : str
        Output filename (written directly inside ``filepath``).
    index : list, optional
        Row index labels. Default is an empty list.
    columns : list, optional
        Column labels. Default is an empty list.
    """
    op = os.path.join(filepath, name)
    df = pd.DataFrame(arr, index=index, columns=columns)
    df.to_csv(op)


def read_freq_and_amp_from_hdf5(filepath, name):
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
    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df


def write_transients_to_hdf5(filepath, name, z_score, ts, peaksInd):
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
    ts : np.ndarray
        Timestamp array corresponding to ``z_score``.
    peaksInd : np.ndarray
        Integer indices of detected transient peaks in ``z_score``.
    """
    event = f"transient_outputs_{name}"
    write_hdf5(z_score, event, filepath, "z_score")
    write_hdf5(ts, event, filepath, "timestamps")
    write_hdf5(peaksInd, event, filepath, "peaksInd")


def read_transients_from_hdf5(filepath, name):
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
    ts : np.ndarray
        Timestamp array.
    peaksInd : np.ndarray
        Integer indices of detected transient peaks.
    """
    event = f"transient_outputs_{name}"
    z_score = read_hdf5(event, filepath, "z_score")
    ts = read_hdf5(event, filepath, "timestamps")
    peaksInd = read_hdf5(event, filepath, "peaksInd")
    return z_score, ts, peaksInd
