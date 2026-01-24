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
    for name, data in name_to_corrected_data.items():
        write_hdf5(data, name, filepath, "data")


def write_corrected_ttl_timestamps(
    filepath,
    compound_name_to_corrected_ttl_timestamps,
):
    logger.debug("Applying correction of timestamps to the data and event timestamps")
    for compound_name, corrected_ttl_timestamps in compound_name_to_corrected_ttl_timestamps.items():
        write_hdf5(corrected_ttl_timestamps, compound_name, filepath, "ts")
    logger.info("Timestamps corrections applied to the data and event timestamps.")


def read_corrected_data(control_path, signal_path, filepath, name):
    control = read_hdf5("", control_path, "data").reshape(-1)
    signal = read_hdf5("", signal_path, "data").reshape(-1)
    tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")

    return control, signal, tsNew


def write_zscore(filepath, name, z_score, dff, control_fit, temp_control_arr):
    write_hdf5(z_score, "z_score_" + name, filepath, "data")
    write_hdf5(dff, "dff_" + name, filepath, "data")
    write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
    if temp_control_arr is not None:
        write_hdf5(temp_control_arr, "control_" + name, filepath, "data")


def read_corrected_timestamps_pairwise(filepath):
    pair_name_to_tsNew = {}
    pair_name_to_sampling_rate = {}
    path = decide_naming_convention(filepath)
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        if name_1[-1] != name_2[-1]:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
        name = name_1[-1]

        tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
        sampling_rate = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]
        pair_name_to_tsNew[name] = tsNew
        pair_name_to_sampling_rate[name] = sampling_rate
    return pair_name_to_tsNew, pair_name_to_sampling_rate


def read_coords_pairwise(filepath, pair_name_to_tsNew):
    pair_name_to_coords = {}
    path = decide_naming_convention(filepath)
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")
        if name_1[-1] != name_2[-1]:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
        pair_name = name_1[-1]

        tsNew = pair_name_to_tsNew[pair_name]
        coords = fetchCoords(filepath, pair_name, tsNew)
        pair_name_to_coords[pair_name] = coords
    return pair_name_to_coords


def read_corrected_data_dict(filepath, storesList):  # TODO: coordinate with read_corrected_data
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
                logger.error("Error in naming convention of files or Error in storesList file")
                raise Exception("Error in naming convention of files or Error in storesList file")
            compound_name = ttl_name + "_" + name_1
            ts = read_hdf5(compound_name, filepath, "ts")
            compound_name_to_ttl_timestamps[compound_name] = ts

    return compound_name_to_ttl_timestamps


def write_artifact_corrected_timestamps(filepath, pair_name_to_corrected_timestamps):
    for pair_name, timestamps in pair_name_to_corrected_timestamps.items():
        write_hdf5(timestamps, "timeCorrection_" + pair_name, filepath, "timestampNew")


def write_artifact_removal(
    filepath,
    name_to_corrected_data,
    pair_name_to_corrected_timestamps,
    compound_name_to_corrected_ttl_timestamps=None,
):
    write_corrected_data(filepath, name_to_corrected_data)
    write_corrected_ttl_timestamps(filepath, compound_name_to_corrected_ttl_timestamps)
    if pair_name_to_corrected_timestamps is not None:
        write_artifact_corrected_timestamps(filepath, pair_name_to_corrected_timestamps)


def read_timestamps_for_combining_data(filepaths_to_combine):
    path = decide_naming_convention(filepaths_to_combine[0])
    pair_name_to_filepath_to_timestamps: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
        pair_name = name_1
        pair_name_to_filepath_to_timestamps[pair_name] = {}
        for filepath in filepaths_to_combine:
            tsNew = read_hdf5("timeCorrection_" + pair_name, filepath, "timestampNew")
            pair_name_to_filepath_to_timestamps[pair_name][filepath] = tsNew

    return pair_name_to_filepath_to_timestamps


def read_data_for_combining_data(filepaths_to_combine, storesList):
    names_for_storenames = storesList[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    display_name_to_filepath_to_data: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
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
    names_for_storenames = storesList[1, :]
    path = decide_naming_convention(filepaths_to_combine[0])
    compound_name_to_filepath_to_ttl_timestamps: dict[str, dict[str, np.ndarray]] = {}
    for j in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, j])).split(".")[0]).split("_")[-1]
        name_2 = ((os.path.basename(path[1, j])).split(".")[0]).split("_")[-1]
        if name_1 != name_2:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")
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
    for pair_name, tsNew in pair_name_to_tsNew.items():
        write_hdf5(tsNew, "timeCorrection_" + pair_name, output_filepath, "timestampNew")
    for display_name, data in display_name_to_data.items():
        write_hdf5(data, display_name, output_filepath, "data")
    for compound_name, ts in compound_name_to_ttl_timestamps.items():
        write_hdf5(ts, compound_name, output_filepath, "ts")


def write_peak_and_area_to_hdf5(filepath, arr, name, index=[]):

    op = os.path.join(filepath, "peak_AUC_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index)

    df.to_hdf(op, key="df", mode="w")


def write_peak_and_area_to_csv(filepath, arr, name, index=[]):
    op = os.path.join(filepath, "peak_AUC_" + name + ".csv")
    df = pd.DataFrame(arr, index=index)

    df.to_csv(op)


def write_freq_and_amp_to_hdf5(filepath, arr, name, index=[], columns=[]):

    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    dirname = os.path.dirname(filepath)

    df = pd.DataFrame(arr, index=index, columns=columns)

    df.to_hdf(op, key="df", mode="w")


def write_freq_and_amp_to_csv(filepath, arr, name, index=[], columns=[]):
    op = os.path.join(filepath, name)
    df = pd.DataFrame(arr, index=index, columns=columns)
    df.to_csv(op)


def read_freq_and_amp_from_hdf5(filepath, name):
    op = os.path.join(filepath, "freqAndAmp_" + name + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df
