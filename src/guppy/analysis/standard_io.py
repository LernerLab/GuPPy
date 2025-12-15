import logging

import numpy as np

from .io_utils import (
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
        idx_c = np.where(storesList == control_name)[0]
        idx_s = np.where(storesList == signal_name)[0]
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
