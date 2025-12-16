import logging
import os

import numpy as np
from scipy import signal as ss

from .control_channel import helper_create_control_channel
from .io_utils import (
    fetchCoords,
    find_files,
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


# compute z-score and deltaF/F and save it to hdf5 file
def compute_z_score(filepath, inputParameters):

    logger.debug(f"Computing z-score for each of the data in {filepath}")

    path_1 = find_files(filepath, "control_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'control*'))
    path_2 = find_files(filepath, "signal_*", ignore_case=True)  # glob.glob(os.path.join(filepath, 'signal*'))

    path = sorted(path_1 + path_2, key=str.casefold)

    if len(path) % 2 != 0:
        logger.error("There are not equal number of Control and Signal data")
        raise Exception("There are not equal number of Control and Signal data")

    path = np.asarray(path).reshape(2, -1)
    removeArtifacts = inputParameters["removeArtifacts"]

    for i in range(path.shape[1]):
        name_1 = ((os.path.basename(path[0, i])).split(".")[0]).split("_")
        name_2 = ((os.path.basename(path[1, i])).split(".")[0]).split("_")

        if name_1[-1] == name_2[-1]:
            name = name_1[-1]
            control = read_hdf5("", path[0, i], "data").reshape(-1)
            signal = read_hdf5("", path[1, i], "data").reshape(-1)
            tsNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")

            coords = get_coords(filepath, name, tsNew, removeArtifacts)
            z_score, dff, control_fit, temp_control_arr = helper_z_score(
                control, signal, tsNew, filepath, name, inputParameters, coords
            )

            write_hdf5(z_score, "z_score_" + name, filepath, "data")
            write_hdf5(dff, "dff_" + name, filepath, "data")
            write_hdf5(control_fit, "cntrl_sig_fit_" + name, filepath, "data")
            if temp_control_arr is not None:
                write_hdf5(temp_control_arr, "control_" + name, filepath, "data")
        else:
            logger.error("Error in naming convention of files or Error in storesList file")
            raise Exception("Error in naming convention of files or Error in storesList file")

    logger.info(f"z-score for the data in {filepath} computed.")


def get_coords(filepath, name, tsNew, removeArtifacts):  # TODO: Make less redundant with fetchCoords
    if removeArtifacts == True:
        coords = fetchCoords(filepath, name, tsNew)
    else:
        dt = tsNew[1] - tsNew[0]
        coords = np.array([[tsNew[0] - dt, tsNew[-1] + dt]])
    return coords


# helper function to compute z-score and deltaF/F
def helper_z_score(
    control, signal, tsNew, filepath, name, inputParameters, coords
):  # helper_z_score(control_smooth, signal_smooth):

    artifactsRemovalMethod = inputParameters["artifactsRemovalMethod"]
    filter_window = inputParameters["filter_window"]
    isosbestic_control = inputParameters["isosbestic_control"]

    if (control == 0).all() == True:
        control = np.zeros(tsNew.shape[0])

    z_score_arr = np.array([])
    norm_data_arr = np.full(tsNew.shape[0], np.nan)
    control_fit_arr = np.full(tsNew.shape[0], np.nan)
    temp_control_arr = np.full(tsNew.shape[0], np.nan)

    # for artifacts removal, each chunk which was selected by user is being processed individually and then
    # z-score is calculated
    for i in range(coords.shape[0]):
        tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        if isosbestic_control == False:
            control_arr = helper_create_control_channel(signal[tsNew_index], tsNew[tsNew_index], window=101)
            signal_arr = signal[tsNew_index]
            norm_data, control_fit = execute_controlFit_dff(control_arr, signal_arr, isosbestic_control, filter_window)
            temp_control_arr[tsNew_index] = control_arr
            if i < coords.shape[0] - 1:
                blank_index = np.where((tsNew > coords[i, 1]) & (tsNew < coords[i + 1, 0]))[0]
                temp_control_arr[blank_index] = np.full(blank_index.shape[0], np.nan)
        else:
            control_arr = control[tsNew_index]
            signal_arr = signal[tsNew_index]
            norm_data, control_fit = execute_controlFit_dff(control_arr, signal_arr, isosbestic_control, filter_window)
        norm_data_arr[tsNew_index] = norm_data
        control_fit_arr[tsNew_index] = control_fit

    if artifactsRemovalMethod == "concatenate":
        norm_data_arr = norm_data_arr[~np.isnan(norm_data_arr)]
        control_fit_arr = control_fit_arr[~np.isnan(control_fit_arr)]
    z_score = z_score_computation(norm_data_arr, tsNew, inputParameters)
    z_score_arr = np.concatenate((z_score_arr, z_score))

    # handle the case if there are chunks being cut in the front and the end
    if isosbestic_control == False:
        coords = coords.flatten()
        # front chunk
        idx = np.where((tsNew >= tsNew[0]) & (tsNew < coords[0]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
        # end chunk
        idx = np.where((tsNew > coords[-1]) & (tsNew <= tsNew[-1]))[0]
        temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
    else:
        temp_control_arr = None

    return z_score_arr, norm_data_arr, control_fit_arr, temp_control_arr


# function to filter control and signal channel, also execute above two function : controlFit and deltaFF
# function will also take care if there is only signal channel and no control channel
# if there is only signal channel, z-score will be computed using just signal channel
def execute_controlFit_dff(control, signal, isosbestic_control, filter_window):

    if isosbestic_control == False:
        signal_smooth = filterSignal(filter_window, signal)  # ss.filtfilt(b, a, signal)
        control_fit = controlFit(control, signal_smooth)
        norm_data = deltaFF(signal_smooth, control_fit)
    else:
        control_smooth = filterSignal(filter_window, control)  # ss.filtfilt(b, a, control)
        signal_smooth = filterSignal(filter_window, signal)  # ss.filtfilt(b, a, signal)
        control_fit = controlFit(control_smooth, signal_smooth)
        norm_data = deltaFF(signal_smooth, control_fit)

    return norm_data, control_fit


# function to compute deltaF/F using fitted control channel and filtered signal channel
def deltaFF(signal, control):

    res = np.subtract(signal, control)
    normData = np.divide(res, control)
    # deltaFF = normData
    normData = normData * 100

    return normData


# function to fit control channel to signal channel
def controlFit(control, signal):

    p = np.polyfit(control, signal, 1)
    arr = (p[0] * control) + p[1]
    return arr


def filterSignal(filter_window, signal):
    if filter_window == 0:
        return signal
    elif filter_window > 1:
        b = np.divide(np.ones((filter_window,)), filter_window)
        a = 1
        filtered_signal = ss.filtfilt(b, a, signal)
        return filtered_signal
    else:
        raise Exception("Moving average filter window value is not correct.")


# function to compute z-score based on z-score computation method
def z_score_computation(dff, timestamps, inputParameters):

    zscore_method = inputParameters["zscore_method"]
    baseline_start, baseline_end = inputParameters["baselineWindowStart"], inputParameters["baselineWindowEnd"]

    if zscore_method == "standard z-score":
        numerator = np.subtract(dff, np.nanmean(dff))
        zscore = np.divide(numerator, np.nanstd(dff))
    elif zscore_method == "baseline z-score":
        idx = np.where((timestamps > baseline_start) & (timestamps < baseline_end))[0]
        if idx.shape[0] == 0:
            logger.error(
                "Baseline Window Parameters for baseline z-score computation zscore_method \
							are not correct."
            )
            raise Exception(
                "Baseline Window Parameters for baseline z-score computation zscore_method \
							are not correct."
            )
        else:
            baseline_mean = np.nanmean(dff[idx])
            baseline_std = np.nanstd(dff[idx])
            numerator = np.subtract(dff, baseline_mean)
            zscore = np.divide(numerator, baseline_std)
    else:
        median = np.median(dff)
        mad = np.median(np.abs(dff - median))
        numerator = 0.6745 * (dff - median)
        zscore = np.divide(numerator, mad)

    return zscore
