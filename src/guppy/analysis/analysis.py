import logging

import numpy as np
from scipy import signal as ss
from scipy.optimize import curve_fit

from .io_utils import fetchCoords, read_hdf5

logger = logging.getLogger(__name__)


# Category: Analysis
# Reason: Pure mathematical function for exponential curve fitting - no dependencies, pure computation
# curve fit exponential function
def curveFitFn(x, a, b, c):
    return a + (b * np.exp(-(1 / c) * x))


# Category: Analysis
# Reason: Pure algorithmic function - applies Savitzky-Golay filter and curve fitting to generate synthetic control channel
# helper function to create control channel using signal channel
# by curve fitting signal channel to exponential function
# when there is no isosbestic control channel is present
def helper_create_control_channel(signal, timestamps, window):
    # check if window is greater than signal shape
    if window > signal.shape[0]:
        window = ((signal.shape[0] + 1) / 2) + 1
        if window % 2 != 0:
            window = window
        else:
            window = window + 1

    filtered_signal = ss.savgol_filter(signal, window_length=window, polyorder=3)

    p0 = [5, 50, 60]

    try:
        popt, pcov = curve_fit(curveFitFn, timestamps, filtered_signal, p0)
    except Exception as e:
        logger.error(str(e))

    # logger.info('Curve Fit Parameters : ', popt)
    control = curveFitFn(timestamps, *popt)

    return control


# Category: Analysis
# Reason: Data validation function - compares array lengths and returns indices for processing
# function to check control and signal channel has same length
# if not, take a smaller length and do pre-processing
def check_cntrl_sig_length(filepath, channels_arr, storenames, storesList):

    indices = []
    for i in range(channels_arr.shape[1]):
        idx_c = np.where(storesList == channels_arr[0, i])[0]
        idx_s = np.where(storesList == channels_arr[1, i])[0]
        control = read_hdf5(storenames[idx_c[0]], filepath, "data")
        signal = read_hdf5(storenames[idx_s[0]], filepath, "data")
        if control.shape[0] < signal.shape[0]:
            indices.append(storesList[idx_c[0]])
        elif control.shape[0] > signal.shape[0]:
            indices.append(storesList[idx_s[0]])
        else:
            indices.append(storesList[idx_s[0]])

    return indices


# Category: Analysis
# Reason: Pure algorithmic function - concatenates data chunks based on coordinate boundaries, adjusts timestamps mathematically
# helper function to process control and signal timestamps
def eliminateData(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    ts = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    data = read_hdf5(event, filepath, "data").reshape(-1)
    coords = fetchCoords(filepath, naming, ts)

    if (data == 0).all() == True:
        data = np.zeros(ts.shape[0])

    arr = np.array([])
    ts_arr = np.array([])
    for i in range(coords.shape[0]):

        index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]

        if len(arr) == 0:
            arr = np.concatenate((arr, data[index]))
            sub = ts[index][0] - timeForLightsTurnOn
            new_ts = ts[index] - sub
            ts_arr = np.concatenate((ts_arr, new_ts))
        else:
            temp = data[index]
            # new = temp + (arr[-1]-temp[0])
            temp_ts = ts[index]
            new_ts = temp_ts - (temp_ts[0] - ts_arr[-1])
            arr = np.concatenate((arr, temp))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    # logger.info(arr.shape, ts_arr.shape)
    return arr, ts_arr


# Category: Analysis
# Reason: Pure algorithmic function - processes event timestamps based on coordinate boundaries, aligns with data timeline
# helper function to align event timestamps with the control and signal timestamps
def eliminateTs(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

    tsNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    ts = read_hdf5(event + "_" + naming, filepath, "ts").reshape(-1)
    coords = fetchCoords(filepath, naming, tsNew)

    ts_arr = np.array([])
    tsNew_arr = np.array([])
    for i in range(coords.shape[0]):
        tsNew_index = np.where((tsNew > coords[i, 0]) & (tsNew < coords[i, 1]))[0]
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]

        if len(tsNew_arr) == 0:
            sub = tsNew[tsNew_index][0] - timeForLightsTurnOn
            tsNew_arr = np.concatenate((tsNew_arr, tsNew[tsNew_index] - sub))
            ts_arr = np.concatenate((ts_arr, ts[ts_index] - sub))
        else:
            temp_tsNew = tsNew[tsNew_index]
            temp_ts = ts[ts_index]
            new_ts = temp_ts - (temp_tsNew[0] - tsNew_arr[-1])
            new_tsNew = temp_tsNew - (temp_tsNew[0] - tsNew_arr[-1])
            tsNew_arr = np.concatenate((tsNew_arr, new_tsNew + (1 / sampling_rate)))
            ts_arr = np.concatenate((ts_arr, new_ts + (1 / sampling_rate)))

    return ts_arr


# Category: Analysis
# Reason: Pure algorithmic function - replaces specified data chunks with NaN based on coordinate boundaries
# adding nan values to removed chunks
# when using artifacts removal method - replace with NaN
def addingNaNValues(filepath, event, naming):

    ts = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    data = read_hdf5(event, filepath, "data").reshape(-1)
    coords = fetchCoords(filepath, naming, ts)

    if (data == 0).all() == True:
        data = np.zeros(ts.shape[0])

    arr = np.array([])
    ts_index = np.arange(ts.shape[0])
    for i in range(coords.shape[0]):

        index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        arr = np.concatenate((arr, index))

    nan_indices = list(set(ts_index).symmetric_difference(arr))
    data[nan_indices] = np.nan

    return data


# Category: Analysis
# Reason: Pure algorithmic function - filters event timestamps to exclude artifact regions based on coordinates
# remove event TTLs which falls in the removed chunks
# when using artifacts removal method - replace with NaN
def removeTTLs(filepath, event, naming):
    tsNew = read_hdf5("timeCorrection_" + naming, filepath, "timestampNew")
    ts = read_hdf5(event + "_" + naming, filepath, "ts").reshape(-1)
    coords = fetchCoords(filepath, naming, tsNew)

    ts_arr = np.array([])
    for i in range(coords.shape[0]):
        ts_index = np.where((ts > coords[i, 0]) & (ts < coords[i, 1]))[0]
        ts_arr = np.concatenate((ts_arr, ts[ts_index]))

    return ts_arr


# Category: Analysis
# Reason: Pure mathematical computation - calculates deltaF/F from signal and control arrays with simple formula
# function to compute deltaF/F using fitted control channel and filtered signal channel
def deltaFF(signal, control):

    res = np.subtract(signal, control)
    normData = np.divide(res, control)
    # deltaFF = normData
    normData = normData * 100

    return normData


# Category: Analysis
# Reason: Pure algorithmic function - performs polynomial linear regression to fit control to signal
# function to fit control channel to signal channel
def controlFit(control, signal):

    p = np.polyfit(control, signal, 1)
    arr = (p[0] * control) + p[1]
    return arr


# Category: Analysis
# Reason: Pure signal processing function - applies moving average filter using scipy filtfilt
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


# Category: Routing
# Reason: Orchestrates signal processing pipeline - sequences calls to filterSignal, controlFit, and deltaFF with conditional logic
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


# Category: Analysis
# Reason: Pure statistical computation - calculates z-score from deltaF/F using different methods (standard, baseline, robust)
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
