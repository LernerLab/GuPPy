import logging
import os
import shutil

import numpy as np
import pandas as pd
from scipy import signal as ss
from scipy.optimize import curve_fit

from .io_utils import (
    read_hdf5,
    write_hdf5,
)

logger = logging.getLogger(__name__)


# This function just creates placeholder Control-HDF5 files that are then immediately overwritten later on in the pipeline.
# TODO: Refactor this function to avoid unnecessary file creation.
# function to add control channel when there is no
# isosbestic control channel and update the storeslist file
def add_control_channel(filepath, arr):

    storenames = arr[0, :]
    storesList = np.char.lower(arr[1, :])

    keep_control = np.array([])
    # check a case if there is isosbestic control channel present
    for i in range(storesList.shape[0]):
        if "control" in storesList[i].lower():
            name = storesList[i].split("_")[-1]
            new_str = "signal_" + str(name).lower()
            find_signal = [True for i in storesList if i == new_str]
            if len(find_signal) > 1:
                logger.error("Error in naming convention of files or Error in storesList file")
                raise Exception("Error in naming convention of files or Error in storesList file")
            if len(find_signal) == 0:
                logger.error(
                    "Isosbectic control channel parameter is set to False and still \
							 	 storeslist file shows there is control channel present"
                )
                raise Exception(
                    "Isosbectic control channel parameter is set to False and still \
							 	 storeslist file shows there is control channel present"
                )
        else:
            continue

    for i in range(storesList.shape[0]):
        if "signal" in storesList[i].lower():
            name = storesList[i].split("_")[-1]
            new_str = "control_" + str(name).lower()
            find_signal = [True for i in storesList if i == new_str]
            if len(find_signal) == 0:
                src, dst = os.path.join(filepath, arr[0, i] + ".hdf5"), os.path.join(
                    filepath, "cntrl" + str(i) + ".hdf5"
                )
                shutil.copyfile(src, dst)
                arr = np.concatenate((arr, [["cntrl" + str(i)], ["control_" + str(arr[1, i].split("_")[-1])]]), axis=1)

    np.savetxt(os.path.join(filepath, "storesList.csv"), arr, delimiter=",", fmt="%s")

    return arr


# main function to create control channel using
# signal channel and save it to a file
def create_control_channel(filepath, arr, window=5001):

    storenames = arr[0, :]
    storesList = arr[1, :]

    for i in range(storesList.shape[0]):
        event_name, event = storesList[i], storenames[i]
        if "control" in event_name.lower() and "cntrl" in event.lower():
            logger.debug("Creating control channel from signal channel using curve-fitting")
            name = event_name.split("_")[-1]
            signal = read_hdf5("signal_" + name, filepath, "data")
            timestampNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
            sampling_rate = np.full(timestampNew.shape, np.nan)
            sampling_rate[0] = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            control = helper_create_control_channel(signal, timestampNew, window)

            write_hdf5(control, event_name, filepath, "data")
            d = {"timestamps": timestampNew, "data": control, "sampling_rate": sampling_rate}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(os.path.dirname(filepath), event.lower() + ".csv"), index=False)
            logger.info("Control channel from signal channel created using curve-fitting")


# TODO: figure out why a control channel is created for both timestamp correction and z-score steps.
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


# curve fit exponential function
def curveFitFn(x, a, b, c):
    return a + (b * np.exp(-(1 / c) * x))
