import logging
import os
import shutil

import numpy as np
import pandas as pd
from scipy import signal as ss
from scipy.optimize import curve_fit

from .io_utils import (
    read_hdf5,
    region_from_channel_label,
    write_hdf5,
)

logger = logging.getLogger(__name__)


# This function just creates placeholder Control-HDF5 files that are then immediately overwritten later on in the pipeline.
# TODO: Refactor this function to avoid unnecessary file creation.
def add_control_channel(filepath: str, store_array: np.ndarray) -> np.ndarray:
    """
    Add synthetic control-channel entries to the storesList when no isosbestic control exists.

    Parameters
    ----------
    filepath : str
        Path to the session output directory containing the storesList CSV.
    store_array : np.ndarray
        2-D store array with rows [store_id, store_label].

    Returns
    -------
    store_array : np.ndarray
        Updated store array with synthetic control entries appended.
    """

    store_ids = store_array[0, :]
    store_labels_lower = np.char.lower(store_array[1, :])

    keep_control = np.array([])
    # check a case if there is isosbestic control channel present
    for i in range(store_labels_lower.shape[0]):
        if "control" in store_labels_lower[i].lower():
            name = region_from_channel_label(store_labels_lower[i])
            expected_signal_name = "signal_" + str(name).lower()
            find_signal = [True for i in store_labels_lower if i == expected_signal_name]
            if len(find_signal) > 1:
                message = (
                    f"Multiple signal channels named '{expected_signal_name}' found in storesList for control "
                    f"channel '{store_labels_lower[i]}' in '{filepath}'. Each signal name must be unique; "
                    "check the storesList file and re-run step 1."
                )
                logger.error(message)
                raise ValueError(message)
            if len(find_signal) == 0:
                message = (
                    "Isosbestic control channel parameter is set to False, but the storesList file "
                    f"in '{filepath}' contains a control channel '{store_labels_lower[i]}' with no matching "
                    f"signal channel '{expected_signal_name}'. Either enable isosbestic control or re-run step 1 "
                    "to remove the unmatched control entry."
                )
                logger.error(message)
                raise ValueError(message)
        else:
            continue

    for i in range(store_labels_lower.shape[0]):
        if "signal" in store_labels_lower[i].lower():
            name = region_from_channel_label(store_labels_lower[i])
            expected_control_name = "control_" + str(name).lower()
            find_signal = [True for i in store_labels_lower if i == expected_control_name]
            if len(find_signal) == 0:
                source_path, destination_path = os.path.join(filepath, store_array[0, i] + ".hdf5"), os.path.join(
                    filepath, "cntrl" + str(i) + ".hdf5"
                )
                shutil.copyfile(source_path, destination_path)
                store_array = np.concatenate(
                    (
                        store_array,
                        [["cntrl" + str(i)], ["control_" + region_from_channel_label(store_array[1, i])]],
                    ),
                    axis=1,
                )

    np.savetxt(os.path.join(filepath, "storesList.csv"), store_array, delimiter=",", fmt="%s")

    return store_array


def create_control_channel(filepath: str, store_array: np.ndarray, window: int = 5001) -> None:
    """
    Fit a synthetic control channel from the signal channel and save it.

    Parameters
    ----------
    filepath : str
        Path to the session output directory where HDF5 and CSV files are written.
    store_array : np.ndarray
        2-D store array with rows [store_id, store_label].
    window : int, optional
        Savitzky-Golay filter window length used for initial smoothing. Default is 5001.
    """

    store_ids = store_array[0, :]
    store_labels = store_array[1, :]

    for i in range(store_labels.shape[0]):
        event_name, event = store_labels[i], store_ids[i]
        if "control" in event_name.lower() and "cntrl" in event.lower():
            logger.debug("Creating control channel from signal channel using curve-fitting")
            name = region_from_channel_label(event_name)
            signal = read_hdf5("signal_" + name, filepath, "data")
            timestampNew = read_hdf5("timeCorrection_" + name, filepath, "timestampNew")
            sampling_rate = np.full(timestampNew.shape, np.nan)
            sampling_rate[0] = read_hdf5("timeCorrection_" + name, filepath, "sampling_rate")[0]

            control = helper_create_control_channel(signal, timestampNew, window)

            write_hdf5(control, event_name, filepath, "data")
            data_dict = {"timestamps": timestampNew, "data": control, "sampling_rate": sampling_rate}
            df = pd.DataFrame(data_dict)
            df.to_csv(os.path.join(os.path.dirname(filepath), event.lower() + ".csv"), index=False)
            logger.info("Control channel from signal channel created using curve-fitting")


# TODO: figure out why a control channel is created for both timestamp correction and z-score steps.
def helper_create_control_channel(signal: np.ndarray, timestamps: np.ndarray, window: int) -> np.ndarray:
    """
    Fit an exponential control channel to the signal using curve fitting.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array.
    timestamps : np.ndarray
        1-D timestamp array corresponding to ``signal``.
    window : int
        Savitzky-Golay filter window length for pre-smoothing before curve fitting.

    Returns
    -------
    control : np.ndarray
        Fitted exponential control channel array.
    """
    # check if window is greater than signal shape
    if window > signal.shape[0]:
        window = ((signal.shape[0] + 1) // 2) + 1
        if window % 2 != 0:
            window = window
        else:
            window = window + 1

    filtered_signal = ss.savgol_filter(signal, window_length=window, polyorder=3)

    p0 = [5, 50, 60]

    try:
        popt, pcov = curve_fit(curveFitFn, timestamps, filtered_signal, p0)
    except Exception as error:
        logger.error(str(error))

    control = curveFitFn(timestamps, *popt)

    return control


def curveFitFn(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Evaluate the exponential model ``a + b * exp(-x / c)``.

    Parameters
    ----------
    x : array-like
        Independent variable (timestamps).
    a : float
        Vertical offset.
    b : float
        Amplitude.
    c : float
        Decay constant.

    Returns
    -------
    np.ndarray
        Evaluated exponential values.
    """
    return a + (b * np.exp(-(1 / c) * x))
