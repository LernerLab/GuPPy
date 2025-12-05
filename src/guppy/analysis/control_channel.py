import logging

import numpy as np
from scipy import signal as ss
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


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


# Category: Analysis
# Reason: Pure mathematical function for exponential curve fitting - no dependencies, pure computation
# curve fit exponential function
def curveFitFn(x, a, b, c):
    return a + (b * np.exp(-(1 / c) * x))
