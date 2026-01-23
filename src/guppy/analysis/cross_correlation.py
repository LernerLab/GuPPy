import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def compute_cross_correlation(arr_A, arr_B, sample_rate):
    cross_corr = list()
    for a, b in zip(arr_A, arr_B):
        if np.isnan(a).any() or np.isnan(b).any():
            corr = signal.correlate(a, b, method="direct")
        else:
            corr = signal.correlate(a, b)
        corr_norm = corr / np.max(np.abs(corr))
        cross_corr.append(corr_norm)
        lag = signal.correlation_lags(len(a), len(b))
        lag_msec = np.array(lag / sample_rate, dtype="float32")

    cross_corr_arr = np.array(cross_corr, dtype="float32")
    lag_msec = lag_msec.reshape(1, -1)
    cross_corr_arr = np.concatenate((cross_corr_arr, lag_msec), axis=0)
    return cross_corr_arr
