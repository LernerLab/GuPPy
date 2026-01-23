import glob
import logging
import os

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def getCorrCombinations(filepath, inputParameters):
    selectForComputePsth = inputParameters["selectForComputePsth"]
    if selectForComputePsth == "z_score":
        path = glob.glob(os.path.join(filepath, "z_score_*"))
    elif selectForComputePsth == "dff":
        path = glob.glob(os.path.join(filepath, "dff_*"))
    else:
        path = glob.glob(os.path.join(filepath, "z_score_*")) + glob.glob(os.path.join(filepath, "dff_*"))

    names = list()
    type = list()
    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split(".")[0]
        names.append(basename.split("_")[-1])
        type.append((os.path.basename(path[i])).split(".")[0].split("_" + names[-1], 1)[0])

    names = list(np.unique(np.array(names)))
    type = list(np.unique(np.array(type)))

    corr_info = list()
    if len(names) <= 1:
        logger.info("Cross-correlation cannot be computed because only one signal is present.")
        return corr_info, type
    elif len(names) == 2:
        corr_info = names
    else:
        corr_info = names
        corr_info.append(names[0])

    return corr_info, type


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
