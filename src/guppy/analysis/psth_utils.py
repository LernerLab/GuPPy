import glob
import logging
import math
import os
import re

import numpy as np
import pandas as pd

from .io_utils import read_hdf5

logger = logging.getLogger(__name__)


# function to create dataframe for each event PSTH and save it to h5 file
def create_Df_for_psth(filepath, event, name, psth, columns=[]):
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")

    # check if file already exists
    # if os.path.exists(op):
    # 	return 0

    # removing psth binned trials
    columns = np.array(columns, dtype="str")
    regex = re.compile("bin_*")
    single_trials = columns[[i for i in range(len(columns)) if not regex.match(columns[i])]]
    single_trials_index = [i for i in range(len(single_trials)) if single_trials[i] != "timestamps"]

    psth = psth.T
    if psth.ndim > 1:
        mean = np.nanmean(psth[:, single_trials_index], axis=1).reshape(-1, 1)
        err = np.nanstd(psth[:, single_trials_index], axis=1) / math.sqrt(psth[:, single_trials_index].shape[1])
        err = err.reshape(-1, 1)
        psth = np.hstack((psth, mean))
        psth = np.hstack((psth, err))
        # timestamps = np.asarray(read_Df(filepath, 'ts_psth', ''))
        # psth = np.hstack((psth, timestamps))
    try:
        ts = read_hdf5(event, filepath, "ts")
        ts = np.append(ts, ["mean", "err"])
    except:
        ts = None

    if len(columns) == 0:
        df = pd.DataFrame(psth, index=None, columns=ts, dtype="float32")
    else:
        columns = np.asarray(columns)
        columns = np.append(columns, ["mean", "err"])
        df = pd.DataFrame(psth, index=None, columns=list(columns), dtype="float32")

    df.to_hdf(op, key="df", mode="w")


# same function used to store PSTH in computePsth file
# Here, cross correlation dataframe is saved instead of PSTH
# cross correlation dataframe has the same structure as PSTH file
def create_Df_for_cross_correlation(filepath, event, name, psth, columns=[]):
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")

    # check if file already exists
    # if os.path.exists(op):
    # 	return 0

    # removing psth binned trials
    columns = list(np.array(columns, dtype="str"))
    regex = re.compile("bin_*")
    single_trials_index = [i for i in range(len(columns)) if not regex.match(columns[i])]
    single_trials_index = [i for i in range(len(columns)) if columns[i] != "timestamps"]

    psth = psth.T
    if psth.ndim > 1:
        mean = np.nanmean(psth[:, single_trials_index], axis=1).reshape(-1, 1)
        err = np.nanstd(psth[:, single_trials_index], axis=1) / math.sqrt(psth[:, single_trials_index].shape[1])
        err = err.reshape(-1, 1)
        psth = np.hstack((psth, mean))
        psth = np.hstack((psth, err))
        # timestamps = np.asarray(read_Df(filepath, 'ts_psth', ''))
        # psth = np.hstack((psth, timestamps))
    try:
        ts = read_hdf5(event, filepath, "ts")
        ts = np.append(ts, ["mean", "err"])
    except:
        ts = None

    if len(columns) == 0:
        df = pd.DataFrame(psth, index=None, columns=ts, dtype="float32")
    else:
        columns = np.asarray(columns)
        columns = np.append(columns, ["mean", "err"])
        df = pd.DataFrame(psth, index=None, columns=columns, dtype="float32")

    df.to_hdf(op, key="df", mode="w")


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
