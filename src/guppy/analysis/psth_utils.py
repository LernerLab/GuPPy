import logging
import math
import os
import re

import numpy as np
import pandas as pd

from .io_utils import read_hdf5

logger = logging.getLogger(__name__)


# function to create dataframe for each event PSTH and save it to h5 file
def create_Df(filepath, event, name, psth, columns=[]):
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
