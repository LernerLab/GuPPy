import glob
import logging
import math
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_Df_for_psth(filepath: str, event: str, name: str, psth: np.ndarray, columns: list[object]) -> None:
    """
    Build a PSTH DataFrame (with mean/error columns) and save it as an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory where the ``.h5`` file is written.
    event : str
        Event label; backslashes and forward-slashes are replaced with underscores.
    name : str
        Channel name suffix appended to the filename.
    psth : np.ndarray
        2-D PSTH matrix (trials × time-points).
    columns : list, optional
        Column labels for the trials axis. Default is an empty list.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")

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

    columns = np.asarray(columns)
    columns = np.append(columns, ["mean", "err"])
    df = pd.DataFrame(psth, index=None, columns=list(columns), dtype="float32")

    df.to_hdf(op, key="df", mode="w")


def create_Df_for_cross_correlation(
    filepath: str, event: str, name: str, psth: np.ndarray, columns: list[object]
) -> None:
    """
    Build a cross-correlation DataFrame (with mean/error columns) and save it as an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output directory where the ``.h5`` file is written.
    event : str
        Event label used to build the filename.
    name : str
        Channel name suffix appended to the filename.
    psth : np.ndarray
        2-D cross-correlation matrix (trials × lags).
    columns : list, optional
        Column labels for the trials axis. Default is an empty list.
    """
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")

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

    columns = np.asarray(columns)
    columns = np.append(columns, ["mean", "err"])
    df = pd.DataFrame(psth, index=None, columns=columns, dtype="float32")

    df.to_hdf(op, key="df", mode="w")


def getCorrCombinations(filepath: str, inputParameters: dict[str, object]) -> tuple[list[str], list[str]]:
    """
    Determine which channel pairs to cross-correlate in a session directory.

    Parameters
    ----------
    filepath : str
        Session output directory containing z-score or dff HDF5 files.
    inputParameters : dict
        Analysis configuration; must include ``'selectForComputePsth'``.

    Returns
    -------
    corr_info : list of str
        Ordered list of channel suffixes to correlate; circular when more than two.
    type : list of str
        Unique channel-type prefixes (e.g. ``['z_score']``).
    """
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
        return names, type
    elif len(names) == 2:
        corr_info = names
    else:
        corr_info = names
        corr_info.append(names[0])

    return corr_info, type
