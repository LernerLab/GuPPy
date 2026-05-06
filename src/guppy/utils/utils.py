import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def takeOnlyDirs(paths):
    """Filter a list of paths to include only directories.

    Parameters
    ----------
    paths : list of str
        Mixed list of file and directory paths.

    Returns
    -------
    list of str
        Subset of ``paths`` containing only directories.
    """
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


def get_all_stores_for_combining_data(folderNames):
    """Group output directories by their output index for cross-session combining.

    Parameters
    ----------
    folderNames : list of str
        Paths to ``*_output_<n>`` directories across all sessions.

    Returns
    -------
    list of list of str
        One inner list per output index (1, 2, …), each containing all
        directories that share that index.
    """
    op = []
    for i in range(100):
        temp = []
        match = r"[\s\S]*" + "_output_" + str(i)
        for j in folderNames:
            temp.append(re.findall(match, j))
        temp = sorted(list(np.concatenate(temp).flatten()), key=str.casefold)
        if len(temp) > 0:
            op.append(temp)

    return op


def read_Df(filepath, event, name):
    """Read a PSTH HDF5 file and return it as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the session output directory.
    event : str
        Event name used in the filename.
    name : str
        z-score/dff basename; when non-empty the filename is
        ``<event>_<name>.h5``, otherwise ``<event>.h5``.

    Returns
    -------
    pandas.DataFrame
        PSTH data loaded from the HDF5 file.
    """
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df
