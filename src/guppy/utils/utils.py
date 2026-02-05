import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def takeOnlyDirs(paths):
    removePaths = []
    for p in paths:
        if os.path.isfile(p):
            removePaths.append(p)
    return list(set(paths) - set(removePaths))


def get_all_stores_for_combining_data(folderNames):
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


# function to read h5 file and make a dataframe from it
def read_Df(filepath, event, name):
    event = event.replace("\\", "_")
    event = event.replace("/", "_")
    if name:
        op = os.path.join(filepath, event + "_{}.h5".format(name))
    else:
        op = os.path.join(filepath, event + ".h5")
    df = pd.read_hdf(op, key="df", mode="r")

    return df
