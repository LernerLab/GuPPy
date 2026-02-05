import logging
import os
import re

import numpy as np

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
