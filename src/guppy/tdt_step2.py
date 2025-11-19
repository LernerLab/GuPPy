import glob
import logging
import os

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16

logger = logging.getLogger(__name__)


# function to read 'tsq' file
def readtsq(filepath):
    names = ("size", "type", "name", "chan", "sort_code", "timestamp", "fp_loc", "strobe", "format", "frequency")
    formats = (int32, int32, "S4", uint16, uint16, float64, int64, float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets}, align=True)
    path = glob.glob(os.path.join(filepath, "*.tsq"))
    if len(path) > 1:
        logger.error("Two tsq files are present at the location.")
        raise Exception("Two tsq files are present at the location.")
    elif len(path) == 0:
        return 0
    else:
        path = path[0]
    tsq = np.fromfile(path, dtype=tsq_dtype)
    df = pd.DataFrame(tsq)
    return df
