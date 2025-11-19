import logging
import multiprocessing as mp
import os
import time
from itertools import repeat

import numpy as np
import pandas as pd

from guppy.common_step3 import write_hdf5

logger = logging.getLogger(__name__)


def execute_import_csv(filepath, event, outputPath, numProcesses=mp.cpu_count()):
    # logger.info("Reading data for event {} ...".format(event))

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(import_csv, zip(repeat(filepath), event, repeat(outputPath)))
    logger.info("Time taken = {0:.5f}".format(time.time() - start))


# function to read event timestamps csv file.
def import_csv(filepath, event, outputPath):
    logger.debug("\033[1m" + "Trying to read data for {} from csv file.".format(event) + "\033[0m")
    if not os.path.exists(os.path.join(filepath, event + ".csv")):
        logger.error("\033[1m" + "No csv file found for event {}".format(event) + "\033[0m")
        raise Exception("\033[1m" + "No csv file found for event {}".format(event) + "\033[0m")

    df = pd.read_csv(os.path.join(filepath, event + ".csv"), index_col=False)
    data = df
    key = list(df.columns)

    if len(key) == 3:
        arr1 = np.array(["timestamps", "data", "sampling_rate"])
        arr2 = np.char.lower(np.array(key))
        if (np.sort(arr1) == np.sort(arr2)).all() == False:
            logger.error("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")
            raise Exception("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")

    if len(key) == 1:
        if key[0].lower() != "timestamps":
            logger.error("\033[1m" + "Column names should be timestamps, data and sampling_rate" + "\033[0m")
            raise Exception("\033[1m" + "Column name should be timestamps" + "\033[0m")

    if len(key) != 3 and len(key) != 1:
        logger.error(
            "\033[1m"
            + "Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."
            + "\033[0m"
        )
        raise Exception(
            "\033[1m"
            + "Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."
            + "\033[0m"
        )

    for i in range(len(key)):
        write_hdf5(data[key[i]].dropna(), event, outputPath, key[i].lower())

    logger.info("\033[1m" + "Reading data for {} from csv file is completed.".format(event) + "\033[0m")

    return data, key
