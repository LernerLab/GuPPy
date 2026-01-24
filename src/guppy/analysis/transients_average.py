import glob
import logging
import os

import numpy as np

from .io_utils import (
    makeAverageDir,
)
from .standard_io import (
    read_freq_and_amp_from_hdf5,
    write_freq_and_amp_to_csv,
    write_freq_and_amp_to_hdf5,
)

logger = logging.getLogger(__name__)


def averageForGroup(folderNames, inputParameters):

    logger.debug("Combining results for frequency and amplitude of transients in z-score data...")
    path = []
    abspath = inputParameters["abspath"]
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]
    path_temp_len = []

    for i in range(len(folderNames)):
        if selectForTransientsComputation == "z_score":
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*"))
        elif selectForTransientsComputation == "dff":
            path_temp = glob.glob(os.path.join(folderNames[i], "dff_*"))
        else:
            path_temp = glob.glob(os.path.join(folderNames[i], "z_score_*")) + glob.glob(
                os.path.join(folderNames[i], "dff_*")
            )

        path_temp_len.append(len(path_temp))

        for j in range(len(path_temp)):
            basename = (os.path.basename(path_temp[j])).split(".")[0]
            # name = name[0]
            temp = [folderNames[i], basename]
            path.append(temp)

    path_temp_len = np.asarray(path_temp_len)
    max_len = np.argmax(path_temp_len)

    naming = []
    for i in range(len(path)):
        naming.append(path[i][1])
    naming = np.unique(np.asarray(naming))

    new_path = [[] for _ in range(path_temp_len[max_len])]
    for i in range(len(path)):
        idx = np.where(naming == path[i][1])[0][0]
        new_path[idx].append(path[i])

    op = makeAverageDir(abspath)

    for i in range(len(new_path)):
        arr = []  # np.zeros((len(new_path[i]), 2))
        fileName = []
        temp_path = new_path[i]
        for j in range(len(temp_path)):
            if not os.path.exists(os.path.join(temp_path[j][0], "freqAndAmp_" + temp_path[j][1] + ".h5")):
                continue
            else:
                df = read_freq_and_amp_from_hdf5(temp_path[j][0], temp_path[j][1])
                arr.append(np.array([df["freq (events/min)"][0], df["amplitude"][0]]))
                fileName.append(os.path.basename(temp_path[j][0]))

        arr = np.asarray(arr)
        write_freq_and_amp_to_hdf5(op, arr, temp_path[j][1], index=fileName, columns=["freq (events/min)", "amplitude"])
        write_freq_and_amp_to_csv(
            op,
            arr,
            "freqAndAmp_" + temp_path[j][1] + ".csv",
            index=fileName,
            columns=["freq (events/min)", "amplitude"],
        )
    logger.info("Results for frequency and amplitude of transients in z-score data are combined.")
