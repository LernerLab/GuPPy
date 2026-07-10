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


def averageForGroup(folderNames: list[str], inputParameters: dict[str, object]) -> None:
    """
    Combine transient frequency and amplitude results across a group of sessions.

    Parameters
    ----------
    folderNames : list of str
        Session directories whose output subdirectories contain precomputed
        ``freqAndAmp_*.h5`` files.
    inputParameters : dict
        Analysis configuration dictionary; must include ``'abspath'`` and
        ``'selectForTransientsComputation'``.
    """

    logger.debug("Combining results for frequency and amplitude of transients in z-score data...")
    path = []
    abspath = inputParameters["abspath"]
    selectForTransientsComputation = inputParameters["selectForTransientsComputation"]

    for i in range(len(folderNames)):
        if selectForTransientsComputation == "z_score":
            matched_paths = glob.glob(os.path.join(folderNames[i], "z_score_*"))
        elif selectForTransientsComputation == "dff":
            matched_paths = glob.glob(os.path.join(folderNames[i], "dff_*"))
        else:
            matched_paths = glob.glob(os.path.join(folderNames[i], "z_score_*")) + glob.glob(
                os.path.join(folderNames[i], "dff_*")
            )

        for j in range(len(matched_paths)):
            basename = (os.path.basename(matched_paths[j])).split(".")[0]
            entry = [folderNames[i], basename]
            path.append(entry)

    naming = []
    for i in range(len(path)):
        naming.append(path[i][1])
    naming = np.unique(np.asarray(naming))

    # Size by the number of unique basenames across all folders so that mismatched
    # or non-overlapping storenames across sessions do not cause an IndexError.
    new_path = [[] for _ in range(len(naming))]
    for i in range(len(path)):
        index = np.where(naming == path[i][1])[0][0]
        new_path[index].append(path[i])

    output_dir = makeAverageDir(abspath)

    for i in range(len(new_path)):
        freq_and_amp_values = []
        fileName = []
        session_entries = new_path[i]
        for j in range(len(session_entries)):
            if not os.path.exists(os.path.join(session_entries[j][0], "freqAndAmp_" + session_entries[j][1] + ".h5")):
                continue
            else:
                df = read_freq_and_amp_from_hdf5(session_entries[j][0], session_entries[j][1])
                freq_and_amp_values.append(np.array([df["freq (events/min)"].iloc[0], df["amplitude"].iloc[0]]))
                fileName.append(os.path.basename(session_entries[j][0]))

        freq_and_amp_values = np.asarray(freq_and_amp_values)
        write_freq_and_amp_to_hdf5(
            output_dir,
            freq_and_amp_values,
            session_entries[j][1],
            index=fileName,
            columns=["freq (events/min)", "amplitude"],
        )
        write_freq_and_amp_to_csv(
            output_dir,
            freq_and_amp_values,
            "freqAndAmp_" + session_entries[j][1] + ".csv",
            index=fileName,
            columns=["freq (events/min)", "amplitude"],
        )
    logger.info("Results for frequency and amplitude of transients in z-score data are combined.")
