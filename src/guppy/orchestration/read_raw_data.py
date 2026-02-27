import glob
import json
import logging
import multiprocessing as mp
import os
import sys

import numpy as np

from guppy.extractors import (
    CsvRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    TdtRecordingExtractor,
    detect_modality,
    read_and_save_all_events,
)
from guppy.frontend.progress import writeToFile
from guppy.utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)


# function to read data from 'tsq' and 'tev' files
def orchestrate_read_raw_data(inputParameters):

    logger.debug("### Reading raw data... ###")
    # get input parameters
    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    # modality = inputParameters["modality"]
    num_ch = inputParameters["noChannels"]
    storesListPath = []
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            "Warning : # of cores parameter set is greater than the cores available \
			   available in your machine"
        )
        numProcesses = mp.cpu_count() - 1
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        storesListPath.append(takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*"))))
    storesListPath = np.concatenate(storesListPath)
    writeToFile(str((storesListPath.shape[0] + 1) * 10) + "\n" + str(10) + "\n")
    step = 0
    for i in range(len(folderNames)):
        filepath = folderNames[i]
        modality = detect_modality(filepath)
        logger.debug(f"### Reading raw data for folder {folderNames[i]}")
        storesListPath = takeOnlyDirs(glob.glob(os.path.join(filepath, "*_output_*")))

        # read data corresponding to each storename selected by user while saving the storeslist file
        for j in range(len(storesListPath)):
            op = storesListPath[j]
            if os.path.exists(os.path.join(op, ".cache_storesList.csv")):
                storesList = np.genfromtxt(
                    os.path.join(op, ".cache_storesList.csv"), dtype="str", delimiter=","
                ).reshape(2, -1)
            else:
                storesList = np.genfromtxt(os.path.join(op, "storesList.csv"), dtype="str", delimiter=",").reshape(
                    2, -1
                )

            events = np.unique(storesList[0, :])
            if modality == "tdt":
                extractor = TdtRecordingExtractor(folder_path=filepath)
            elif modality == "doric":
                event_name_to_event_type = {storesList[0, i]: storesList[1, i] for i in range(storesList.shape[1])}
                extractor = DoricRecordingExtractor(
                    folder_path=filepath, event_name_to_event_type=event_name_to_event_type
                )
            elif modality == "csv":
                extractor = CsvRecordingExtractor(folder_path=filepath)
            elif modality == "npm":
                extractor = NpmRecordingExtractor(folder_path=filepath)
            else:
                raise ValueError("Modality not recognized. Please use 'tdt', 'csv', 'doric', or 'npm'.")
            read_and_save_all_events(extractor, events, op, numProcesses)

            writeToFile(str(10 + ((step + 1) * 10)) + "\n")
            step += 1
        logger.info(f"### Raw data for folder {folderNames[i]} fetched")
    logger.info("Raw data fetched and saved.")
    logger.info("#" * 400)


def main(input_parameters):
    logger.info("run")
    try:
        orchestrate_read_raw_data(input_parameters)
        logger.info("#" * 400)
    except Exception as e:
        with open(os.path.join(os.path.expanduser("~"), "pbSteps.txt"), "a") as file:
            file.write(str(-1) + "\n")
        logger.error(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
