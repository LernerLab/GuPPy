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
    detect_acquisition_formats,
    read_and_save_all_events,
)
from guppy.frontend.progress import writeToFile
from guppy.utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)


def _build_event_to_extractor(*, folder_path, storesList, inputParameters):
    """
    Build a mapping from event name to the extractor instance that owns it.

    Iterates over all acquisition formats present in the folder (via
    :func:`detect_all_formats`). When CSV shares a folder with other formats it is
    treated as an event-only source; only event_csv files are registered.

    Parameters
    ----------
    folder_path : str
        Path to the session folder.
    storesList : np.ndarray, shape (2, n)
        Row 0: original store names. Row 1: semantic labels.
    inputParameters : dict
        Full pipeline input parameters (needed for NPM extractor configuration).

    Returns
    -------
    dict
        Maps each event name (str) to the extractor instance responsible for it.
    """
    event_to_extractor = {}
    num_ch = inputParameters["noChannels"]
    all_formats = detect_acquisition_formats(folder_path)
    # Doric extractor requires a store-name→event-type mapping built from storesList
    event_name_to_event_type = {storesList[0, col]: storesList[1, col] for col in range(storesList.shape[1])}

    for fmt in sorted(all_formats):
        if fmt == "tdt":
            extractor = TdtRecordingExtractor(folder_path=folder_path)
            fmt_events, _ = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif fmt == "doric":
            extractor = DoricRecordingExtractor(
                folder_path=folder_path, event_name_to_event_type=event_name_to_event_type
            )
            fmt_events, _ = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif fmt == "csv":
            extractor = CsvRecordingExtractor(folder_path=folder_path)
            fmt_events, _ = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif fmt == "npm":
            extractor = NpmRecordingExtractor(folder_path=folder_path)
            fmt_events, _ = NpmRecordingExtractor.discover_events_and_flags(
                folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
            )
        else:
            raise ValueError(f"Format not recognized: '{fmt}'. Expected one of 'tdt', 'csv', 'doric', 'npm'.")

        for event in fmt_events:
            if event not in event_to_extractor:
                event_to_extractor[event] = extractor

    return event_to_extractor


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
            event_to_extractor = _build_event_to_extractor(
                folder_path=filepath,
                storesList=storesList,
                inputParameters=inputParameters,
            )
            # Restrict to only the events named in storesList
            store_event_to_extractor = {}
            for event in events:
                ext = event_to_extractor.get(event)
                if ext is None:
                    raise ValueError(f"Event '{event}' not found in any extractor for folder {filepath}.")
                store_event_to_extractor[event] = ext
            read_and_save_all_events(store_event_to_extractor, op, numProcesses)

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
