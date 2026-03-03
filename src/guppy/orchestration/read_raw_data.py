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
    detect_ttl_modalities,
    read_and_save_all_events,
)
from guppy.frontend.progress import writeToFile
from guppy.utils.utils import takeOnlyDirs

logger = logging.getLogger(__name__)


def _build_event_to_extractor(*, data_modality, ttl_modalities, folder_path, storesList, inputParameters):
    """
    Build a mapping from event name to the extractor instance that owns it.

    Used when TTL modalities differ from the data modality so each event in
    storesList can be routed to the correct extractor.

    Parameters
    ----------
    data_modality : str
        The modality of the photometry (control/signal) data.
    ttl_modalities : set of str
        All modalities that supply TTL/event data in this folder.
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

    # Register all events discoverable by the data modality extractor
    if data_modality == "tdt":
        data_extractor = TdtRecordingExtractor(folder_path=folder_path)
        data_events, _ = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
    elif data_modality == "doric":
        event_name_to_event_type = {storesList[0, col]: storesList[1, col] for col in range(storesList.shape[1])}
        data_extractor = DoricRecordingExtractor(
            folder_path=folder_path, event_name_to_event_type=event_name_to_event_type
        )
        data_events, _ = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
    elif data_modality == "csv":
        data_extractor = CsvRecordingExtractor(folder_path=folder_path)
        data_events, _ = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
    elif data_modality == "npm":
        data_extractor = NpmRecordingExtractor(folder_path=folder_path)
        data_events, _ = NpmRecordingExtractor.discover_events_and_flags(
            folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
        )
    else:
        raise ValueError(f"Data modality not recognized: '{data_modality}'.")

    for event in data_events:
        event_to_extractor[event] = data_extractor

    # Register events from TTL-only modalities (those not already covered by data extractor)
    for ttl_mod in ttl_modalities - {data_modality}:
        if ttl_mod == "csv":
            ttl_extractor = CsvRecordingExtractor(folder_path=folder_path)
            ttl_events, ttl_flags = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
            for event, flag in zip(ttl_events, ttl_flags):
                # Only route event_csv files; data_csv files in a mixed folder belong to the data extractor
                if "event_csv" in flag and event not in event_to_extractor:
                    event_to_extractor[event] = ttl_extractor
        elif ttl_mod == "tdt":
            ttl_extractor = TdtRecordingExtractor(folder_path=folder_path)
            ttl_events, _ = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
            for event in ttl_events:
                if event not in event_to_extractor:
                    event_to_extractor[event] = ttl_extractor
        elif ttl_mod == "doric":
            event_name_to_event_type = {storesList[0, col]: storesList[1, col] for col in range(storesList.shape[1])}
            ttl_extractor = DoricRecordingExtractor(
                folder_path=folder_path, event_name_to_event_type=event_name_to_event_type
            )
            ttl_events, _ = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
            for event in ttl_events:
                if event not in event_to_extractor:
                    event_to_extractor[event] = ttl_extractor
        elif ttl_mod == "npm":
            ttl_extractor = NpmRecordingExtractor(folder_path=folder_path)
            ttl_events, _ = NpmRecordingExtractor.discover_events_and_flags(
                folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
            )
            for event in ttl_events:
                if event not in event_to_extractor:
                    event_to_extractor[event] = ttl_extractor

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
        requested_modality = inputParameters.get("modality", "auto")
        modality = detect_modality(filepath) if requested_modality == "auto" else requested_modality
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
            ttl_modalities = detect_ttl_modalities(filepath)

            if ttl_modalities == {modality}:
                # Fast path: a single extractor covers both photometry and TTL (unchanged behavior)
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
                    raise ValueError("Modality not recognized. Please use 'auto', 'tdt', 'csv', 'doric', or 'npm'.")
                read_and_save_all_events(extractor, events, op, numProcesses)
            else:
                # Mixed path: TTL sources differ from the data modality; route per-event
                event_to_extractor = _build_event_to_extractor(
                    data_modality=modality,
                    ttl_modalities=ttl_modalities,
                    folder_path=filepath,
                    storesList=storesList,
                    inputParameters=inputParameters,
                )
                # Group events by their assigned extractor, preserving insertion order
                seen_ids = {}
                partitions = []
                for event in events:
                    ext = event_to_extractor.get(event)
                    if ext is None:
                        raise ValueError(
                            f"Event '{event}' not found in any extractor for folder {filepath}. "
                            f"Data modality: '{modality}', TTL modalities: {ttl_modalities}."
                        )
                    ext_id = id(ext)
                    if ext_id not in seen_ids:
                        seen_ids[ext_id] = len(partitions)
                        partitions.append((ext, []))
                    partitions[seen_ids[ext_id]][1].append(event)
                for ext, ext_events in partitions:
                    read_and_save_all_events(ext, ext_events, op, numProcesses)

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
