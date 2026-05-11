import json
import logging
import multiprocessing as mp
import os
import sys
import threading

import numpy as np

from guppy.extractors import (
    CsvRecordingExtractor,
    DandiNwbRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    NwbRecordingExtractor,
    TdtRecordingExtractor,
    detect_acquisition_formats,
    read_and_save_all_events,
)
from guppy.frontend.progress import PB_STEPS_FILE, subprocess_main_handler, writeToFile
from guppy.utils.utils import select_output_dirs

logger = logging.getLogger(__name__)


def _progress_poller(samples_done, stop_event, *, file_path):
    """Periodically flush the shared samples counter into ``PB_STEPS_FILE``.

    Runs in the parent process while the multiprocessing pool drains. Each
    tick reads ``samples_done.value`` and appends ``value * 10`` to the
    progress file in the existing line-per-update format.
    """
    last_written = -1
    while not stop_event.wait(0.2):
        current = int(samples_done.value)
        if current != last_written:
            writeToFile(str(current * 10) + "\n", file_path=file_path)
            last_written = current


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

    # DANDI mode bypasses local format detection — discover and read via streaming
    if inputParameters is not None and inputParameters.get("mode") == "dandi":
        dandi_uri = inputParameters["dandi_uri_map"][folder_path]
        extractor = DandiNwbRecordingExtractor(folder_path=dandi_uri)
        fmt_events, _ = DandiNwbRecordingExtractor.discover_events_and_flags(folder_path=dandi_uri)
        for event in fmt_events:
            event_to_extractor[event] = extractor
        return event_to_extractor

    num_ch = inputParameters["noChannels"]
    all_formats = detect_acquisition_formats(folder_path)
    # Doric extractor requires a store-name→event-type mapping built from storesList
    event_name_to_event_type = {storesList[0, col]: storesList[1, col] for col in range(storesList.shape[1])}

    for fmt in sorted(all_formats):
        if fmt == "nwb":
            extractor = NwbRecordingExtractor(folder_path=folder_path)
            fmt_events, _ = NwbRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif fmt == "tdt":
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
            raise ValueError(f"Format not recognized: '{fmt}'. Expected one of 'nwb', 'tdt', 'csv', 'doric', 'npm'.")

        for event in fmt_events:
            if event not in event_to_extractor:
                event_to_extractor[event] = extractor

    return event_to_extractor


def orchestrate_read_raw_data(inputParameters):
    """Read raw acquisition data for all sessions and save to HDF5.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``folderNames``, ``numberOfCores``,
        and ``noChannels`` among other keys.
    """
    logger.debug("### Reading raw data... ###")
    # get input parameters
    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    numProcesses = inputParameters["numberOfCores"]
    selected_outputs = inputParameters.get("selectedOutputs", {}) or {}
    if numProcesses == 0:
        numProcesses = mp.cpu_count()
    elif numProcesses > mp.cpu_count():
        logger.warning(
            f"Number of cores requested ({numProcesses}) exceeds available cores "
            f"({mp.cpu_count()}); using {mp.cpu_count() - 1}."
        )
        numProcesses = mp.cpu_count() - 1

    # Pre-flight: walk every output folder, build its extractor map, and ask each
    # extractor for the per-event sample count via metadata-only inspection. The
    # resulting dict feeds the progress-bar denominator and per-worker reconciliation.
    work_units = []
    total_samples = 0
    for filepath in folderNames:
        for op in select_output_dirs(filepath, selected_outputs.get(filepath)):
            storesList = _load_stores_list(op)
            events = np.unique(storesList[0, :])
            event_to_extractor = _build_event_to_extractor(
                folder_path=filepath,
                storesList=storesList,
                inputParameters=inputParameters,
            )
            store_event_to_extractor = {}
            event_total_samples = {}
            for event in events:
                extractor = event_to_extractor.get(event)
                if extractor is None:
                    available = sorted(event_to_extractor.keys())
                    raise ValueError(
                        f"Event '{event}' not found in any extractor for folder {filepath}. "
                        f"Available events: {available}."
                    )
                store_event_to_extractor[event] = extractor
                event_total_samples[event] = (
                    int(extractor.count_samples(event=event)) if hasattr(extractor, "count_samples") else 0
                )
                total_samples += event_total_samples[event]
            work_units.append(
                {
                    "filepath": filepath,
                    "output_dir": op,
                    "event_to_extractor": store_event_to_extractor,
                    "event_total_samples": event_total_samples,
                }
            )

    # Bar denominator. Falls back to 10 so the file stays valid for the
    # rare degenerate case where no extractor reports samples (all-ndx-events runs).
    progress_max = max(total_samples, 1) * 10
    writeToFile(f"{progress_max}\n0\n", file_path=PB_STEPS_FILE)

    samples_done = mp.Value("q", 0)
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_progress_poller,
        args=(samples_done, stop_event),
        kwargs={"file_path": PB_STEPS_FILE},
        daemon=True,
    )
    poller.start()
    try:
        for unit in work_units:
            logger.debug(f"### Reading raw data for folder {unit['filepath']}")
            read_and_save_all_events(
                unit["event_to_extractor"],
                unit["output_dir"],
                numProcesses,
                samples_done=samples_done,
                event_total_samples=unit["event_total_samples"],
            )
            logger.info(f"### Raw data for folder {unit['filepath']} fetched")
    finally:
        # Reconcile to exact total so the bar always finishes at 100%, then drain
        # the poller thread before returning.
        with samples_done.get_lock():
            samples_done.value = max(int(samples_done.value), int(total_samples))
        stop_event.set()
        poller.join(timeout=2.0)
        # Final write guarantees the last value is on disk even if the poller
        # was mid-sleep when we set the stop event.
        writeToFile(f"{progress_max}\n", file_path=PB_STEPS_FILE)

    logger.info("Raw data fetched and saved.")
    logger.info("#" * 400)


def _load_stores_list(output_dir):
    """Load the storesList CSV (preferring the cached copy if it exists)."""
    cached_path = os.path.join(output_dir, ".cache_storesList.csv")
    source_path = cached_path if os.path.exists(cached_path) else os.path.join(output_dir, "storesList.csv")
    return np.genfromtxt(source_path, dtype="str", delimiter=",").reshape(2, -1)


@subprocess_main_handler
def main(input_parameters):
    """Subprocess entry point for step-3 raw-data extraction.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters deserialized from the subprocess argument.
    """
    logger.info("run")
    orchestrate_read_raw_data(input_parameters)


if __name__ == "__main__":
    input_parameters = json.loads(sys.argv[1])
    main(input_parameters=input_parameters)
