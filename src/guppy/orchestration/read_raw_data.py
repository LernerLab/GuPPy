import logging
import multiprocessing as mp
import os

import numpy as np

from guppy.extractors import (
    CsvRecordingExtractor,
    DandiNwbRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    NwbRecordingExtractor,
    TdtRecordingExtractor,
    detect_acquisition_formats,
    read_and_save_events_for_extractor,
)
from guppy.extractors import base_recording_extractor as base_module
from guppy.extractors.base_recording_extractor import _pool_initializer
from guppy.orchestration.save_parameters import save_parameters
from guppy.utils import progress
from guppy.utils.progress import step_error_handler
from guppy.utils.utils import load_npm_params, select_run_folders

logger = logging.getLogger(__name__)


def _group_events_by_extractor(event_to_extractor: dict, events: np.ndarray) -> dict:
    """Partition ``events`` by their owning extractor instance, preserving order.

    Single-extractor sessions produce one group; mixed-modality sessions produce
    one group per format. Uses ``id(extractor)`` as the partitioning key so that
    the same instance is reused across multiple events (e.g. TDT's three stores
    all share one ``TdtRecordingExtractor`` instance).
    """
    grouped = {}
    for event in events:
        extractor = event_to_extractor[event]
        key = id(extractor)
        if key not in grouped:
            grouped[key] = (extractor, [])
        grouped[key][1].append(str(event))
    return {extractor: event_list for extractor, event_list in grouped.values()}


def _build_event_to_extractor(*, folder_path: str, store_array: np.ndarray, inputParameters: dict[str, object]) -> dict:
    """
    Build a mapping from event name to the extractor instance that owns it.

    Iterates over all acquisition formats present in the folder (via
    :func:`detect_all_formats`). When CSV shares a folder with other formats it is
    treated as an event-only source; only event_csv files are registered.

    Parameters
    ----------
    folder_path : str
        Path to the session folder.
    store_array : np.ndarray, shape (2, n)
        Row 0: store IDs. Row 1: store labels.
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
        format_events, _ = DandiNwbRecordingExtractor.discover_events_and_flags(folder_path=dandi_uri)
        for event in format_events:
            event_to_extractor[event] = extractor
        return event_to_extractor

    num_ch = inputParameters["noChannels"]
    all_formats = detect_acquisition_formats(folder_path)
    # Doric extractor requires a store-name→event-type mapping built from store_array
    event_name_to_event_type = {
        store_array[0, column_index]: store_array[1, column_index] for column_index in range(store_array.shape[1])
    }

    for acquisition_format in sorted(all_formats):
        if acquisition_format == "nwb":
            extractor = NwbRecordingExtractor(folder_path=folder_path)
            format_events, _ = NwbRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif acquisition_format == "tdt":
            extractor = TdtRecordingExtractor(folder_path=folder_path)
            format_events, _ = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif acquisition_format == "doric":
            extractor = DoricRecordingExtractor(
                folder_path=folder_path, event_name_to_event_type=event_name_to_event_type
            )
            format_events, _ = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif acquisition_format == "csv":
            extractor = CsvRecordingExtractor(folder_path=folder_path)
            format_events, _ = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif acquisition_format == "npm":
            extractor = NpmRecordingExtractor(
                folder_path=folder_path,
                num_ch=num_ch,
                npm_timestamp_column_name=inputParameters.get("npm_timestamp_column_name"),
                npm_time_unit=inputParameters.get("npm_time_unit"),
                npm_split_events=inputParameters.get("npm_split_events"),
            )
            format_events, _ = NpmRecordingExtractor.discover_events_and_flags(
                folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
            )
        else:
            raise ValueError(
                f"Format not recognized: '{acquisition_format}'. Expected one of 'nwb', 'tdt', 'csv', 'doric', 'npm'."
            )

        for event in format_events:
            if event not in event_to_extractor:
                event_to_extractor[event] = extractor

    return event_to_extractor


def orchestrate_read_raw_data(inputParameters: dict[str, object]) -> None:
    """Read raw acquisition data for all sessions and save to HDF5.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``session_folders``, ``numberOfCores``,
        and ``noChannels`` among other keys.
    """
    logger.debug("### Reading raw data... ###")
    # Snapshot the parameters being executed into each selected output dir so the
    # on-disk GuPPyParamtersUsed.json always reflects the last-run configuration.
    save_parameters(inputParameters=inputParameters)
    # get input parameters
    inputParameters = inputParameters
    session_folders = inputParameters["session_folders"]
    numProcesses = inputParameters["numberOfCores"]
    selected_runs = inputParameters.get("selected_runs", {}) or {}
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
    # resulting per-(session, extractor) tasks feed both the progress-bar
    # denominator and the worker pool.
    tasks = []
    total_samples = 0
    for filepath in session_folders:
        for run_folder in select_run_folders(filepath, selected_runs.get(filepath)):
            store_array = _load_stores_list(run_folder)
            events = np.unique(store_array[0, :])
            # NPM decomposition params chosen in Step 1 are persisted in the output dir;
            # merge them so the NPM extractor reproduces the same streams (e.g. split events).
            effective_parameters = {**inputParameters, **load_npm_params(run_folder)}
            event_to_extractor = _build_event_to_extractor(
                folder_path=filepath,
                store_array=store_array,
                inputParameters=effective_parameters,
            )
            event_total_samples = {}
            for event in events:
                extractor = event_to_extractor.get(event)
                if extractor is None:
                    available = sorted(event_to_extractor.keys())
                    raise ValueError(
                        f"Event '{event}' not found in any extractor for folder {filepath}. "
                        f"Available events: {available}."
                    )
                event_total_samples[event] = int(extractor.count_samples(event=event))
                total_samples += event_total_samples[event]

            # Group events by extractor instance identity so each task is one
            # batched read for one (session, extractor) pair. Mixed-modality
            # sessions (e.g. TDT data + CSV TTL) contribute one task per format.
            for extractor, grouped_events in _group_events_by_extractor(event_to_extractor, events).items():
                tasks.append(
                    (
                        extractor,
                        grouped_events,
                        run_folder,
                        {event: event_total_samples[event] for event in grouped_events},
                    )
                )

    # Bar denominator. Falls back to 1 for the rare degenerate case where no extractor
    # reports samples (all-ndx-events runs).
    progress.start(max(total_samples, 1))

    # This runs on a background thread of the Panel server process, which also carries the
    # Tornado IOLoop threads and a bound Bokeh socket. Forking that would copy only the
    # calling thread, leaving any lock another thread happened to hold (logging, HDF5)
    # locked forever in the child, so the start method is pinned instead of inherited.
    # ``samples_done`` must come from this same context to survive pickling into initargs.
    spawn_context = mp.get_context("spawn")
    samples_done = spawn_context.Value("q", 0)
    # The pool workers can only report into this shared counter, so progress is pulled from
    # it rather than pushed -- no thread is needed to copy the value across.
    progress.track(lambda: samples_done.value)

    if numProcesses <= 1:
        # Serial path: run tasks in the parent process so the shared counter
        # plumbing (and any test monkeypatches on the extractors) stays in scope.
        base_module._SAMPLES_DONE = samples_done
        try:
            for extractor, grouped_events, run_folder, event_totals in tasks:
                logger.debug(f"### Reading raw data for {len(grouped_events)} event(s) into {run_folder}")
                read_and_save_events_for_extractor(extractor, grouped_events, run_folder, event_totals)
        finally:
            base_module._SAMPLES_DONE = None
    else:
        with spawn_context.Pool(numProcesses, initializer=_pool_initializer, initargs=(samples_done,)) as pool:
            pool.starmap(read_and_save_events_for_extractor, tasks)
    logger.info("### Raw data fetched for all sessions")

    logger.info("Raw data fetched and saved.")
    logger.info("#" * 400)


def _load_stores_list(run_folder: str) -> np.ndarray:
    """Load the storesList CSV from the output directory.

    store_array is finalized in step 1 (including TDT split sub-events) and is no
    longer mutated during extraction, so it is read directly.
    """
    return np.genfromtxt(os.path.join(run_folder, "storesList.csv"), dtype="str", delimiter=",").reshape(2, -1)


@step_error_handler
def run_read_raw_data_step(input_parameters: dict[str, object]) -> None:
    """Run step-2 raw-data extraction with failure reporting attached.

    Parameters
    ----------
    input_parameters : dict
        Full pipeline input parameters.
    """
    logger.info("run")
    orchestrate_read_raw_data(input_parameters)
