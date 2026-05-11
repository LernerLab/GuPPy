"""Base class for recording extractors."""

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# Per-worker handle on the shared "samples done" counter. Installed by
# ``_pool_initializer`` when ``read_and_save_all_events`` opens its
# multiprocessing pool. Stays ``None`` outside of that pool so direct unit-test
# calls to ``read_and_save_event`` are no-ops on progress accounting.
_SAMPLES_DONE: "mp.sharedctypes.Synchronized | None" = None


def _pool_initializer(samples_done):
    global _SAMPLES_DONE
    _SAMPLES_DONE = samples_done


def add_samples_done(delta: int) -> None:
    """Atomically add ``delta`` samples to the shared progress counter.

    No-op when the worker was started outside ``read_and_save_all_events`` or
    when ``delta`` is non-positive.
    """
    if _SAMPLES_DONE is None or delta <= 0:
        return
    with _SAMPLES_DONE.get_lock():
        _SAMPLES_DONE.value += int(delta)


class BaseRecordingExtractor(ABC):
    """
    Abstract base class for recording extractors.

    Defines the interface contract for reading and saving fiber photometry
    data from various acquisition formats (TDT, Doric, CSV, NPM, etc.).
    """

    @classmethod
    @abstractmethod
    def discover_events_and_flags(cls) -> tuple[list[str], list[str]]:
        """
        Discover available events and format flags from data files.

        Returns
        -------
        events : list of str
            Names of all events/stores available in the dataset.
        flags : list of str
            Format indicators or file type flags.
        """
        # NOTE: This method signature is intentionally minimal and flexible.
        # Different formats have different discovery requirements:
        # - TDT/CSV/Doric: need only folder_path parameter
        # - NPM: needs folder_path, num_ch, and optional inputParameters for interleaved channels
        # Each child class defines its own signature with the parameters it needs.
        pass

    @abstractmethod
    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data from source files for specified events.

        Parameters
        ----------
        events : list of str
            List of event/store names to extract from the data.
        outputPath : str
            Path to the output directory.

        Returns
        -------
        list of dict
            List of dictionaries containing extracted data. Each dictionary
            represents one event/store and contains keys such as 'storename',
            'timestamps', 'data', 'sampling_rate', etc.
        """
        pass

    @abstractmethod
    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 format.

        Parameters
        ----------
        output_dicts : list of dict
            List of data dictionaries from read().
        outputPath : str
            Path to the output directory.
        """
        pass

    @abstractmethod
    def stub(self, *, folder_path, duration_in_seconds=1.0):
        """
        Create a stubbed copy of the data folder truncated to a short duration.

        Copies the source folder to `folder_path`, then truncates data files so
        that only the first `duration_in_seconds` of recorded data are retained.
        If `folder_path` already exists it is overwritten.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed data. Created if it does not
            exist; overwritten if it already exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain in seconds. Default is 1.0.
        """
        pass


def read_and_save_event(extractor, event, outputPath, event_total_samples=0):
    """
    Read data for a single event and save it to HDF5.

    Intended as the per-worker function called by :func:`read_and_save_all_events`
    inside a multiprocessing pool.

    Parameters
    ----------
    extractor : BaseRecordingExtractor
        Extractor instance used to read and save the event.
    event : str
        Name of the event/store to read.
    outputPath : str
        Path to the output directory where HDF5 files are written.
    event_total_samples : int, optional
        Pre-computed total sample count for this event. Used to advance the
        shared progress counter. Default ``0`` (no progress reporting).
    """
    output_dicts = extractor.read(events=[event], outputPath=outputPath)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)
    # Extractors that report progress incrementally during read (e.g. DANDI's
    # passive byte counter) expose ``committed_samples_for_event``. Subtract
    # what they already committed so we only add the residual at event end.
    already_committed = 0
    if hasattr(extractor, "committed_samples_for_event"):
        already_committed = int(extractor.committed_samples_for_event(event))
    add_samples_done(int(event_total_samples) - already_committed)
    logger.info("Data for event {} fetched and stored.".format(event))


def read_and_save_all_events(
    event_to_extractor, outputPath, numProcesses=mp.cpu_count(), samples_done=None, event_total_samples=None
):
    """
    Read and save all events in parallel using a multiprocessing pool.

    Parameters
    ----------
    event_to_extractor : dict
        Mapping from event name (str) to the :class:`BaseRecordingExtractor`
        instance responsible for reading that event.
    outputPath : str
        Path to the output directory where HDF5 files are written.
    numProcesses : int, optional
        Number of worker processes. Defaults to ``multiprocessing.cpu_count()``.
    samples_done : multiprocessing.Value, optional
        Shared int64 counter used to track total samples processed across all
        workers. Installed into each worker via the pool initializer.
    event_total_samples : dict, optional
        Mapping from event name to its pre-computed total sample count. Used
        by workers to advance ``samples_done`` after each event completes.
    """
    events = list(event_to_extractor.keys())
    logger.info("Reading data for event {} ...".format(events))

    if event_total_samples is None:
        event_total_samples = {}

    start = time.time()
    # str() normalizes np.str_ scalars (e.g. dtype <U34 from NWB reads) before pickling.
    args = [
        (extractor, str(event), outputPath, int(event_total_samples.get(event, 0)))
        for event, extractor in event_to_extractor.items()
    ]
    with mp.Pool(numProcesses, initializer=_pool_initializer, initargs=(samples_done,)) as p:
        p.starmap(read_and_save_event, args)
    logger.info("Time taken = {0:.5f}".format(time.time() - start))
