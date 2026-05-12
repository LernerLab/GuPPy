"""Base class for recording extractors."""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# Per-worker handle on the shared "samples done" counter. Installed by
# ``_pool_initializer`` when the orchestrator opens its multiprocessing pool,
# or directly by the orchestrator's serial path. Stays ``None`` outside of
# either context so direct unit-test calls are no-ops on progress accounting.
_SAMPLES_DONE: "mp.sharedctypes.Synchronized | None" = None


def _pool_initializer(samples_done):
    global _SAMPLES_DONE
    _SAMPLES_DONE = samples_done


def add_samples_done(delta: int) -> None:
    """Atomically add ``delta`` samples to the shared progress counter.

    No-op when ``_SAMPLES_DONE`` has not been installed, or when ``delta`` is
    non-positive.
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


def read_and_save_events_for_extractor(extractor, events, outputPath, event_total_samples):
    """
    Read all events owned by one extractor in a single batched call, save, and
    account progress per event.

    Intended as the per-task unit of work invoked by ``orchestrate_read_raw_data``
    — either directly in the parent process (single-core path) or inside a
    multiprocessing pool worker (multi-core path).

    Parameters
    ----------
    extractor : BaseRecordingExtractor
        Extractor instance used to read and save the events.
    events : list of str
        Names of the events/stores to read in this single batched call.
    outputPath : str
        Path to the output directory where HDF5 files are written.
    event_total_samples : dict
        Mapping from event name (str) to its pre-computed total sample count.
        Used to advance the shared progress counter after the read returns.
    """
    # str() normalizes np.str_ scalars (e.g. dtype <U34 from NWB reads) so the
    # event names are plain Python strings before they cross any boundaries.
    normalized_events = [str(event) for event in events]
    output_dicts = extractor.read(events=normalized_events, outputPath=outputPath)
    extractor.save(output_dicts=output_dicts, outputPath=outputPath)
    # Extractors that report progress incrementally during read (e.g. DANDI's
    # passive byte counter) expose ``committed_samples_for_event``. Subtract
    # what they already committed so we only add the residual at event end.
    has_passive_counter = hasattr(extractor, "committed_samples_for_event")
    for event in normalized_events:
        already_committed = int(extractor.committed_samples_for_event(event)) if has_passive_counter else 0
        add_samples_done(int(event_total_samples.get(event, 0)) - already_committed)
        logger.info("Data for event {} fetched and stored.".format(event))
