"""DANDI streaming NWB recording extractor for GuPPy fiber photometry pipeline."""

import logging
from typing import Any

import h5py
import numpy as np
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

from guppy.extractors.base_recording_extractor import add_samples_done
from guppy.extractors.nwb_recording_extractor import (
    NwbRecordingExtractor,
    _discover_events_from_nwbfile,
    _read_events_from_nwbfile,
)

logger = logging.getLogger(__name__)

DANDI_URI_PREFIX = "dandi://"


def is_dandi_uri(path):
    """
    Check whether a path is a DANDI URI.

    Parameters
    ----------
    path : str
        Path to check.

    Returns
    -------
    bool
        True if the path starts with ``dandi://``.
    """
    return isinstance(path, str) and path.startswith(DANDI_URI_PREFIX)


def parse_dandi_uri(uri):
    """
    Parse a DANDI URI into its dandiset ID and asset path.

    Parameters
    ----------
    uri : str
        A DANDI URI of the form ``dandi://DANDISET_ID/asset/path.nwb``.

    Returns
    -------
    dandiset_id : str
        The dandiset identifier (e.g. ``"000971"``).
    asset_path : str
        The asset path within the dandiset (e.g. ``"sub-112-283/file.nwb"``).
    """
    stripped = uri[len(DANDI_URI_PREFIX) :]
    parts = stripped.split("/", 1)
    dandiset_id = parts[0]
    asset_path = parts[1]
    return dandiset_id, asset_path


class _CountingRemfile:
    """File-like wrapper that counts bytes read from a ``remfile.File``.

    Forwards every file-protocol method verbatim — h5py sees the same interface
    as the underlying object — and on each ``.read(n)`` increments an internal
    byte counter. When ``set_event`` has been called, bytes are converted to
    a samples delta using the event's known byte/sample budget and pushed to
    the shared ``samples_done`` counter via ``add_samples_done``. Bytes seen
    outside an active event (during NWB file open, metadata reads, etc.) are
    not attributed to progress.

    The wrapper does not buffer, reshape, or coalesce reads; performance is
    unchanged from talking to the underlying ``remfile.File`` directly.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._current_event = None
        self._event_total_bytes = 0
        self._event_total_samples = 0
        self._bytes_seen_for_event = 0
        self._samples_committed_for_event = 0
        self._committed_samples_by_event: dict[str, int] = {}

    def set_event(self, *, event: str, total_bytes: int, total_samples: int) -> None:
        self.end_event()
        self._current_event = event
        self._event_total_bytes = int(total_bytes)
        self._event_total_samples = int(total_samples)
        self._bytes_seen_for_event = 0
        self._samples_committed_for_event = 0

    def end_event(self) -> None:
        if self._current_event is None:
            return
        # Top up to the event's full sample count: DANDI datasets are typically
        # compressed, so bytes-on-wire is less than the uncompressed dataset size
        # and the proportional counter never reaches 100% on its own. The
        # event boundary is the authoritative "done" signal.
        remainder = self._event_total_samples - self._samples_committed_for_event
        if remainder > 0:
            add_samples_done(remainder)
            self._samples_committed_for_event = self._event_total_samples
        self._committed_samples_by_event[self._current_event] = self._samples_committed_for_event
        self._current_event = None
        self._event_total_bytes = 0
        self._event_total_samples = 0
        self._bytes_seen_for_event = 0
        self._samples_committed_for_event = 0

    def committed_samples_for_event(self, event: str) -> int:
        return self._committed_samples_by_event.get(event, 0)

    def read(self, *args, **kwargs):
        data = self._wrapped.read(*args, **kwargs)
        if self._current_event is not None and data and self._event_total_bytes > 0:
            self._bytes_seen_for_event += len(data)
            target_samples = min(
                self._event_total_samples,
                self._bytes_seen_for_event * self._event_total_samples // self._event_total_bytes,
            )
            delta = target_samples - self._samples_committed_for_event
            if delta > 0:
                add_samples_done(delta)
                self._samples_committed_for_event = target_samples
        return data

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# Excluded from coverage: hits the real DANDI Archive over the network. Exercised by
# the local-only live suite tests/unit/extractors/test_dandi_nwb_live.py (marker
# ``dandi_live``, deselected in CI). Offline tests monkeypatch this function to open
# a local mock NWB file instead.
def _stream_nwb(*, dandiset_id, asset_path):  # pragma: no cover
    """
    Open a streaming connection to an NWB file on the DANDI Archive.

    Parameters
    ----------
    dandiset_id : str
        Dandiset ID.
    asset_path : str
        Path to the NWB file within the dandiset.

    Returns
    -------
    nwbfile : pynwb.NWBFile
        The opened NWB file.
    io : pynwb.NWBHDF5IO
        The IO object. Must be closed by the caller when done.
    counter : _CountingRemfile
        Wrapper around the underlying remfile providing per-event byte tracking.
    """
    with DandiAPIClient() as client:
        client.dandi_authenticate()
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(asset_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)
    file_system = remfile.File(s3_url)
    counter = _CountingRemfile(file_system)
    file = h5py.File(counter, mode="r")
    io = NWBHDF5IO(file=file, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io, counter


class DandiNwbRecordingExtractor(NwbRecordingExtractor):
    """
    Recording extractor that streams NWB fiber photometry data from the DANDI Archive.

    Accepts a DANDI URI (``dandi://DANDISET_ID/asset_path``) as the folder path
    and streams data via remfile without downloading the entire file.

    The streaming IO connection is opened during ``__init__`` and remains open
    for the lifetime of the extractor so that lazy data access works during
    ``read()``. Call ``close()`` when finished, or use as a context manager.
    """

    @classmethod
    def discover_events_and_flags(cls, folder_path) -> tuple[list[str], list[str]]:
        """
        Discover available events from a DANDI-hosted NWB file.

        Parameters
        ----------
        folder_path : str
            A DANDI URI of the form ``dandi://DANDISET_ID/asset_path``.

        Returns
        -------
        events : list of str
            Event names derived from ``FiberPhotometryResponseSeries`` and ndx-events objects.
        flags : list of str
            Always empty for NWB files.
        """
        dandiset_id, asset_path = parse_dandi_uri(folder_path)
        nwbfile, io, _ = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        events = _discover_events_from_nwbfile(nwbfile=nwbfile, io=io)
        io.close()
        return events, []

    def __init__(self, *, folder_path):
        self.folder_path = folder_path
        self._sample_count_cache: dict[str, int] | None = None
        self._byte_count_cache: dict[str, int] | None = None
        self._last_counter: _CountingRemfile | None = None

    def _ensure_count_cache(self) -> None:
        if self._sample_count_cache is not None:
            return
        dandiset_id, asset_path = parse_dandi_uri(self.folder_path)
        nwbfile, io, _ = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        sample_counts: dict[str, int] = {}
        byte_counts: dict[str, int] = {}
        for neurodata_object in nwbfile.objects.values():
            if neurodata_object.neurodata_type != "FiberPhotometryResponseSeries":
                continue
            data = neurodata_object.data
            total_samples = int(data.shape[0])
            total_bytes = int(data.dtype.itemsize * int(np.prod(data.shape)))
            if len(data.shape) == 2:
                for column_index in range(data.shape[1]):
                    event_name = f"{neurodata_object.name}_{column_index}"
                    sample_counts[event_name] = total_samples
                    byte_counts[event_name] = total_bytes
            else:
                sample_counts[neurodata_object.name] = total_samples
                byte_counts[neurodata_object.name] = total_bytes
        io.close()
        self._sample_count_cache = sample_counts
        self._byte_count_cache = byte_counts

    def count_samples(self, *, event: str) -> int:
        """Return the total number of samples for ``event`` via cached metadata."""
        self._ensure_count_cache()
        return int(self._sample_count_cache.get(event, 0))

    def committed_samples_for_event(self, event: str) -> int:
        """Return samples already pushed to the shared counter for ``event`` during read."""
        if self._last_counter is None:
            return 0
        return self._last_counter.committed_samples_for_event(event)

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data for the given events by streaming from DANDI.

        Reads events one at a time so that the passive byte counter on the
        underlying remfile can attribute bytes to the correct event.

        Parameters
        ----------
        events : list of str
            Event names to extract (as returned by ``discover_events_and_flags``).
        outputPath : str
            Path to the output directory (unused during read; required by interface).

        Returns
        -------
        list of dict
            One dictionary per event with keys: ``storename``, ``sampling_rate``,
            ``timestamps``, ``data``, ``npoints``.
        """
        dandiset_id, asset_path = parse_dandi_uri(self.folder_path)
        nwbfile, io, counter = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        self._last_counter = counter
        sample_counts = self._sample_count_cache or {}
        byte_counts = self._byte_count_cache or {}
        output_dicts: list[dict[str, Any]] = []
        try:
            for event in events:
                total_samples = int(sample_counts.get(event, 0))
                total_bytes = int(byte_counts.get(event, 0))
                counter.set_event(event=event, total_bytes=total_bytes, total_samples=total_samples)
                try:
                    output_dicts.extend(_read_events_from_nwbfile(nwbfile=nwbfile, io=io, events=[event]))
                finally:
                    counter.end_event()
        finally:
            io.close()
        return output_dicts

    def stub(self, *, folder_path, duration_in_seconds=1.0):
        """Stub method is not supported for DANDI-streamed NWB files."""
        raise NotImplementedError("Stub method is not supported for DANDI-streamed NWB files.")
