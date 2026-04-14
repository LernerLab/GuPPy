"""DANDI streaming NWB recording extractor for GuPPy fiber photometry pipeline."""

import logging
from typing import Any

import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

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
    """
    with DandiAPIClient() as client:
        client.dandi_authenticate()
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(asset_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)
    file_system = remfile.File(s3_url)
    file = h5py.File(file_system, mode="r")
    io = NWBHDF5IO(file=file, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io


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
        nwbfile, io = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        events = _discover_events_from_nwbfile(nwbfile=nwbfile, io=io)
        io.close()
        return events, []

    def __init__(self, *, folder_path):
        self.folder_path = folder_path

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read data for the given events by streaming from DANDI.

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
        nwbfile, io = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        output_dicts = _read_events_from_nwbfile(nwbfile=nwbfile, io=io, events=events)
        io.close()
        return output_dicts

    def stub(self, *, folder_path, duration_in_seconds=1.0):
        """Stub method is not supported for DANDI-streamed NWB files."""
        raise NotImplementedError("Stub method is not supported for DANDI-streamed NWB files.")
