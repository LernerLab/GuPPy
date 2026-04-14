"""Live streaming smoke tests for DandiNwbRecordingExtractor.

These tests hit the real DANDI Archive and are intended for **local use only**.
They are deselected in CI (see ``.github/workflows/run-tests.yml``) because
streaming from DANDI is subject to network flakiness and upstream availability.

Run locally with::

    pytest tests/unit/extractors/test_dandi_nwb_live.py -v -m dandi_live
"""

import h5py
import pytest

from guppy.extractors.dandi_nwb_recording_extractor import DandiNwbRecordingExtractor

# Target asset matches scripts/dandi_streaming_prototype.py — a relatively small
# behavior-only NWB file from a published fiber photometry dandiset.
DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"
DANDI_URI = f"dandi://{DANDISET_ID}/{ASSET_PATH}"

EXPECTED_FIBER_PHOTOMETRY_EVENTS = [
    "fiber_photometry_response_series_0",
    "fiber_photometry_response_series_1",
    "fiber_photometry_response_series_2",
    "fiber_photometry_response_series_3",
]


@pytest.mark.dandi_live
class TestDandiLiveStreaming:
    def test_discover_returns_expected_fiber_photometry_events(self):
        events, flags = DandiNwbRecordingExtractor.discover_events_and_flags(DANDI_URI)
        for expected_event in EXPECTED_FIBER_PHOTOMETRY_EVENTS:
            assert expected_event in events, f"Expected {expected_event!r} in discovered events {events}"
        assert flags == []

    def test_read_single_event_roundtrip(self, tmp_path):
        event = "fiber_photometry_response_series_0"
        extractor = DandiNwbRecordingExtractor(folder_path=DANDI_URI)
        output_dicts = extractor.read(events=[event], outputPath=str(tmp_path))
        assert len(output_dicts) == 1
        assert output_dicts[0]["storename"] == event

        extractor.save(output_dicts=output_dicts, outputPath=str(tmp_path))

        with h5py.File(tmp_path / f"{event}.hdf5", "r") as file:
            timestamps = file["timestamps"][:]
            data = file["data"][:]
            assert timestamps.shape[0] > 0
            assert data.shape[0] == timestamps.shape[0]
