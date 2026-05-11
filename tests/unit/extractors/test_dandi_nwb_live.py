"""Live streaming contract tests for DandiNwbRecordingExtractor.

These tests hit the real DANDI Archive and are intended for **local use only**.
They are deselected in CI (see ``.github/workflows/run-tests.yml``) because
streaming from DANDI is subject to network flakiness and upstream availability.

Run locally with::

    pytest tests/unit/extractors/test_dandi_nwb_live.py -v -m dandi_live

They inherit the full ``NwbRecordingExtractorTestMixin`` contract so the live
session is exercised by the same discover/read/save/roundtrip tests as the
offline mocks — just against a real streamed NWB file instead of a local one.
"""

import numpy as np
import pytest

from guppy.extractors.dandi_nwb_recording_extractor import (
    DandiNwbRecordingExtractor,
    _stream_nwb,
    parse_dandi_uri,
)

from .test_nwb_recording_extractor import NwbRecordingExtractorTestMixin

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
class TestDandiLiveContract(NwbRecordingExtractorTestMixin):
    """Full NWB recording-extractor contract against a real DANDI-streamed session."""

    extractor_class = DandiNwbRecordingExtractor
    folder_path = DANDI_URI
    file_path = DANDI_URI  # unused: expected_*_data fixtures below stream instead
    extractor_instance = DandiNwbRecordingExtractor(folder_path=DANDI_URI)
    expected_events = EXPECTED_FIBER_PHOTOMETRY_EVENTS
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    # TTL event matches one of the behavior events mapped in
    # scripts/dandi_streaming_prototype.py's STORENAMES_MAP for this asset.
    ttl_event = "right_nose_poke_times"

    @pytest.fixture
    def isolated_folder_path(self):
        # Override base-mixin default which wraps folder_path in Path(), which
        # collapses ``dandi://`` to ``dandi:/``. The DANDI extractor needs the
        # raw URI string.
        return self.folder_path

    @pytest.fixture(scope="class")
    def streamed_nwbfile(self):
        """Stream the live NWB file once per class and hold IO open for the session."""
        dandiset_id, asset_path = parse_dandi_uri(DANDI_URI)
        nwbfile, io, _ = _stream_nwb(dandiset_id=dandiset_id, asset_path=asset_path)
        yield nwbfile
        io.close()

    @pytest.fixture
    def expected_control_timestamps(self, streamed_nwbfile):
        series = streamed_nwbfile.acquisition["fiber_photometry_response_series"]
        starting_time = series.starting_time if series.starting_time is not None else 0.0
        return starting_time + np.arange(series.data.shape[0]) / series.rate

    @pytest.fixture
    def expected_control_data(self, streamed_nwbfile):
        return np.array(streamed_nwbfile.acquisition["fiber_photometry_response_series"].data[:, 0])

    @pytest.fixture
    def expected_signal_timestamps(self, streamed_nwbfile):
        series = streamed_nwbfile.acquisition["fiber_photometry_response_series"]
        starting_time = series.starting_time if series.starting_time is not None else 0.0
        return starting_time + np.arange(series.data.shape[0]) / series.rate

    @pytest.fixture
    def expected_signal_data(self, streamed_nwbfile):
        return np.array(streamed_nwbfile.acquisition["fiber_photometry_response_series"].data[:, 1])

    @pytest.fixture
    def expected_ttl_timestamps(self, streamed_nwbfile):
        ttl_objects = [obj for obj in streamed_nwbfile.objects.values() if obj.name == self.ttl_event]
        assert (
            len(ttl_objects) == 1
        ), f"Expected exactly one NWB object named {self.ttl_event!r}, found {len(ttl_objects)}"
        return np.array(ttl_objects[0].timestamps[:])

    def test_byte_counter_commits_full_event_after_read(self, tmp_path):
        """After a real DANDI streamed read, the passive byte counter should have
        attributed enough bytes to commit the full sample count for the event."""
        extractor = DandiNwbRecordingExtractor(folder_path=DANDI_URI)
        total_samples = extractor.count_samples(event=self.control_event)
        extractor.read(events=[self.control_event], outputPath=str(tmp_path))
        committed = extractor.committed_samples_for_event(self.control_event)
        assert committed == total_samples
