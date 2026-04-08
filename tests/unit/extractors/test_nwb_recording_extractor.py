"""Tests for NwbRecordingExtractor and its private helper functions."""

import numpy as np
import pytest
from conftest import STUBBED_TESTING_DATA
from pynwb import TimeSeries, read_nwb

from guppy.extractors.nwb_recording_extractor import (
    NwbRecordingExtractor,
    _find_nwb_file,
    _parse_event_name,
    _resolve_timing,
)

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

MOCK_NWB_FOLDER = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile"
MOCK_NWB_FILE = MOCK_NWB_FOLDER / "mock_nwbfile.nwb"

_NUM_SAMPLES = 3000
_SAMPLING_RATE = 30.0
_STARTING_TIME = 0.0


class TestFindNwbFile:
    def test_finds_single_nwb_file(self, tmp_path):
        nwb_file = tmp_path / "test.nwb"
        nwb_file.touch()
        result = _find_nwb_file(tmp_path)
        assert result == str(nwb_file)

    def test_raises_when_no_nwb_file(self, tmp_path):
        with pytest.raises(Exception, match="No NWB file found"):
            _find_nwb_file(tmp_path)

    def test_raises_when_two_nwb_files(self, tmp_path):
        (tmp_path / "first.nwb").touch()
        (tmp_path / "second.nwb").touch()
        with pytest.raises(Exception, match="Two NWB files"):
            _find_nwb_file(tmp_path)


class TestParseEventName:
    def test_resolves_single_channel_series(self):
        series_name_to_object = {"my_series": object()}
        series_name, column_index = _parse_event_name("my_series", series_name_to_object)
        assert series_name == "my_series"
        assert column_index is None

    def test_resolves_multi_channel_event(self):
        series_name_to_object = {"my_series": object()}
        series_name, column_index = _parse_event_name("my_series_1", series_name_to_object)
        assert series_name == "my_series"
        assert column_index == 1

    def test_raises_for_unknown_event(self):
        series_name_to_object = {"my_series": object()}
        with pytest.raises(ValueError, match="could not be resolved"):
            _parse_event_name("unknown_event", series_name_to_object)


class TestResolveTiming:
    def test_rate_with_starting_time(self):
        series = TimeSeries(name="test", data=np.zeros(5), unit="n.a.", rate=30.0, starting_time=10.0)
        sampling_rate, timestamps = _resolve_timing(series, 5)
        assert sampling_rate == 30.0
        np.testing.assert_allclose(timestamps, 10.0 + np.arange(5) / 30.0)

    def test_rate_with_no_starting_time_defaults_to_zero(self):
        series = TimeSeries(name="test", data=np.zeros(5), unit="n.a.", rate=30.0)
        sampling_rate, timestamps = _resolve_timing(series, 5)
        assert sampling_rate == 30.0
        np.testing.assert_allclose(timestamps[0], 0.0)

    def test_explicit_timestamps(self):
        series = TimeSeries(name="test", data=np.zeros(3), unit="n.a.", timestamps=[0.0, 0.1, 0.2])
        sampling_rate, timestamps = _resolve_timing(series, 3)
        assert sampling_rate == pytest.approx(10.0, rel=1e-3)
        np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2])

    def test_raises_when_neither_rate_nor_timestamps(self):
        # pynwb requires either rate or timestamps at construction time, so we use a
        # minimal stand-in to exercise the error branch in _resolve_timing.
        class _NoTimingMockSeries:
            name = "mock"
            rate = None
            timestamps = None

        with pytest.raises(Exception, match="must have either 'rate' or 'timestamps'"):
            _resolve_timing(_NoTimingMockSeries(), 5)


class TestNwbRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = NwbRecordingExtractor
    folder_path = str(MOCK_NWB_FOLDER)
    expected_events = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
        "AnnotatedEventsTable_Reward",
        "AnnotatedEventsTable_Punishment",
        "labeled_events_label_1",
        "labeled_events_label_2",
        "labeled_events_label_3",
        "events",
    ]
    discover_kwargs = {}
    extractor_instance = NwbRecordingExtractor(folder_path=str(MOCK_NWB_FOLDER))
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    ttl_event = "events"

    @pytest.fixture
    def expected_control_timestamps(self):
        # rate=30 Hz, starting_time=0.0 → timestamps are exactly arange(3000)/30
        return np.arange(_NUM_SAMPLES) / _SAMPLING_RATE

    @pytest.fixture
    def expected_control_data(self):
        nwbfile = read_nwb(str(MOCK_NWB_FILE))
        return np.array(nwbfile.acquisition["fiber_photometry_response_series"].data[:, 0])

    @pytest.fixture
    def expected_signal_timestamps(self):
        return np.arange(_NUM_SAMPLES) / _SAMPLING_RATE

    @pytest.fixture
    def expected_signal_data(self):
        nwbfile = read_nwb(str(MOCK_NWB_FILE))
        return np.array(nwbfile.acquisition["fiber_photometry_response_series"].data[:, 1])

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # Events timestamps: 45, 46, ..., 54
        return np.arange(45, 55, dtype=np.float64)

    # --- override stub tests (stub() raises NotImplementedError for NWB) ---

    def test_stub_data_matches_original(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    def test_stub_idempotent(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    def test_stub_ttl_timestamps_within_duration(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    @pytest.mark.parametrize("stub_duration_in_seconds", [0.5, 1.0, 2.0])
    def test_stub_duration(self, tmp_path, isolated_extractor_instance, stub_duration_in_seconds):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    # --- NWB-specific tests ---

    def test_discover_events_count(self):
        events, _ = NwbRecordingExtractor.discover_events_and_flags(self.folder_path)
        assert len(events) == 8

    def test_discover_flags_empty(self):
        _, flags = NwbRecordingExtractor.discover_events_and_flags(self.folder_path)
        assert flags == []

    def test_read_annotated_reward_event_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=["AnnotatedEventsTable_Reward"], outputPath=str(tmp_path))
        np.testing.assert_array_equal(result[0]["timestamps"], np.array([41.0, 42.0, 43.0, 44.0, 45.0]))

    def test_read_annotated_punishment_event_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=["AnnotatedEventsTable_Punishment"], outputPath=str(tmp_path))
        np.testing.assert_array_equal(result[0]["timestamps"], np.array([55.0, 56.0, 57.0, 58.0, 59.0]))

    def test_read_labeled_events_label_1_timestamps(self, tmp_path):
        # label_1 is index 0 in the LabeledEvents data array: positions 0, 3, 6, 9, 12 → 40, 43, 46, 49, 52
        result = self.extractor_instance.read(events=["labeled_events_label_1"], outputPath=str(tmp_path))
        np.testing.assert_array_equal(result[0]["timestamps"], np.array([40.0, 43.0, 46.0, 49.0, 52.0]))

    def test_read_events_output_has_no_data_key(self, tmp_path):
        result = self.extractor_instance.read(events=["events"], outputPath=str(tmp_path))
        assert "data" not in result[0]

    def test_read_fiber_photometry_series_has_sampling_rate(self, tmp_path):
        result = self.extractor_instance.read(events=[self.control_event], outputPath=str(tmp_path))
        assert "sampling_rate" in result[0]
        assert result[0]["sampling_rate"] == pytest.approx(30.0)

    def test_read_fiber_photometry_series_has_npoints(self, tmp_path):
        result = self.extractor_instance.read(events=[self.control_event], outputPath=str(tmp_path))
        assert result[0]["npoints"] == 1

    def test_discover_raises_on_missing_nwb_file(self, tmp_path):
        with pytest.raises(Exception):
            NwbRecordingExtractor.discover_events_and_flags(str(tmp_path))
