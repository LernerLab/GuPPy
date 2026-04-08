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
        np.testing.assert_allclose(timestamps, np.arange(5) / 30.0)

    def test_explicit_timestamps(self):
        series = TimeSeries(name="test", data=np.zeros(3), unit="n.a.", timestamps=[0.0, 0.1, 0.2])
        sampling_rate, timestamps = _resolve_timing(series, 3)
        assert sampling_rate == pytest.approx(10.0, rel=1e-3)
        np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2])


# ---------------------------------------------------------------------------
# Shared mixin for all NWB contract test classes
# ---------------------------------------------------------------------------


class NwbRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides shared fixtures and stub overrides for all NWB extractor test classes.

    NWB's ``stub()`` always raises ``NotImplementedError``, so all four stub tests
    from ``RecordingExtractorTestMixin`` are overridden here. All three concrete
    test classes use the same FiberPhotometryResponseSeries as control and signal,
    so those expected-value fixtures are also centralised here.
    """

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


# ---------------------------------------------------------------------------
# Contract test classes — one per TTL event type
# ---------------------------------------------------------------------------


class TestNwbRecordingExtractorEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using a plain ndx-events ``Events`` object as the TTL channel.

    Also hosts general NWB-specific tests (discover count, flags, per-type reads)
    that apply to the extractor as a whole rather than a specific TTL type.
    """

    ttl_event = "events"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # Events timestamps: 45, 46, ..., 54
        return np.arange(45, 55, dtype=np.float64)


class TestNwbExtractorAnnotatedEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using an ``AnnotatedEventsTable`` row as the TTL channel."""

    ttl_event = "AnnotatedEventsTable_Reward"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        return np.array([41.0, 42.0, 43.0, 44.0, 45.0])


class TestNwbRecordingExtractorLabeledEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using a ``LabeledEvents`` label as the TTL channel."""

    ttl_event = "labeled_events_label_1"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # label_1 is index 0: timestamps at positions where data == 0 → 40, 43, 46, 49, 52
        return np.array([40.0, 43.0, 46.0, 49.0, 52.0])
