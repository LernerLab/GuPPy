"""Unit tests for BaseRecordingExtractor contract."""

import h5py
import numpy as np
import pytest

from guppy.extractors.base_recording_extractor import (
    read_and_save_all_events,
    read_and_save_event,
)
from guppy.testing.mock_recording_extractor import (
    _MOCK_DURATION_IN_SECONDS,
    _MOCK_SAMPLING_RATE,
    MockRecordingExtractor,
)

NUMBER_OF_SAMPLES = int(_MOCK_DURATION_IN_SECONDS * _MOCK_SAMPLING_RATE)

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# read_and_save_event unit tests
# ---------------------------------------------------------------------------


def test_read_and_save_event_produces_hdf5_file(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    read_and_save_event(extractor, "mock_signal", str(tmp_path))

    assert (tmp_path / "mock_signal.hdf5").exists()


def test_read_and_save_event_hdf5_has_timestamps_dataset(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    read_and_save_event(extractor, "mock_signal", str(tmp_path))

    with h5py.File(tmp_path / "mock_signal.hdf5", "r") as file:
        assert "timestamps" in file


# ---------------------------------------------------------------------------
# read_and_save_all_events unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_read_and_save_all_events_produces_hdf5_files_for_all_events(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    event_to_extractor = {"mock_signal": extractor, "mock_control": extractor}
    read_and_save_all_events(event_to_extractor, str(tmp_path), numProcesses=2)

    assert (tmp_path / "mock_signal.hdf5").exists()
    assert (tmp_path / "mock_control.hdf5").exists()


@pytest.mark.parallel
def test_read_and_save_all_events_all_hdf5_have_timestamps_dataset(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    event_to_extractor = {"mock_signal": extractor, "mock_control": extractor}
    read_and_save_all_events(event_to_extractor, str(tmp_path), numProcesses=2)

    for event in event_to_extractor:
        with h5py.File(tmp_path / f"{event}.hdf5", "r") as file:
            assert "timestamps" in file


@pytest.mark.parallel
def test_read_and_save_all_events_with_different_extractor_instances(tmp_path):
    signal_extractor = MockRecordingExtractor("mock_folder_1")
    control_extractor = MockRecordingExtractor("mock_folder_2")
    event_to_extractor = {"mock_signal": signal_extractor, "mock_control": control_extractor}
    read_and_save_all_events(event_to_extractor, str(tmp_path), numProcesses=2)

    assert (tmp_path / "mock_signal.hdf5").exists()
    assert (tmp_path / "mock_control.hdf5").exists()
    for event in event_to_extractor:
        with h5py.File(tmp_path / f"{event}.hdf5", "r") as file:
            assert "timestamps" in file


@pytest.mark.parallel
def test_read_and_save_all_events_with_u34_storename(tmp_path):
    storename1 = np.str_("fiber_photometry_response_series_0")
    storename2 = np.str_("fiber_photometry_response_series_1")
    assert storename1.dtype == "<U34"
    assert storename2.dtype == "<U34"
    signal_extractor = MockRecordingExtractor("mock_folder_1")
    control_extractor = MockRecordingExtractor("mock_folder_2")
    event_to_extractor = {storename1: signal_extractor, storename2: control_extractor}
    read_and_save_all_events(event_to_extractor, str(tmp_path), numProcesses=2)

    assert (tmp_path / "fiber_photometry_response_series_0.hdf5").exists()
    with h5py.File(tmp_path / "fiber_photometry_response_series_0.hdf5", "r") as file:
        assert "timestamps" in file


# ---------------------------------------------------------------------------
# Concrete test class: MockRecordingExtractor
# ---------------------------------------------------------------------------


class TestBaseRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = MockRecordingExtractor
    folder_path = "mock_folder"
    extractor_instance = MockRecordingExtractor(folder_path)
    expected_events = ["mock_signal", "mock_control", "mock_ttl"]
    discover_kwargs = {}
    control_event = "mock_control"
    signal_event = "mock_signal"
    ttl_event = "mock_ttl"
    stub_ttl_test_duration_in_seconds = 0.5

    @pytest.fixture
    def expected_control_timestamps(self):
        return np.arange(NUMBER_OF_SAMPLES, dtype=float) / _MOCK_SAMPLING_RATE

    @pytest.fixture
    def expected_control_data(self):
        return np.linspace(0.0, 1.0, NUMBER_OF_SAMPLES)

    @pytest.fixture
    def expected_signal_timestamps(self):
        return np.arange(NUMBER_OF_SAMPLES, dtype=float) / _MOCK_SAMPLING_RATE

    @pytest.fixture
    def expected_signal_data(self):
        return np.linspace(0.0, 1.0, NUMBER_OF_SAMPLES)

    @pytest.fixture
    def expected_ttl_timestamps(self):
        return np.arange(NUMBER_OF_SAMPLES, dtype=float) / _MOCK_SAMPLING_RATE
