"""Unit tests for BaseRecordingExtractor contract."""

import h5py
import numpy as np
import pytest

from guppy.extractors.base_recording_extractor import read_and_save_events_for_extractor
from guppy.testing.mock_recording_extractor import (
    _MOCK_DURATION_IN_SECONDS,
    _MOCK_SAMPLING_RATE,
    MockRecordingExtractor,
)

NUMBER_OF_SAMPLES = int(_MOCK_DURATION_IN_SECONDS * _MOCK_SAMPLING_RATE)

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# read_and_save_events_for_extractor unit tests
# ---------------------------------------------------------------------------


def test_single_event_produces_hdf5_file(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    read_and_save_events_for_extractor(extractor, ["mock_signal"], str(tmp_path), {"mock_signal": 0})

    assert (tmp_path / "mock_signal.hdf5").exists()


def test_single_event_hdf5_has_timestamps_dataset(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    read_and_save_events_for_extractor(extractor, ["mock_signal"], str(tmp_path), {"mock_signal": 0})

    with h5py.File(tmp_path / "mock_signal.hdf5", "r") as file:
        assert "timestamps" in file


def test_batched_events_produces_hdf5_files_for_all_events(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    events = ["mock_signal", "mock_control"]
    read_and_save_events_for_extractor(extractor, events, str(tmp_path), {event: 0 for event in events})

    assert (tmp_path / "mock_signal.hdf5").exists()
    assert (tmp_path / "mock_control.hdf5").exists()


def test_batched_events_all_hdf5_have_timestamps_dataset(tmp_path):
    extractor = MockRecordingExtractor("mock_folder")
    events = ["mock_signal", "mock_control"]
    read_and_save_events_for_extractor(extractor, events, str(tmp_path), {event: 0 for event in events})

    for event in events:
        with h5py.File(tmp_path / f"{event}.hdf5", "r") as file:
            assert "timestamps" in file


def test_batched_events_with_u34_store_id_normalizes_dtype(tmp_path):
    """np.str_ scalars (dtype <U34, produced by NWB reads) must be coerced to plain str."""
    store_id1 = np.str_("fiber_photometry_response_series_0")
    store_id2 = np.str_("fiber_photometry_response_series_1")
    assert store_id1.dtype == "<U34"
    assert store_id2.dtype == "<U34"
    extractor = MockRecordingExtractor("mock_folder")
    read_and_save_events_for_extractor(extractor, [store_id1, store_id2], str(tmp_path), {store_id1: 0, store_id2: 0})

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
