"""Unit tests for BaseRecordingExtractor contract and _write_hdf5 static method."""

import h5py
import numpy as np

from guppy.extractors.base_recording_extractor import BaseRecordingExtractor
from guppy.testing.mock_recording_extractor import MockRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# _write_hdf5 standalone unit tests
# ---------------------------------------------------------------------------


def test_write_hdf5_creates_new_file_with_numpy_array(tmp_path):
    data = np.array([1.0, 2.0, 3.0])
    BaseRecordingExtractor._write_hdf5(data, "my_store", str(tmp_path), "timestamps")

    filepath = tmp_path / "my_store.hdf5"
    assert filepath.exists()
    with h5py.File(filepath, "r") as file:
        assert "timestamps" in file
        np.testing.assert_array_equal(file["timestamps"][:], data)


def test_write_hdf5_creates_new_file_with_scalar(tmp_path):
    BaseRecordingExtractor._write_hdf5(250.0, "my_store", str(tmp_path), "sampling_rate")

    filepath = tmp_path / "my_store.hdf5"
    assert filepath.exists()
    with h5py.File(filepath, "r") as file:
        assert "sampling_rate" in file
        assert file["sampling_rate"][()] == 250.0


def test_write_hdf5_appends_second_key_to_existing_file(tmp_path):
    BaseRecordingExtractor._write_hdf5(np.array([0.0, 1.0]), "my_store", str(tmp_path), "timestamps")
    BaseRecordingExtractor._write_hdf5(100.0, "my_store", str(tmp_path), "sampling_rate")

    filepath = tmp_path / "my_store.hdf5"
    with h5py.File(filepath, "r") as file:
        assert "timestamps" in file
        assert "sampling_rate" in file


def test_write_hdf5_overwrites_existing_key_with_numpy_array(tmp_path):
    original = np.array([1.0, 2.0, 3.0])
    replacement = np.array([9.0, 8.0])
    BaseRecordingExtractor._write_hdf5(original, "my_store", str(tmp_path), "timestamps")
    BaseRecordingExtractor._write_hdf5(replacement, "my_store", str(tmp_path), "timestamps")

    filepath = tmp_path / "my_store.hdf5"
    with h5py.File(filepath, "r") as file:
        np.testing.assert_array_equal(file["timestamps"][:], replacement)


def test_write_hdf5_overwrites_existing_key_with_scalar(tmp_path):
    BaseRecordingExtractor._write_hdf5(100.0, "my_store", str(tmp_path), "sampling_rate")
    BaseRecordingExtractor._write_hdf5(500.0, "my_store", str(tmp_path), "sampling_rate")

    filepath = tmp_path / "my_store.hdf5"
    with h5py.File(filepath, "r") as file:
        assert file["sampling_rate"][()] == 500.0


def test_write_hdf5_sanitizes_backslash_in_storename(tmp_path):
    BaseRecordingExtractor._write_hdf5(np.array([1.0]), "a\\b", str(tmp_path), "timestamps")

    assert (tmp_path / "a_b.hdf5").exists()
    assert not any("\\" in str(path) for path in tmp_path.iterdir())


def test_write_hdf5_sanitizes_forward_slash_in_storename(tmp_path):
    BaseRecordingExtractor._write_hdf5(np.array([1.0]), "DI/O-1", str(tmp_path), "timestamps")

    assert (tmp_path / "DI_O-1.hdf5").exists()


# ---------------------------------------------------------------------------
# Concrete test class: MockRecordingExtractor
# ---------------------------------------------------------------------------


class TestMockRecordingExtractorContract(RecordingExtractorTestMixin):
    extractor_class = MockRecordingExtractor
    folder_path = "mock_folder"
    extractor_instance = MockRecordingExtractor(folder_path)
    expected_events = ["mock_signal", "mock_control"]
    discover_kwargs = {}
