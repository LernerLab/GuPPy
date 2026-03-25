import numpy as np
import pytest

from guppy.analysis.io_utils import (
    decide_naming_convention,
    find_files,
    read_hdf5,
    write_hdf5,
)

# ── find_files ────────────────────────────────────────────────────────────────


def test_find_files_matches_glob_pattern(tmp_path):
    (tmp_path / "control_signal.hdf5").touch()
    (tmp_path / "signal_data.hdf5").touch()
    (tmp_path / "other.txt").touch()
    result = find_files(str(tmp_path), "control_*")
    assert len(result) == 1
    assert "control_signal.hdf5" in result[0]


def test_find_files_ignore_case_matches_regardless_of_case(tmp_path):
    (tmp_path / "Control_Data.hdf5").touch()
    result = find_files(str(tmp_path), "control_*", ignore_case=True)
    assert len(result) == 1


def test_find_files_returns_empty_list_when_no_match(tmp_path):
    (tmp_path / "other.txt").touch()
    result = find_files(str(tmp_path), "control_*")
    assert result == []


# ── decide_naming_convention ──────────────────────────────────────────────────


def test_decide_naming_convention_one_control_one_signal_returns_2x1_array(tmp_path):
    (tmp_path / "control_DMS.hdf5").touch()
    (tmp_path / "signal_DMS.hdf5").touch()
    result = decide_naming_convention(str(tmp_path))
    assert result.shape == (2, 1)


def test_decide_naming_convention_two_pairs_returns_2x2_array(tmp_path):
    for name in ["control_DMS", "signal_DMS", "control_NAc", "signal_NAc"]:
        (tmp_path / f"{name}.hdf5").touch()
    result = decide_naming_convention(str(tmp_path))
    assert result.shape == (2, 2)


def test_decide_naming_convention_unequal_counts_raises(tmp_path):
    (tmp_path / "control_DMS.hdf5").touch()
    (tmp_path / "control_NAc.hdf5").touch()
    (tmp_path / "signal_DMS.hdf5").touch()
    with pytest.raises(Exception):
        decide_naming_convention(str(tmp_path))


# ── read_hdf5 / write_hdf5 ────────────────────────────────────────────────────


def test_write_and_read_hdf5_roundtrip_numpy_array(tmp_path):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    write_hdf5(data, "my_event", str(tmp_path), "timestamps")
    result = read_hdf5("my_event", str(tmp_path), "timestamps")
    np.testing.assert_array_equal(result, data)


def test_write_hdf5_creates_file(tmp_path):
    data = np.ones(10)
    write_hdf5(data, "test_event", str(tmp_path), "data")
    assert (tmp_path / "test_event.hdf5").exists()


def test_write_hdf5_overwrites_existing_key_with_new_array(tmp_path):
    original = np.array([1.0, 2.0, 3.0])
    updated = np.array([9.0, 8.0, 7.0])
    write_hdf5(original, "event", str(tmp_path), "data")
    write_hdf5(updated, "event", str(tmp_path), "data")
    result = read_hdf5("event", str(tmp_path), "data")
    np.testing.assert_array_equal(result, updated)


def test_write_hdf5_adds_second_key_to_existing_file(tmp_path):
    data_a = np.array([1.0, 2.0])
    data_b = np.array([3.0, 4.0])
    write_hdf5(data_a, "event", str(tmp_path), "timestamps")
    write_hdf5(data_b, "event", str(tmp_path), "data")
    result_a = read_hdf5("event", str(tmp_path), "timestamps")
    result_b = read_hdf5("event", str(tmp_path), "data")
    np.testing.assert_array_equal(result_a, data_a)
    np.testing.assert_array_equal(result_b, data_b)


def test_read_hdf5_nonexistent_file_raises(tmp_path):
    with pytest.raises(Exception):
        read_hdf5("nonexistent_event", str(tmp_path), "data")


def test_write_hdf5_sanitizes_backslash_in_event_name(tmp_path):
    data = np.array([1.0, 2.0])
    write_hdf5(data, "DI\\O1", str(tmp_path), "timestamps")
    # Should create "DI_O1.hdf5" (backslash → underscore)
    assert (tmp_path / "DI_O1.hdf5").exists()


def test_write_hdf5_sanitizes_forward_slash_in_event_name(tmp_path):
    data = np.array([1.0, 2.0])
    write_hdf5(data, "CAM1/EXC1", str(tmp_path), "timestamps")
    # Should create "CAM1_EXC1.hdf5" (forward slash → underscore)
    assert (tmp_path / "CAM1_EXC1.hdf5").exists()
