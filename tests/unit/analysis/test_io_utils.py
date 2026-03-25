import numpy as np
import pytest

from guppy.analysis.io_utils import (
    check_storeslistfile,
    check_TDT,
    decide_naming_convention,
    fetchCoords,
    find_files,
    get_control_and_signal_channel_names,
    get_coords,
    make_dir_for_cross_correlation,
    makeAverageDir,
    read_hdf5,
    write_combined_stores_list,
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


# ── check_TDT ─────────────────────────────────────────────────────────────────


def test_check_TDT_returns_true_when_tsq_file_present(tmp_path):
    (tmp_path / "session.tsq").touch()
    assert check_TDT(str(tmp_path)) is True


def test_check_TDT_returns_false_when_no_tsq_file(tmp_path):
    (tmp_path / "session.hdf5").touch()
    assert check_TDT(str(tmp_path)) is False


# ── fetchCoords ───────────────────────────────────────────────────────────────


def test_fetch_coords_returns_default_range_when_no_file(tmp_path):
    data = np.array([0.0, 1.0, 2.0, 5.0])
    result = fetchCoords(str(tmp_path), "dms", data)
    # No file: coords = [0, data[-1]] = [0, 5.0], reshaped to [[0, 5.0]]
    np.testing.assert_array_equal(result, np.array([[0, 5.0]]))


def test_fetch_coords_reads_npy_file_and_returns_first_column_pairs(tmp_path):
    # File has shape (2, 2); [:, 0] = [1.0, 3.0]; reshape(-1, 2) = [[1.0, 3.0]]
    npy_data = np.array([[1.0, 99.0], [3.0, 99.0]])
    np.save(tmp_path / "coordsForPreProcessing_dms.npy", npy_data)
    result = fetchCoords(str(tmp_path), "dms", np.array([0.0, 10.0]))
    np.testing.assert_array_equal(result, np.array([[1.0, 3.0]]))


# ── get_coords ────────────────────────────────────────────────────────────────


def test_get_coords_without_artifact_removal_returns_full_range_with_margin(tmp_path):
    tsNew = np.array([0.0, 1.0, 2.0, 3.0])
    result = get_coords(str(tmp_path), "dms", tsNew, removeArtifacts=False)
    # dt = 1.0; range = [[0.0 - 1.0, 3.0 + 1.0]] = [[-1.0, 4.0]]
    np.testing.assert_array_equal(result, np.array([[-1.0, 4.0]]))


def test_get_coords_with_artifact_removal_delegates_to_fetch_coords(tmp_path):
    tsNew = np.array([0.0, 1.0, 2.0, 5.0])
    result = get_coords(str(tmp_path), "dms", tsNew, removeArtifacts=True)
    # No coords file: default = [[0, tsNew[-1]]] = [[0, 5.0]]
    np.testing.assert_array_equal(result, np.array([[0, 5.0]]))


# ── check_storeslistfile ──────────────────────────────────────────────────────


def test_check_storeslistfile_reads_stores_list_from_output_subdirectory(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    output_dir = session_dir / "session_output_1"
    output_dir.mkdir()
    stores_list = np.array([["sig0", "ctrl0"], ["signal_dms", "control_dms"]])
    np.savetxt(output_dir / "storesList.csv", stores_list, fmt="%s", delimiter=",")
    result = check_storeslistfile([str(session_dir)])
    # np.unique sorts columns; "ctrl0"/"control_dms" < "sig0"/"signal_dms"
    np.testing.assert_array_equal(result, np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]]))


# ── write_combined_stores_list ────────────────────────────────────────────────


def test_write_combined_stores_list_creates_csv_in_each_output_path(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stores_list = np.array([["sig0"], ["signal_dms"]])
    write_combined_stores_list([[str(output_dir)]], stores_list)
    result = np.genfromtxt(output_dir / "combine_storesList.csv", dtype="str", delimiter=",").reshape(2, -1)
    np.testing.assert_array_equal(result, stores_list)


# ── get_control_and_signal_channel_names ──────────────────────────────────────


def test_get_control_and_signal_channel_names_filters_and_sorts_channels():
    stores_list = np.array([["sig0", "ctrl0", "event0"], ["signal_dms", "control_dms", "TTL1"]])
    result = get_control_and_signal_channel_names(stores_list)
    # Filters to ["signal_dms", "control_dms"], sorts to ["control_dms", "signal_dms"],
    # reshapes to (2, 1)
    np.testing.assert_array_equal(result, np.array([["control_dms"], ["signal_dms"]]))


def test_get_control_and_signal_channel_names_raises_for_odd_count():
    # Three control/signal entries cannot reshape to (2, -1)
    stores_list = np.array([["s0", "c0", "c1"], ["signal_dms", "control_dms", "control_nac"]])
    with pytest.raises(Exception):
        get_control_and_signal_channel_names(stores_list)


# ── make_dir_for_cross_correlation ────────────────────────────────────────────


def test_make_dir_for_cross_correlation_creates_directory_and_returns_path(tmp_path):
    result = make_dir_for_cross_correlation(str(tmp_path))
    assert result == str(tmp_path / "cross_correlation_output")
    assert (tmp_path / "cross_correlation_output").is_dir()


def test_make_dir_for_cross_correlation_is_idempotent(tmp_path):
    make_dir_for_cross_correlation(str(tmp_path))
    result = make_dir_for_cross_correlation(str(tmp_path))
    assert result == str(tmp_path / "cross_correlation_output")


# ── makeAverageDir ────────────────────────────────────────────────────────────


def test_make_average_dir_creates_directory_and_returns_path(tmp_path):
    result = makeAverageDir(str(tmp_path))
    assert result == str(tmp_path / "average")
    assert (tmp_path / "average").is_dir()


def test_make_average_dir_is_idempotent(tmp_path):
    makeAverageDir(str(tmp_path))
    result = makeAverageDir(str(tmp_path))
    assert result == str(tmp_path / "average")
