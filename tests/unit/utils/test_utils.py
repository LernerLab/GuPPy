import pandas as pd

from guppy.utils.utils import get_all_stores_for_combining_data, read_Df, takeOnlyDirs

# ── takeOnlyDirs ──────────────────────────────────────────────────────────────


def test_take_only_dirs_returns_only_directories(tmp_path):
    directory = tmp_path / "a_dir"
    directory.mkdir()
    file = tmp_path / "a_file.txt"
    file.touch()

    result = takeOnlyDirs([str(directory), str(file)])

    assert set(result) == {str(directory)}


def test_take_only_dirs_all_directories_returns_all(tmp_path):
    dir_one = tmp_path / "dir_one"
    dir_two = tmp_path / "dir_two"
    dir_one.mkdir()
    dir_two.mkdir()

    result = takeOnlyDirs([str(dir_one), str(dir_two)])

    assert set(result) == {str(dir_one), str(dir_two)}


def test_take_only_dirs_all_files_returns_empty(tmp_path):
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.touch()
    file_two.touch()

    result = takeOnlyDirs([str(file_one), str(file_two)])

    assert result == []


def test_take_only_dirs_empty_input_returns_empty():
    result = takeOnlyDirs([])

    assert result == []


# ── get_all_stores_for_combining_data ─────────────────────────────────────────


def test_get_all_stores_for_combining_data_single_folder_one_group():
    result = get_all_stores_for_combining_data(["session_output_0"])

    assert len(result) == 1
    assert result[0] == ["session_output_0"]


def test_get_all_stores_for_combining_data_two_indices_two_groups():
    folder_names = ["session_output_0", "session_output_1"]
    result = get_all_stores_for_combining_data(folder_names)

    assert len(result) == 2
    assert ["session_output_0"] in result
    assert ["session_output_1"] in result


def test_get_all_stores_for_combining_data_multiple_folders_same_index_sorted():
    # Both match _output_0; sorted case-insensitively → "Apple_output_0" < "banana_output_0"
    folder_names = ["banana_output_0", "Apple_output_0"]
    result = get_all_stores_for_combining_data(folder_names)

    assert len(result) == 1
    assert result[0] == ["Apple_output_0", "banana_output_0"]


def test_get_all_stores_for_combining_data_no_match_returns_empty():
    result = get_all_stores_for_combining_data(["no_pattern_here", "also_nothing"])

    assert result == []


# ── read_Df ───────────────────────────────────────────────────────────────────


def test_read_df_roundtrip_returns_same_dataframe(tmp_path):
    dataframe = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    dataframe.to_hdf(tmp_path / "my_event.h5", key="df", mode="w")

    result = read_Df(str(tmp_path), "my_event", name=None)

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_df_with_name_constructs_event_name_filename(tmp_path):
    dataframe = pd.DataFrame({"signal": [4.0, 5.0]})
    dataframe.to_hdf(tmp_path / "lever_press_timestamps.h5", key="df", mode="w")

    result = read_Df(str(tmp_path), "lever_press", name="timestamps")

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_df_without_name_constructs_event_only_filename(tmp_path):
    dataframe = pd.DataFrame({"x": [10.0]})
    dataframe.to_hdf(tmp_path / "event_only.h5", key="df", mode="w")

    result = read_Df(str(tmp_path), "event_only", name=None)

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_df_backslash_in_event_name_is_replaced_with_underscore(tmp_path):
    # Event "a\\b" → filename "a_b.h5"
    dataframe = pd.DataFrame({"y": [7.0, 8.0]})
    dataframe.to_hdf(tmp_path / "a_b.h5", key="df", mode="w")

    result = read_Df(str(tmp_path), "a\\b", name=None)

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_df_forward_slash_in_event_name_is_replaced_with_underscore(tmp_path):
    # Event "a/b" → filename "a_b.h5"
    dataframe = pd.DataFrame({"z": [9.0]})
    dataframe.to_hdf(tmp_path / "a_b.h5", key="df", mode="w")

    result = read_Df(str(tmp_path), "a/b", name=None)

    pd.testing.assert_frame_equal(result, dataframe)
