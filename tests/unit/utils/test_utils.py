import pandas as pd
import pytest

from guppy.utils.utils import (
    NPM_PARAM_KEYS,
    discover_run_folders,
    get_all_stores_for_combining_data,
    load_npm_params,
    parse_run_name,
    read_Df,
    run_folder_for_run,
    select_run_folders,
    takeOnlyDirs,
    validate_run_name,
    write_npm_params,
)

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


# ── parse_run_name ────────────────────────────────────────────────────────────


def test_parse_run_name_legacy_integer_suffix():
    assert parse_run_name("/sessions/mySession/mySession_output_1") == "1"


def test_parse_run_name_alphanumeric_suffix():
    assert parse_run_name("/sessions/mySession/mySession_output_baseline50") == "baseline50"


def test_parse_run_name_uses_last_marker_occurrence():
    # If a session basename happens to embed "_output_", the last occurrence is the boundary.
    assert parse_run_name("/path/foo_output_data/foo_output_data_output_strict") == "strict"


def test_parse_run_name_trailing_separator_is_tolerated():
    assert parse_run_name("/sessions/mySession/mySession_output_1/") == "1"


def test_parse_run_name_raises_when_marker_missing():
    with pytest.raises(ValueError, match="does not match"):
        parse_run_name("/sessions/no_match_here")


# ── run_folder_for_run ────────────────────────────────────────────────────────


def test_output_dir_for_run_builds_expected_path(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    result = run_folder_for_run(str(session), "baseline")
    assert result == str(session / "mySession_output_baseline")


def test_output_dir_for_run_does_not_create_directory(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    result = run_folder_for_run(str(session), "x")
    import os as _os

    assert not _os.path.exists(result)


# ── discover_run_folders ──────────────────────────────────────────────────────


def test_discover_output_dirs_returns_only_output_dirs(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_1").mkdir()
    (session / "mySession_output_2").mkdir()
    (session / "unrelated.txt").touch()
    (session / "raw_data").mkdir()

    result = discover_run_folders(str(session))

    assert result == [
        str(session / "mySession_output_1"),
        str(session / "mySession_output_2"),
    ]


def test_discover_output_dirs_orders_numeric_then_alpha(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_10").mkdir()
    (session / "mySession_output_2").mkdir()
    (session / "mySession_output_baseline").mkdir()
    (session / "mySession_output_alpha").mkdir()

    result = [parse_run_name(directory) for directory in discover_run_folders(str(session))]

    assert result == ["2", "10", "alpha", "baseline"]


def test_discover_output_dirs_empty_when_no_outputs(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()

    assert discover_run_folders(str(session)) == []


# ── select_run_folders ────────────────────────────────────────────────────────


def test_select_output_dirs_none_raises(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_1").mkdir()

    with pytest.raises(ValueError, match="explicit non-empty list"):
        select_run_folders(str(session), None)


def test_select_output_dirs_filters_to_requested_runs(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    for run_name in ("1", "baseline", "strict"):
        directory = session / f"mySession_output_{run_name}"
        directory.mkdir()
        (directory / "storesList.csv").touch()

    result = select_run_folders(str(session), ["baseline"])

    assert result == [str(session / "mySession_output_baseline")]


def test_select_output_dirs_raises_for_missing_run_name(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_1").mkdir()
    (session / "mySession_output_1" / "storesList.csv").touch()

    with pytest.raises(ValueError, match="Output directory not found"):
        select_run_folders(str(session), ["nonexistent"])


def test_select_output_dirs_raises_when_storeslist_missing(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_baseline").mkdir()  # no storesList.csv

    with pytest.raises(ValueError, match="storesList.csv"):
        select_run_folders(str(session), ["baseline"])


def test_select_output_dirs_empty_list_raises(tmp_path):
    session = tmp_path / "mySession"
    session.mkdir()
    (session / "mySession_output_1").mkdir()

    with pytest.raises(ValueError, match="explicit non-empty list"):
        select_run_folders(str(session), [])


# ── validate_run_name ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("run_name", ["baseline", "1", "run-2", "v2.0", "alpha_beta"])
def test_validate_run_name_accepts_valid_names(run_name):
    validate_run_name(run_name)  # should not raise


@pytest.mark.parametrize(
    "run_name, match",
    [
        ("", "non-empty"),
        ("   ", "whitespace"),
        (" leading", "whitespace"),
        ("trailing ", "whitespace"),
        ("a/b", "forbidden character"),
        ("a\\b", "forbidden character"),
        ("a:b", "forbidden character"),
        ("a..b", "'..'"),
        ("foo_output_bar", "_output_"),
    ],
)
def test_validate_run_name_rejects_invalid(run_name, match):
    with pytest.raises(ValueError, match=match):
        validate_run_name(run_name)


def test_validate_run_name_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        validate_run_name(123)


# ── get_all_stores_for_combining_data — new name-based grouping ───────────────


def test_get_all_stores_for_combining_data_groups_by_alphanumeric_run_name():
    folder_names = [
        "/data/sessionA/sessionA_output_baseline",
        "/data/sessionB/sessionB_output_baseline",
        "/data/sessionA/sessionA_output_strict",
    ]

    result = get_all_stores_for_combining_data(folder_names)

    assert result == [
        ["/data/sessionA/sessionA_output_baseline", "/data/sessionB/sessionB_output_baseline"],
        ["/data/sessionA/sessionA_output_strict"],
    ]


def test_get_all_stores_for_combining_data_orders_numeric_before_alpha():
    folder_names = [
        "/d/x/x_output_baseline",
        "/d/y/y_output_2",
        "/d/x/x_output_2",
        "/d/y/y_output_baseline",
    ]

    result = get_all_stores_for_combining_data(folder_names)

    # Numeric "2" group first (sorted by integer value), then alphanumeric "baseline".
    assert result[0] == ["/d/x/x_output_2", "/d/y/y_output_2"]
    assert result[1] == ["/d/x/x_output_baseline", "/d/y/y_output_baseline"]


def test_get_all_stores_for_combining_data_skips_paths_without_marker():
    folder_names = ["/data/no_match", "/data/x/x_output_1"]
    result = get_all_stores_for_combining_data(folder_names)
    assert result == [["/data/x/x_output_1"]]


# ── write_npm_params / load_npm_params ────────────────────────────────────────


def test_write_then_load_npm_params_round_trips(tmp_path):
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    input_parameters = {
        "npm_split_events": [True, False],
        "npm_time_units": ["milliseconds", "seconds"],
        "npm_timestamp_column_names": ["ComputerTimestamp", None],
        "unrelated_key": "ignored",
    }

    write_npm_params(
        run_folder=str(run_folder),
        npm_params={key: input_parameters[key] for key in NPM_PARAM_KEYS},
    )

    assert load_npm_params(str(run_folder)) == {
        "npm_split_events": [True, False],
        "npm_time_units": ["milliseconds", "seconds"],
        "npm_timestamp_column_names": ["ComputerTimestamp", None],
    }


def test_load_npm_params_returns_empty_when_file_absent(tmp_path):
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    assert load_npm_params(str(run_folder)) == {}
