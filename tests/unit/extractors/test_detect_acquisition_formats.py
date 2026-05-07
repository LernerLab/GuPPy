"""Unit tests for pure helper functions in detect_acquisition_formats."""

import os
import shutil

import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.extractors import NpmRecordingExtractor
from guppy.extractors.detect_acquisition_formats import (
    _classify_csv_file,
    _is_event_csv,
    _is_float,
    detect_acquisition_formats,
)

# ---------------------------------------------------------------------------
# _is_float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1.5", True),
        ("1", True),
        ("0", True),
        ("-3.14", True),
        (1.0, True),
        (0, True),
        ("abc", False),
        ("1.5.6", False),
        (None, False),
        ("", False),
    ],
)
def test_is_float(value, expected):
    assert _is_float(value) == expected


# ---------------------------------------------------------------------------
# _is_event_csv
# ---------------------------------------------------------------------------


def test_is_event_csv_single_timestamps_column(tmp_path):
    path = tmp_path / "event.csv"
    path.write_text("timestamps\n0.1\n0.2\n")
    assert _is_event_csv(str(path)) is True


def test_is_event_csv_multiple_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("timestamps,data,sampling_rate\n0.1,1.0,250\n")
    assert _is_event_csv(str(path)) is False


def test_is_event_csv_single_non_timestamps_column(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("values\n1.0\n2.0\n")
    assert _is_event_csv(str(path)) is False


# ---------------------------------------------------------------------------
# _classify_csv_file
# ---------------------------------------------------------------------------


def test_classify_csv_file_returns_doric_for_all_string_header(tmp_path):
    # Doric CSV: first 2 rows are all strings (metadata + units rows), no numeric values
    path = tmp_path / "doric.csv"
    path.write_text(
        "Time(s),AIN01xAOUT01-LockIn/Values,AIN01xAOUT03-LockIn/Values\n" "Unit,V,V\n" "0.0,0.5,0.6\n" "0.1,0.51,0.61\n"
    )
    assert _classify_csv_file(str(path)) == "doric"


def test_classify_csv_file_returns_csv_for_single_timestamps_column(tmp_path):
    path = tmp_path / "event.csv"
    path.write_text("timestamps\n0.1\n0.2\n")
    assert _classify_csv_file(str(path)) == "csv"


def test_classify_csv_file_returns_csv_for_three_column_data_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("timestamps,data,sampling_rate\n0.1,1.0,250\n0.2,1.1,250\n")
    assert _classify_csv_file(str(path)) == "csv"


def test_classify_csv_file_returns_npm_for_multicolumn_data(tmp_path):
    path = tmp_path / "npm.csv"
    path.write_text("FrameCounter,Timestamp,LedState,Region0G,Region1G\n" "1,0.1,1,100,200\n" "2,0.2,2,101,201\n")
    assert _classify_csv_file(str(path)) == "npm"


def test_classify_csv_file_raises_for_single_column_with_wrong_name(tmp_path):
    path = tmp_path / "wrong_name.csv"
    path.write_text("not_timestamps\n0.1\n0.2\n")
    with pytest.raises(ValueError, match=r"requires the column to be named 'timestamps'"):
        _classify_csv_file(str(path))


def test_classify_csv_file_raises_for_three_columns_with_wrong_names(tmp_path):
    path = tmp_path / "wrong_columns.csv"
    path.write_text("foo,bar,baz\n0.1,1.0,250\n0.2,1.1,250\n")
    with pytest.raises(ValueError, match=r"requires column names 'timestamps', 'data', 'sampling_rate'"):
        _classify_csv_file(str(path))


def test_classify_csv_file_raises_for_headerless_single_numeric_column(tmp_path):
    """A headerless single-numeric-column CSV triggers the unrecognized-layout branch."""
    path = tmp_path / "headerless.csv"
    path.write_text("0\n1\n2\n")
    with pytest.raises(ValueError, match=r"requires the column to be named 'timestamps'"):
        _classify_csv_file(str(path))


# ---------------------------------------------------------------------------
# detect_acquisition_formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session_subdir, expected_formats",
    [
        ("tdt/Photo_63_207-181030-103332", {"tdt"}),
        ("doric/sample_doric_1", {"doric"}),
        ("csv/sample_data_csv_1", {"csv"}),
        ("npm/sampleData_NPM_1", {"npm"}),
    ],
    ids=["tdt", "doric", "csv", "npm"],
)
def test_detect_acquisition_formats(session_subdir, expected_formats):
    folder_path = os.path.join(STUBBED_TESTING_DATA, session_subdir)
    assert detect_acquisition_formats(folder_path) == expected_formats


@pytest.mark.parametrize(
    "session_subdir, expected_formats",
    [
        ("tdt/Photo_63_207-181030-103332", {"tdt", "csv"}),
        ("doric/sample_doric_1", {"doric", "csv"}),
        # External CSV event file named without "event" prefix is NOT suppressed by NPM logic
        ("npm/sampleData_NPM_1", {"npm", "csv"}),
        # Adding another event CSV to a CSV session leaves the result unchanged
        ("csv/sample_data_csv_1", {"csv"}),
    ],
    ids=["tdt_csv", "doric_csv", "npm_csv", "csv_csv"],
)
def test_detect_acquisition_formats_with_external_csv_events(tmp_path, session_subdir, expected_formats):
    src = os.path.join(STUBBED_TESTING_DATA, session_subdir)
    session_copy = tmp_path / os.path.basename(session_subdir)
    shutil.copytree(src, session_copy)
    (session_copy / "port_entries.csv").write_text("timestamps\n0.1\n0.2\n0.3\n")
    assert detect_acquisition_formats(str(session_copy)) == expected_formats


def test_detect_acquisition_formats_after_npm_split_events(tmp_path):
    # When an NPM session has split-event TTLs, NpmRecordingExtractor.discover_events_and_flags
    # writes intermediate single-column "event*.csv" files that must subsequently be read by
    # CsvRecordingExtractor. detect_acquisition_formats must therefore report both "npm" and
    # "csv" so the read path dispatches CsvRecordingExtractor for those events.
    src = os.path.join(STUBBED_TESTING_DATA, "npm/sampleData_NPM_4")
    session_copy = tmp_path / "sampleData_NPM_4"
    shutil.copytree(src, session_copy)

    inputParameters = {
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
        "npm_split_events": [True, True],
    }
    NpmRecordingExtractor.discover_events_and_flags(
        folder_path=str(session_copy),
        num_ch=2,
        inputParameters=inputParameters,
    )

    assert detect_acquisition_formats(str(session_copy)) == {"npm", "csv"}
