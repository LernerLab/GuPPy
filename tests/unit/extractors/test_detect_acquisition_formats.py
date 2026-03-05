"""Unit tests for pure helper functions in detect_acquisition_formats."""

import pytest

from guppy.extractors.detect_acquisition_formats import (
    _classify_csv_file,
    _is_event_csv,
    _is_float,
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
