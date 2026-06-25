import os

import pandas as pd
import pytest

from guppy.extractors.csv_recording_extractor import CsvRecordingExtractor
from guppy.utils.custom_events import (
    is_sorted,
    parse_pasted_timestamps,
    write_custom_event_csv,
)


class TestParsePastedTimestamps:
    def test_newline_separated(self):
        assert parse_pasted_timestamps("0.1\n0.2\n0.3") == [0.1, 0.2, 0.3]

    def test_comma_separated(self):
        assert parse_pasted_timestamps("0.1, 0.2, 0.3") == [0.1, 0.2, 0.3]

    def test_mixed_whitespace_and_trailing_blanks(self):
        assert parse_pasted_timestamps("1\t2\n3 \n\n4\n") == [1.0, 2.0, 3.0, 4.0]

    def test_empty_string_yields_empty_list(self):
        assert parse_pasted_timestamps("   \n  ") == []

    def test_order_is_preserved_not_sorted(self):
        assert parse_pasted_timestamps("3\n1\n2") == [3.0, 1.0, 2.0]

    def test_non_numeric_token_raises_naming_token(self):
        with pytest.raises(ValueError, match="'abc'"):
            parse_pasted_timestamps("0.1\nabc\n0.3")


class TestIsSorted:
    def test_sorted(self):
        assert is_sorted([0.1, 0.2, 0.2, 0.3]) is True

    def test_unsorted(self):
        assert is_sorted([0.1, 0.3, 0.2]) is False

    def test_empty_and_single(self):
        assert is_sorted([]) is True
        assert is_sorted([5.0]) is True


class TestWriteCustomEventCsv:
    def test_writes_header_and_rows(self, tmp_path):
        path = write_custom_event_csv(name="movement_onset", timestamps=[0.5, 1.5, 2.5], folder_path=str(tmp_path))
        assert path == os.path.join(str(tmp_path), "movement_onset.csv")
        df = pd.read_csv(path)
        assert list(df.columns) == ["timestamps"]
        assert df["timestamps"].tolist() == [0.5, 1.5, 2.5]

    def test_raises_when_exists_and_not_overwrite(self, tmp_path):
        write_custom_event_csv(name="ev", timestamps=[1.0], folder_path=str(tmp_path))
        with pytest.raises(FileExistsError, match="ev"):
            write_custom_event_csv(name="ev", timestamps=[2.0], folder_path=str(tmp_path))

    def test_overwrites_when_requested(self, tmp_path):
        write_custom_event_csv(name="ev", timestamps=[1.0], folder_path=str(tmp_path))
        write_custom_event_csv(name="ev", timestamps=[9.0, 8.0], folder_path=str(tmp_path), overwrite=True)
        df = pd.read_csv(os.path.join(str(tmp_path), "ev.csv"))
        assert df["timestamps"].tolist() == [9.0, 8.0]

    def test_empty_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            write_custom_event_csv(name="  ", timestamps=[1.0], folder_path=str(tmp_path))

    def test_name_with_path_separator_raises(self, tmp_path):
        with pytest.raises(ValueError, match="path separator"):
            write_custom_event_csv(name="a/b", timestamps=[1.0], folder_path=str(tmp_path))

    def test_written_csv_is_discovered_as_event_store(self, tmp_path):
        write_custom_event_csv(name="lever_press", timestamps=[0.1, 0.2], folder_path=str(tmp_path))
        events, flags = CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))
        assert events == ["lever_press"]
        assert flags == ["event_csv"]
