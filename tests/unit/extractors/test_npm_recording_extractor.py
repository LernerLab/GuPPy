"""Contract tests for NpmRecordingExtractor."""

import os

import numpy as np
import pandas as pd
import pytest

from guppy.extractors.npm_recording_extractor import NpmRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# check_channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state, expected_num_channels",
    [
        # 1 unique value in state[2:12]
        ([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1),
        # 2 unique values in state[2:12]
        ([0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 2),
        # 3 unique values in state[2:12]
        ([0, 0, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], 3),
    ],
)
def test_check_channels_returns_correct_num_channels(state, expected_num_channels):
    state_array = np.array(state)
    num_channels, unique_channels = NpmRecordingExtractor.check_channels(state_array)
    assert num_channels == expected_num_channels
    assert len(unique_channels) == expected_num_channels


def test_check_channels_raises_for_more_than_three_channels():
    state = np.array([0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
    with pytest.raises(Exception):
        NpmRecordingExtractor.check_channels(state)


# ---------------------------------------------------------------------------
# decide_indices
# ---------------------------------------------------------------------------


def test_decide_indices_data_np_flag_partitions_rows_by_channel():
    dataframe = pd.DataFrame({"col1": range(6), "col2": range(6, 12)})
    file = "file0_"
    result_df, indices_dict, num_channels = NpmRecordingExtractor.decide_indices(file, dataframe, "data_np", num_ch=2)
    np.testing.assert_array_equal(indices_dict["file0_chev"], [0, 2, 4])
    np.testing.assert_array_equal(indices_dict["file0_chod"], [1, 3, 5])
    assert num_channels == 2


# ---------------------------------------------------------------------------
# _update_df_with_timestamp_columns
# ---------------------------------------------------------------------------


def test_update_df_with_timestamp_columns_single_timestamp_column_unchanged():
    # Only one timestamp column → function returns df unchanged (no insertion)
    dataframe = pd.DataFrame({"FrameCounter": [1, 2], "Timestamp_ms": [0.1, 0.2], "Values": [10, 20]})
    result = NpmRecordingExtractor._update_df_with_timestamp_columns(dataframe, None)
    assert list(result.columns) == ["FrameCounter", "Timestamp_ms", "Values"]


def test_update_df_with_timestamp_columns_multiple_timestamps_uses_first_by_default():
    # Multiple timestamp columns → function inserts canonical "Timestamp" from the first one and drops both originals
    dataframe = pd.DataFrame(
        {
            "FrameCounter": [1, 2],
            "Timestamp_ms": [0.1, 0.2],
            "Timestamp_s": [0.0001, 0.0002],
            "Values": [10, 20],
        }
    )
    result = NpmRecordingExtractor._update_df_with_timestamp_columns(dataframe, None)
    assert "Timestamp" in result.columns
    assert "Timestamp_ms" not in result.columns
    assert "Timestamp_s" not in result.columns


def test_update_df_with_timestamp_columns_explicit_column_name_used():
    dataframe = pd.DataFrame(
        {
            "FrameCounter": [1, 2],
            "Timestamp_ms": [0.1, 0.2],
            "Timestamp_s": [0.0001, 0.0002],
            "Values": [10, 20],
        }
    )
    result = NpmRecordingExtractor._update_df_with_timestamp_columns(dataframe, "Timestamp_s")
    assert "Timestamp" in result.columns
    assert "Timestamp_s" not in result.columns
    np.testing.assert_array_equal(result["Timestamp"].to_numpy(), [0.0001, 0.0002])


# ---------------------------------------------------------------------------
# has_multiple_event_ttls
# ---------------------------------------------------------------------------


def test_has_multiple_event_ttls_data_file_returns_false(tmp_path):
    # Multi-column file → classified as data_np, not an event file → False
    dataframe = pd.DataFrame({"FrameCounter": [1, 2], "LedState": [1, 2], "Signal": [0.1, 0.2]})
    dataframe.to_csv(tmp_path / "data.csv", index=False)
    result = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path))
    assert result == [False]


def test_has_multiple_event_ttls_single_ttl_event_file_returns_false(tmp_path):
    # 2-column event file with a single unique TTL value → False
    dataframe = pd.DataFrame({"timestamp": [0.1, 0.2, 0.3], "value": [1, 1, 1]})
    dataframe.to_csv(tmp_path / "stimuli.csv", index=False)
    result = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path))
    assert result == [False]


def test_has_multiple_event_ttls_multiple_ttl_event_file_returns_true(tmp_path):
    # 2-column event file with multiple unique TTL values → True
    dataframe = pd.DataFrame({"timestamp": [0.1, 0.2, 0.3], "value": [1, 3, 1]})
    dataframe.to_csv(tmp_path / "stimuli.csv", index=False)
    result = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path))
    assert result == [True]


# ---------------------------------------------------------------------------
# needs_ts_unit
# ---------------------------------------------------------------------------


def test_needs_ts_unit_event_file_returns_false(tmp_path):
    # 2-column event file → classified as event_np and skipped → False
    dataframe = pd.DataFrame({"timestamp": [0.1, 0.2], "value": [1, 1]})
    dataframe.to_csv(tmp_path / "stimuli.csv", index=False)
    ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(folder_path=str(tmp_path), num_ch=2)
    assert ts_unit_needs == [False]
    assert col_names_ts == [""]


def test_needs_ts_unit_single_timestamp_column_returns_false(tmp_path):
    # data_np_v2 file with only one timestamp column → does not exceed threshold → False
    dataframe = pd.DataFrame(
        {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Signal": [0.5, 0.6]}
    )
    dataframe.to_csv(tmp_path / "data.csv", index=False)
    ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(folder_path=str(tmp_path), num_ch=2)
    assert ts_unit_needs == [False]
    assert col_names_ts == ["", "Timestamp"]


def test_needs_ts_unit_multiple_timestamp_columns_returns_true(tmp_path):
    # data_np_v2 file with two timestamp columns → exceeds threshold → True
    dataframe = pd.DataFrame(
        {
            "FrameCounter": [1, 2],
            "SystemTimestamp": [0.1, 0.2],
            "LedState": [1, 2],
            "ComputerTimestamp": [0.001, 0.002],
            "Signal": [0.5, 0.6],
        }
    )
    dataframe.to_csv(tmp_path / "data.csv", index=False)
    ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(folder_path=str(tmp_path), num_ch=2)
    assert ts_unit_needs == [True]
    assert col_names_ts == ["", "SystemTimestamp", "ComputerTimestamp"]


_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestNpmRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Neurophotometrics", "sampleData_NPM_1")
    extractor_instance = NpmRecordingExtractor(folder_path)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    control_event = "file0_chod1"
    signal_event = "file0_chev1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0

    @pytest.fixture
    def expected_control_timestamps(self):
        # discover must run first to create the intermediate CSV files that read() depends on
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chod1"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chod1"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chev1"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chev1"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["event0"], outputPath="")
        return result[0]["timestamps"]
