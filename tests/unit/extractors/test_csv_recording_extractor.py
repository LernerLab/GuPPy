"""Contract tests for CsvRecordingExtractor."""

import os

import pandas as pd
import pytest

from guppy.extractors.csv_recording_extractor import CsvRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# _check_header
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "columns, expected_floats",
    [
        (["0", "1", "2"], [0.0, 1.0, 2.0]),  # all numeric headers
        (["timestamps", "data", "sampling_rate"], []),  # all string headers
        (["0", "timestamps", "2"], [0.0, 2.0]),  # mixed headers
    ],
)
def test_check_header_returns_all_columns_and_numeric_conversions(columns, expected_floats):
    dataframe = pd.DataFrame(columns=columns)
    all_columns, float_conversions = CsvRecordingExtractor._check_header(dataframe)
    assert all_columns == columns
    assert float_conversions == expected_floats


_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestCsvRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = CsvRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_csv", "sample_data_csv_1")
    extractor_instance = CsvRecordingExtractor(folder_path)
    expected_events = ["Sample_Control_Channel", "Sample_Signal_Channel", "Sample_TTL"]
    discover_kwargs = {}
    control_event = "Sample_Control_Channel"
    signal_event = "Sample_Signal_Channel"
    ttl_event = "Sample_TTL"

    @property
    def expected_control_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_Control_Channel.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()

    @property
    def expected_control_data(self):
        csv_path = os.path.join(self.folder_path, "Sample_Control_Channel.csv")
        return pd.read_csv(csv_path)["data"].to_numpy()

    @property
    def expected_signal_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_Signal_Channel.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()

    @property
    def expected_signal_data(self):
        csv_path = os.path.join(self.folder_path, "Sample_Signal_Channel.csv")
        return pd.read_csv(csv_path)["data"].to_numpy()

    @property
    def expected_ttl_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_TTL.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()
