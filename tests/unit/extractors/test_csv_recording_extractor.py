"""Contract tests for CsvRecordingExtractor."""

import os

import numpy as np
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

    @pytest.fixture
    def expected_control_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_Control_Channel.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()

    @pytest.fixture
    def expected_control_data(self):
        csv_path = os.path.join(self.folder_path, "Sample_Control_Channel.csv")
        return pd.read_csv(csv_path)["data"].to_numpy()

    @pytest.fixture
    def expected_signal_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_Signal_Channel.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()

    @pytest.fixture
    def expected_signal_data(self):
        csv_path = os.path.join(self.folder_path, "Sample_Signal_Channel.csv")
        return pd.read_csv(csv_path)["data"].to_numpy()

    @pytest.fixture
    def expected_ttl_timestamps(self):
        csv_path = os.path.join(self.folder_path, "Sample_TTL.csv")
        return pd.read_csv(csv_path)["timestamps"].to_numpy()

    def test_stub_data_matches_original(self, tmp_path):
        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(folder_path=stub_folder_path)

        original_dataframe = pd.read_csv(os.path.join(self.folder_path, f"{self.control_event}.csv"))
        stubbed_dataframe = pd.read_csv(stub_folder_path / f"{self.control_event}.csv")

        np.testing.assert_array_equal(
            stubbed_dataframe["data"].values,
            original_dataframe["data"].values[: len(stubbed_dataframe)],
        )
        np.testing.assert_array_equal(
            stubbed_dataframe["timestamps"].values,
            original_dataframe["timestamps"].values[: len(stubbed_dataframe)],
        )

    def test_stub_idempotent(self, tmp_path):
        stub_folder_path = tmp_path / "stubbed"

        self.extractor_instance.stub(folder_path=stub_folder_path)
        first_dataframe = pd.read_csv(stub_folder_path / f"{self.control_event}.csv")

        self.extractor_instance.stub(folder_path=stub_folder_path)
        second_dataframe = pd.read_csv(stub_folder_path / f"{self.control_event}.csv")

        np.testing.assert_array_equal(first_dataframe["data"].values, second_dataframe["data"].values)

    def test_stub_ttl_timestamps_within_duration(self, tmp_path):
        stub_duration_in_seconds = 200.0
        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(folder_path=stub_folder_path, duration_in_seconds=stub_duration_in_seconds)

        original_control_dataframe = pd.read_csv(os.path.join(self.folder_path, f"{self.control_event}.csv"))
        first_continuous_timestamp = original_control_dataframe["timestamps"].iloc[0]
        cutoff_timestamp = first_continuous_timestamp + stub_duration_in_seconds

        original_ttl_timestamps = pd.read_csv(os.path.join(self.folder_path, f"{self.ttl_event}.csv"))[
            "timestamps"
        ].values
        stubbed_ttl_timestamps = pd.read_csv(stub_folder_path / f"{self.ttl_event}.csv")["timestamps"].values

        assert np.all(stubbed_ttl_timestamps <= cutoff_timestamp)
        assert len(stubbed_ttl_timestamps) < len(original_ttl_timestamps)
        assert np.all(np.isin(stubbed_ttl_timestamps, original_ttl_timestamps))

    @pytest.mark.parametrize("stub_duration_in_seconds", [0.5, 1.0, 2.0])
    def test_stub_duration(self, tmp_path, stub_duration_in_seconds):
        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(folder_path=stub_folder_path, duration_in_seconds=stub_duration_in_seconds)

        stubbed_dataframe = pd.read_csv(stub_folder_path / f"{self.control_event}.csv")
        stubbed_timestamps = stubbed_dataframe["timestamps"].values
        stubbed_data = stubbed_dataframe["data"].values
        sampling_rate = stubbed_dataframe["sampling_rate"].iloc[0]

        duration_in_seconds = stubbed_timestamps[-1] - stubbed_timestamps[0]
        assert duration_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)

        duration_from_samples_in_seconds = len(stubbed_data) / sampling_rate
        assert duration_from_samples_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)
