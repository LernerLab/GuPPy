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


from conftest import STUBBED_TESTING_DATA


class TestCsvRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = CsvRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "csv", "sample_data_csv_1")
    extractor_instance = CsvRecordingExtractor(folder_path)
    expected_events = ["Sample_Control_Channel", "Sample_Signal_Channel", "Sample_TTL"]
    discover_kwargs = {}
    control_event = "Sample_Control_Channel"
    signal_event = "Sample_Signal_Channel"
    ttl_event = "Sample_TTL"
    stub_ttl_test_duration_in_seconds = 200.0

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


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def _force_classify_to_csv(monkeypatch):
    """Monkeypatch ``_classify_csv_file`` to always return ``"csv"`` so the
    duplicated validation inside ``discover_events_and_flags`` is reached."""
    monkeypatch.setattr(
        "guppy.extractors.csv_recording_extractor._classify_csv_file",
        lambda path: "csv",
    )


def test_read_csv_missing_event_raises_file_not_found(tmp_path):
    extractor = CsvRecordingExtractor(folder_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No CSV file found for event 'missing_event'"):
        extractor._read_csv("missing_event")


def test_discover_raises_for_non_csv_extension(monkeypatch, tmp_path):
    # Bypass the *.csv glob filter by monkeypatching glob to return a .txt path
    fake_path = tmp_path / "fake.txt"
    fake_path.write_text("timestamps\n0\n")
    monkeypatch.setattr(
        "guppy.extractors.csv_recording_extractor.glob.glob",
        lambda pattern: [str(fake_path)],
    )
    _force_classify_to_csv(monkeypatch)
    with pytest.raises(ValueError, match="Only .csv files are supported"):
        CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))


def test_discover_raises_when_inner_check_sees_doric_shaped_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "doric_shaped.csv"
    pd.DataFrame({"meta_a": ["unit_a", "unit_a"], "meta_b": ["unit_b", "unit_b"]}).to_csv(csv_path, index=False)
    _force_classify_to_csv(monkeypatch)
    with pytest.raises(ValueError, match="appears to be a Doric .csv"):
        CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))


def test_discover_raises_for_single_column_with_wrong_name(monkeypatch, tmp_path):
    csv_path = tmp_path / "wrong_name.csv"
    pd.DataFrame({"not_timestamps": [0.0, 1.0, 2.0]}).to_csv(csv_path, index=False)
    _force_classify_to_csv(monkeypatch)
    with pytest.raises(ValueError, match=r"requires the column to be named 'timestamps'"):
        CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))


def test_discover_raises_for_three_columns_with_wrong_names(monkeypatch, tmp_path):
    csv_path = tmp_path / "wrong_columns.csv"
    pd.DataFrame({"foo": [0.0], "bar": [1.0], "baz": [2.0]}).to_csv(csv_path, index=False)
    _force_classify_to_csv(monkeypatch)
    with pytest.raises(ValueError, match=r"requires column names 'timestamps', 'data', 'sampling_rate'"):
        CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))


def test_discover_raises_for_two_column_npm_shaped_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "npm_shaped.csv"
    pd.DataFrame({"timestamps": [0.0, 1.0], "value": [10.0, 20.0]}).to_csv(csv_path, index=False)
    _force_classify_to_csv(monkeypatch)
    with pytest.raises(ValueError, match="matches the Neurophotometrics"):
        CsvRecordingExtractor.discover_events_and_flags(str(tmp_path))
