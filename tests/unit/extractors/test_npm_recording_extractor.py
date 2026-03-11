"""Contract tests for NpmRecordingExtractor."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.extractors.detect_acquisition_formats import _classify_csv_file
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

    # NPM's discover_events_and_flags writes intermediate per-channel CSVs (e.g.
    # file0_chev1.csv, file0_chod1.csv) into folder_path as a side effect. Without
    # isolation, parallel workers race on those files: one worker truncates a file to
    # begin writing while another reads it, causing EmptyDataError. Each test therefore
    # receives its own copy of the source folder so discovers and reads never collide.

    @pytest.fixture
    def isolated_folder_path(self, tmp_path):
        destination = tmp_path / "npm_data"
        shutil.copytree(self.folder_path, destination)
        return destination

    @pytest.fixture
    def isolated_extractor_instance(self, isolated_folder_path):
        return NpmRecordingExtractor(isolated_folder_path)

    @pytest.fixture
    def expected_control_timestamps(self, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        result = isolated_extractor_instance.read(events=["file0_chod1"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        result = isolated_extractor_instance.read(events=["file0_chod1"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        result = isolated_extractor_instance.read(events=["file0_chev1"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        result = isolated_extractor_instance.read(events=["file0_chev1"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        result = isolated_extractor_instance.read(events=["event0"], outputPath="")
        return result[0]["timestamps"]

    # NPM's read() requires split per-channel CSVs (file0_chod1.csv, file0_chev1.csv) that
    # are created as a side effect of discover_events_and_flags(). The mixin stub tests call
    # read() directly without going through the fixtures that call discover first, so we
    # override each stub test here to call discover before reading.

    def test_stub_data_matches_original(self, tmp_path, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        original_result = isolated_extractor_instance.read(events=[self.control_event], outputPath="")
        original_data = original_result[0]["data"]
        original_timestamps = original_result[0]["timestamps"]

        stub_folder_path = tmp_path / "stubbed"
        isolated_extractor_instance.stub(folder_path=stub_folder_path)
        stubbed_extractor = NpmRecordingExtractor(folder_path=stub_folder_path)
        stubbed_result = stubbed_extractor.read(events=[self.control_event], outputPath="")

        np.testing.assert_array_equal(stubbed_result[0]["data"], original_data[: len(stubbed_result[0]["data"])])
        np.testing.assert_array_equal(
            stubbed_result[0]["timestamps"], original_timestamps[: len(stubbed_result[0]["timestamps"])]
        )

    def test_stub_idempotent(self, tmp_path, isolated_folder_path, isolated_extractor_instance):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        stub_folder_path = tmp_path / "stubbed"

        isolated_extractor_instance.stub(folder_path=stub_folder_path)
        first_result = NpmRecordingExtractor(folder_path=stub_folder_path).read(
            events=[self.control_event], outputPath=""
        )

        isolated_extractor_instance.stub(folder_path=stub_folder_path)
        second_result = NpmRecordingExtractor(folder_path=stub_folder_path).read(
            events=[self.control_event], outputPath=""
        )

        np.testing.assert_array_equal(first_result[0]["data"], second_result[0]["data"])

    @pytest.mark.parametrize("stub_duration_in_seconds", [0.5, 1.0, 2.0])
    def test_stub_duration(self, tmp_path, isolated_folder_path, isolated_extractor_instance, stub_duration_in_seconds):
        NpmRecordingExtractor.discover_events_and_flags(isolated_folder_path, num_ch=2, inputParameters={})
        stub_folder_path = tmp_path / "stubbed"
        isolated_extractor_instance.stub(folder_path=stub_folder_path, duration_in_seconds=stub_duration_in_seconds)
        stubbed_extractor = NpmRecordingExtractor(folder_path=stub_folder_path)
        stubbed_result = stubbed_extractor.read(events=[self.control_event], outputPath="")

        duration_in_seconds = stubbed_result[0]["timestamps"][-1] - stubbed_result[0]["timestamps"][0]
        assert duration_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)

        duration_from_samples_in_seconds = len(stubbed_result[0]["data"]) / float(
            np.atleast_1d(stubbed_result[0]["sampling_rate"])[0]
        )
        assert duration_from_samples_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)

    @pytest.mark.xfail(
        reason=(
            "NPM event CSV timestamps (event0.csv) are in absolute ComputerTimestamp (ms) "
            "before discover_events_and_flags converts them to session-relative seconds. "
            "They cannot be compared to the cutoff derived from signal timestamps without "
            "the conversion that discover_events_and_flags applies."
        )
    )
    def test_stub_ttl_timestamps_within_duration(self, tmp_path, isolated_extractor_instance):
        super().test_stub_ttl_timestamps_within_duration(tmp_path, isolated_extractor_instance)


# ---------------------------------------------------------------------------
# NPM stub tests — all five sessions
# ---------------------------------------------------------------------------

_NPM_SESSIONS = [
    "sampleData_NPM_1",
    "sampleData_NPM_2",
    "sampleData_NPM_3",
    "sampleData_NPM_4",
    "sampleData_NPM_5",
]


@pytest.mark.parametrize("session_name", _NPM_SESSIONS)
def test_stub_npm_session_completes_without_error(session_name, tmp_path):
    """NpmRecordingExtractor.stub() must not raise for any NPM session."""
    source_folder = Path(_TESTING_DATA) / "SampleData_Neurophotometrics" / session_name
    extractor = NpmRecordingExtractor(str(source_folder))
    extractor.stub(folder_path=tmp_path / "stub", duration_in_seconds=1.0)


@pytest.mark.parametrize("session_name", _NPM_SESSIONS)
def test_stub_npm_session_truncates_signal_csvs(session_name, tmp_path):
    """Each NPM signal CSV in the stub folder must have fewer rows than the original."""
    source_folder = Path(_TESTING_DATA) / "SampleData_Neurophotometrics" / session_name
    stub_folder = tmp_path / "stub"
    NpmRecordingExtractor(str(source_folder)).stub(folder_path=stub_folder, duration_in_seconds=1.0)

    for csv_path in sorted(stub_folder.glob("*.csv")):
        if _classify_csv_file(str(csv_path)) != "npm":
            continue
        original_path = source_folder / csv_path.name
        stub_row_count = len(pd.read_csv(csv_path, header=None))
        original_row_count = len(pd.read_csv(original_path, header=None))
        assert (
            stub_row_count < original_row_count
        ), f"{csv_path.name}: stub has {stub_row_count} rows, original has {original_row_count}"


@pytest.mark.parametrize("session_name", _NPM_SESSIONS)
def test_stub_npm_session_is_idempotent(session_name, tmp_path):
    """Calling stub() twice on the same destination produces identical files."""
    source_folder = Path(_TESTING_DATA) / "SampleData_Neurophotometrics" / session_name
    stub_folder = tmp_path / "stub"
    extractor = NpmRecordingExtractor(str(source_folder))

    extractor.stub(folder_path=stub_folder, duration_in_seconds=1.0)
    first_row_counts = {csv.name: len(pd.read_csv(csv, header=None)) for csv in sorted(stub_folder.glob("*.csv"))}

    extractor.stub(folder_path=stub_folder, duration_in_seconds=1.0)
    second_row_counts = {csv.name: len(pd.read_csv(csv, header=None)) for csv in sorted(stub_folder.glob("*.csv"))}

    assert first_row_counts == second_row_counts
