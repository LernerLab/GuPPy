"""Contract tests for NpmRecordingExtractor."""

import os
import shutil

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
    with pytest.raises(ValueError, match=r"4 unique channel states"):
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


def test_decide_indices_raises_when_num_ch_exceeds_three():
    dataframe = pd.DataFrame({"col1": range(8), "col2": range(8, 16)})
    with pytest.raises(ValueError, match=r"set to 4, which exceeds the maximum of 3 channels"):
        NpmRecordingExtractor.decide_indices("file0_", dataframe, "data_np", num_ch=4)


def test_decide_indices_v2_flag_raises_when_flags_and_ledstate_columns_missing():
    dataframe = pd.DataFrame(
        {"FrameCounter": [1, 2, 3, 4], "Timestamp": [0.1, 0.2, 0.3, 0.4], "Signal": [10, 20, 30, 40]}
    )
    with pytest.raises(ValueError, match=r"do not contain a 'Flags' or 'LedState' column"):
        NpmRecordingExtractor.decide_indices("file0_", dataframe, "data_np_v2", num_ch=2)


@pytest.mark.parametrize("state_column", ["Flags", "flags", "FLAGS", "LedState", "ledstate", "LEDSTATE"])
def test_decide_indices_v2_resolves_flag_columns_case_insensitively(state_column):
    # Detection matches Flags/LedState case-insensitively, so decide_indices must
    # resolve the actual (possibly mixed-case) column name before indexing (issue #381).
    dataframe = pd.DataFrame(
        {
            "FrameCounter": range(12),
            state_column: [0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Timestamp": np.arange(12) * 0.01,
            "Signal": np.arange(12, dtype=float),
        }
    )
    result_df, indices_dict, num_channels = NpmRecordingExtractor.decide_indices(
        "file0_", dataframe, "data_np_v2", num_ch=2
    )
    np.testing.assert_array_equal(indices_dict["file0_chev"], [2, 4, 6, 8, 10])
    np.testing.assert_array_equal(indices_dict["file0_chod"], [3, 5, 7, 9, 11])
    assert num_channels == 2
    assert list(result_df.columns) == ["Timestamp", "Signal"]


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


def test_update_df_with_timestamp_columns_raises_for_missing_name():
    dataframe = pd.DataFrame(
        {
            "FrameCounter": [1, 2],
            "Timestamp_ms": [0.1, 0.2],
            "Timestamp_s": [0.0001, 0.0002],
            "Values": [10, 20],
        }
    )
    with pytest.raises(ValueError, match=r"'BogusTimestamp' not found in columns"):
        NpmRecordingExtractor._update_df_with_timestamp_columns(dataframe, "BogusTimestamp")


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


def test_has_multiple_event_ttls_raises_for_unrecognized_layout(tmp_path):
    pd.DataFrame({"event_code": [1, 2]}).to_csv(tmp_path / "single_column.csv", index=False)

    with pytest.raises(ValueError, match=r"has 1 columns, which is not a recognized NPM layout"):
        NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path))


def test_has_multiple_event_ttls_intra_session_mixed_modality_npm_with_csv_event(tmp_path):
    # Mixed intra-session folder: NPM files plus external 1-column CSV event should not crash helper detection.
    source_folder = STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1"
    session_folder = tmp_path / "sampleData_NPM_1"
    shutil.copytree(source_folder, session_folder)

    csv_ttl_timestamps = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    np.savetxt(session_folder / "csv_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    result = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(session_folder))
    assert result == [False, True]


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


def test_needs_ts_unit_raises_for_unrecognized_layout(tmp_path):
    pd.DataFrame({"event_code": [1, 2]}).to_csv(tmp_path / "single_column.csv", index=False)

    with pytest.raises(ValueError, match=r"has 1 columns, which is not a recognized NPM layout"):
        NpmRecordingExtractor.needs_ts_unit(folder_path=str(tmp_path), num_ch=2)


def test_needs_ts_unit_intra_session_mixed_modality_npm_with_csv_event(tmp_path):
    # Mixed intra-session folder: NPM files plus external 1-column CSV event should not crash TS-unit inference.
    source_folder = STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1"
    session_folder = tmp_path / "sampleData_NPM_1"
    shutil.copytree(source_folder, session_folder)

    csv_ttl_timestamps = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    np.savetxt(session_folder / "csv_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(folder_path=str(session_folder), num_ch=2)
    assert ts_unit_needs == [True, False]
    assert col_names_ts == ["", "SystemTimestamp", "ComputerTimestamp"]


# ---------------------------------------------------------------------------
# discover_events_and_flags error paths
# ---------------------------------------------------------------------------


def test_discover_raises_when_doric_extension_present(tmp_path):
    (tmp_path / "session.doric").write_bytes(b"\x00")  # contents irrelevant, never read
    with pytest.raises(ValueError, match=r"Doric files are not supported by NpmRecordingExtractor"):
        NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})


def test_discover_raises_for_doric_shaped_csv(tmp_path):
    csv_path = tmp_path / "doric_shaped.csv"
    csv_path.write_text("Time(s),Region0/Values\nUnit,V\n0.0,0.5\n0.1,0.51\n")
    with pytest.raises(ValueError, match=r"appears to be a Doric .csv"):
        NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})


def test_discover_raises_for_event_csv_one_column(tmp_path):
    # Column name not exactly lowercase "timestamps" so it bypasses the
    # _is_event_csv pre-filter and reaches the inner 1-column check.
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("Timestamp\n0.1\n0.2\n")
    with pytest.raises(ValueError, match=r"event .csv layout"):
        NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})


def test_discover_raises_for_data_csv_three_columns(tmp_path):
    csv_path = tmp_path / "three.csv"
    csv_path.write_text("timestamps,data,sampling_rate\n0.1,1.0,250\n0.2,1.1,250\n")
    with pytest.raises(ValueError, match=r"data .csv layout"):
        NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})


def test_discover_raises_when_channel_group_counts_do_not_match(tmp_path):
    # Two data_np_v2 files with different channel counts (2 vs 3, by LedState) decompose
    # into unequal per-channel-group counts (chev=2, chod=2, chpr=1), which is rejected.
    two_channel_csv = (
        "FrameCounter,LedState,Timestamp,Signal\n"
        "0,0,0.00,0.0\n1,0,0.01,0.0\n2,1,0.02,1.0\n3,2,0.03,2.0\n4,1,0.04,3.0\n5,2,0.05,4.0\n"
        "6,1,0.06,5.0\n7,2,0.07,6.0\n8,1,0.08,7.0\n9,2,0.09,8.0\n10,1,0.10,9.0\n11,2,0.11,10.0\n"
    )
    three_channel_csv = (
        "FrameCounter,LedState,Timestamp,Signal\n"
        "0,0,0.00,0.0\n1,0,0.01,0.0\n2,1,0.02,1.0\n3,2,0.03,2.0\n4,4,0.04,3.0\n5,1,0.05,4.0\n"
        "6,2,0.06,5.0\n7,4,0.07,6.0\n8,1,0.08,7.0\n9,2,0.09,8.0\n10,4,0.10,9.0\n11,1,0.11,10.0\n"
    )
    (tmp_path / "a_data.csv").write_text(two_channel_csv)
    (tmp_path / "b_data.csv").write_text(three_channel_csv)

    with pytest.raises(ValueError, match=r"Number of channel files must match across channel groups"):
        NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})


from conftest import STUBBED_TESTING_DATA

# ---------------------------------------------------------------------------
# Shared fixtures and stub tests for all NPM test classes
# ---------------------------------------------------------------------------


class NpmRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides the ``expected_*`` fixtures for all NPM extractor test classes.

    NPM now demultiplexes in memory: ``discover_events_and_flags`` and ``read`` write
    nothing to ``folder_path``, so no per-test folder isolation is needed and the base
    mixin's shared fixtures and stub contract tests apply directly. Each NPM test class
    passes its decomposition parameters via the constructor (``extractor_instance`` and
    ``stub_extractor_kwargs``). TTL tests become no-ops when ``ttl_event is None``.
    """

    @pytest.fixture
    def expected_control_timestamps(self, isolated_extractor_instance):
        result = isolated_extractor_instance.read(events=[self.control_event], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self, isolated_extractor_instance):
        result = isolated_extractor_instance.read(events=[self.control_event], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self, isolated_extractor_instance):
        result = isolated_extractor_instance.read(events=[self.signal_event], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self, isolated_extractor_instance):
        result = isolated_extractor_instance.read(events=[self.signal_event], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self, isolated_extractor_instance):
        if self.ttl_event is None:
            return None
        result = isolated_extractor_instance.read(events=[self.ttl_event], outputPath="")
        return result[0]["timestamps"]

    def test_stub_ttl_timestamps_within_duration(self, tmp_path, isolated_extractor_instance):
        # NPM stub() truncates the raw files in their native timestamp units, so a
        # seconds-based cutoff (base mixin) does not apply. Assert instead that stubbing
        # retains exactly a non-empty prefix of the original TTL events.
        if self.ttl_event is None:
            return
        original_ttl = isolated_extractor_instance.read(events=[self.ttl_event], outputPath="")[0]["timestamps"]

        stub_folder_path = tmp_path / "stubbed"
        isolated_extractor_instance.stub(
            folder_path=stub_folder_path, duration_in_seconds=self.stub_ttl_test_duration_in_seconds
        )
        stubbed_extractor = self.extractor_class(folder_path=stub_folder_path, **self.stub_extractor_kwargs)
        stubbed_ttl = stubbed_extractor.read(events=[self.ttl_event], outputPath="")[0]["timestamps"]

        assert 0 < len(stubbed_ttl) < len(original_ttl)
        np.testing.assert_array_equal(stubbed_ttl, original_ttl[: len(stubbed_ttl)])


# ---------------------------------------------------------------------------
# Contract test classes
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractor(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_1")
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "file0_chod1"
    signal_event = "file0_chev1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0


class TestNpmRecordingExtractorSession2(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_2")
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = ["file0_chev6", "file1_chev6"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "file0_chev6"
    signal_event = "file1_chev6"
    ttl_event = None


class TestNpmRecordingExtractorSession3(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_3")
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "file0_chod1"
    signal_event = "file0_chev1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 600.0


class TestNpmRecordingExtractorSession4(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_4")
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, npm_split_events=[True, True])
    expected_events = ["file0_chev1", "file0_chod1", "eventTrue"]
    # npm_split_events=[True, True] splits the boolean event stream into eventTrue/eventFalse.
    discover_kwargs = {"num_ch": 2, "inputParameters": {"npm_split_events": [True, True]}}
    stub_extractor_kwargs = {"num_ch": 2, "npm_split_events": [True, True]}
    control_event = "file0_chev1"
    signal_event = "file0_chod1"
    ttl_event = "eventTrue"
    stub_ttl_test_duration_in_seconds = 100.0


class TestNpmRecordingExtractorSession5(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_5")
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    # npm_split_events=None means no splitting: the event stream becomes event0.
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "file0_chev1"
    signal_event = "file0_chod1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0
