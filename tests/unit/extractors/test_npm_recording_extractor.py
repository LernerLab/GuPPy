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
    # Multi-column file → classified as data_np_v2, not an event file → False
    dataframe = pd.DataFrame(
        {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Signal": [0.1, 0.2]}
    )
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

    with pytest.raises(ValueError, match=r"has 1 column \(event \.csv layout\)"):
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
# timestamp_column_options
# ---------------------------------------------------------------------------


def test_timestamp_column_options_event_file_returns_no_options(tmp_path):
    # 2-column event file → classified as event_np and skipped → no columns to choose from
    dataframe = pd.DataFrame({"timestamp": [0.1, 0.2], "value": [1, 1]})
    dataframe.to_csv(tmp_path / "stimuli.csv", index=False)
    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path)) == []


def test_timestamp_column_options_single_timestamp_column(tmp_path):
    dataframe = pd.DataFrame(
        {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Signal": [0.5, 0.6]}
    )
    dataframe.to_csv(tmp_path / "data.csv", index=False)
    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path)) == ["Timestamp"]


def test_timestamp_column_options_multiple_timestamp_columns(tmp_path):
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
    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path)) == [
        "SystemTimestamp",
        "ComputerTimestamp",
    ]


def test_timestamp_column_options_headerless_session_has_no_options(tmp_path):
    # Header-less files carry no column names, so there is nothing to disambiguate.
    (tmp_path / "data.csv").write_text("".join(f"{700000.0 + 500.0 * i},{i},{10 + i},{20 + i}\n" for i in range(6)))
    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path)) == []


def test_timestamp_column_options_repeated_column_offered_once(tmp_path):
    # sampleData_NPM_2 layout: two data files with the same single timestamp column. The
    # option list must not grow per file — that accumulation used to make GuPPy ask which
    # timestamp column to use for the second file, a question with one possible answer.
    for file_name in ("FiberData415.csv", "FiberData470.csv"):
        pd.DataFrame(
            {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Region0G": [0.5, 0.6]}
        ).to_csv(tmp_path / file_name, index=False)
    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path)) == ["Timestamp"]


def test_timestamp_column_options_raises_for_unrecognized_layout(tmp_path):
    pd.DataFrame({"event_code": [1, 2]}).to_csv(tmp_path / "single_column.csv", index=False)

    with pytest.raises(ValueError, match=r"has 1 column \(event \.csv layout\)"):
        NpmRecordingExtractor.timestamp_column_options(folder_path=str(tmp_path))


def test_timestamp_column_options_intra_session_mixed_modality_npm_with_csv_event(tmp_path):
    # Mixed intra-session folder: NPM files plus external 1-column CSV event should not crash detection.
    source_folder = STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1"
    session_folder = tmp_path / "sampleData_NPM_1"
    shutil.copytree(source_folder, session_folder)

    csv_ttl_timestamps = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    np.savetxt(session_folder / "csv_event.csv", csv_ttl_timestamps, header="timestamps", comments="", fmt="%.6f")

    assert NpmRecordingExtractor.timestamp_column_options(folder_path=str(session_folder)) == [
        "SystemTimestamp",
        "ComputerTimestamp",
    ]


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


from guppy_test_data import STUBBED_TESTING_DATA

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
        # NPM stub() truncates each raw file at its own first timestamp plus the duration,
        # so the event file's window is not the data file's window and the base mixin's
        # cutoff (anchored on the continuous stream) does not apply. Assert instead that
        # stubbing retains exactly a non-empty prefix of the original TTL events.
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
    # This session offers two timestamp columns and its stimuli file rides ComputerTimestamp;
    # the defaults would put the events on a different clock than the photometry. See the
    # sampleData_NPM_1 entry in stubbed_testing_data/README.md.
    clock_kwargs = {"npm_timestamp_column_name": "ComputerTimestamp", "npm_time_unit": "milliseconds"}
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, **clock_kwargs)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": clock_kwargs}
    stub_extractor_kwargs = {"num_ch": 2, **clock_kwargs}
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
    # Same two-column shape as sampleData_NPM_1: ttls.csv rides ComputerTimestamp.
    clock_kwargs = {"npm_timestamp_column_name": "ComputerTimestamp", "npm_time_unit": "milliseconds"}
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, **clock_kwargs)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": clock_kwargs}
    stub_extractor_kwargs = {"num_ch": 2, **clock_kwargs}
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
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, npm_time_unit="milliseconds")
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    # npm_split_events=None means no splitting: the event stream becomes event0.
    discover_kwargs = {"num_ch": 2, "inputParameters": {"npm_time_unit": "milliseconds"}}
    stub_extractor_kwargs = {"num_ch": 2, "npm_time_unit": "milliseconds"}
    control_event = "file0_chev1"
    signal_event = "file0_chod1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0


# ---------------------------------------------------------------------------
# Absolute-clock contract
# ---------------------------------------------------------------------------


@pytest.fixture
def headerless_session(tmp_path):
    """A data_np session (headerless) whose clock starts at 700000 ms."""
    session_folder = tmp_path / "headerless"
    session_folder.mkdir()
    rows = "".join(f"{700000.0 + 500.0 * i},{i},{10 + i},{20 + i}\n" for i in range(6))
    (session_folder / "a_data.csv").write_text(rows)
    (session_folder / "z_events.csv").write_text("701500.0,1\n")
    return session_folder


class TestNpmAbsoluteTime:
    """NPM keeps the acquisition's own clock (issue #407): neither the continuous channels
    nor the event streams are re-zeroed to start at 0. The conversion to seconds still applies.

    The per-session contract classes above cannot catch a regression here — their
    ``expected_*`` fixtures compare ``read()`` against ``read()``, so they hold for any time
    basis. These tests pin the basis with hand-computed literals.
    """

    @pytest.fixture
    def headered_session(self, tmp_path):
        """A data_np_v2 session (LedState interleaving) whose clock starts at 500 s."""
        session_folder = tmp_path / "headered"
        session_folder.mkdir()
        rows = "".join(f"{i},{500.0 + 0.5 * i},{1 if i % 2 == 0 else 2},{i}\n" for i in range(12))
        (session_folder / "a_signals.csv").write_text("FrameCounter,Timestamp,LedState,Region0G\n" + rows)
        # Two events on the same clock as the signal file.
        (session_folder / "b_events.csv").write_text("502.0,1\n504.0,1\n")
        return session_folder

    def test_headered_channel_timestamps_are_absolute(self, headered_session):
        streams = NpmRecordingExtractor(
            str(headered_session),
            num_ch=2,
            npm_time_unit="seconds",
            npm_split_events=[False, False],
        ).decompose()

        # LedState==1 selects rows 0,2,4,6,8,10 → Timestamp 500.0 + 0.5*row. Re-zeroing
        # would have produced [0, 1, 2, 3, 4, 5].
        expected = np.array([500.0, 501.0, 502.0, 503.0, 504.0, 505.0])
        np.testing.assert_allclose(streams["file0_chev1"]["timestamps"], expected)
        # chod borrows chev's axis, so it is absolute too.
        np.testing.assert_allclose(streams["file0_chod1"]["timestamps"], expected)
        # 6 samples spanning 505.0 - 500.0 = 5.0 s.
        np.testing.assert_allclose(streams["file0_chev1"]["sampling_rate"], np.array([1.2]))

    def test_headered_event_timestamps_are_absolute(self, headered_session):
        streams = NpmRecordingExtractor(
            str(headered_session),
            num_ch=2,
            npm_time_unit="seconds",
            npm_split_events=[False, False],
        ).decompose()

        # Raw event values, not shifted by the 500.0 chev reference (which gave [2.0, 4.0]).
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([502.0, 504.0]))

    def test_headerless_timestamps_are_absolute_and_converted_to_seconds(self, headerless_session):
        streams = NpmRecordingExtractor(
            str(headerless_session), num_ch=2, npm_time_unit="milliseconds", npm_split_events=[False, False]
        ).decompose()

        # Headerless files are milliseconds: rows 0,2,4 → 700000/701000/702000 ms.
        # The divisor still applies; only the re-zeroing is gone ([0, 1, 2] before).
        np.testing.assert_allclose(streams["file0_chev1"]["timestamps"], np.array([700.0, 701.0, 702.0]))
        np.testing.assert_allclose(streams["file0_chev1"]["sampling_rate"], np.array([1.5]))
        # The event file shares the millisecond clock: 701500 ms → 701.5 s, not 1.5 s.
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([701.5]))


# ---------------------------------------------------------------------------
# Timestamp unit
# ---------------------------------------------------------------------------


class TestNpmTimeUnit:
    """The timestamp unit is one value per session folder, supplied as a parameter.

    Nothing in the raw files states it, so it is never inferred from the layout: the
    same folder read with a different unit yields timestamps scaled by that factor.
    """

    def test_unit_defaults_to_seconds(self, headerless_session):
        streams = NpmRecordingExtractor(str(headerless_session), num_ch=2, npm_split_events=[False, False]).decompose()

        # Left unspecified, the raw values are taken to be seconds already: no division.
        np.testing.assert_allclose(streams["file0_chev1"]["timestamps"], np.array([700000.0, 701000.0, 702000.0]))
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([701500.0]))

    def test_unit_applies_to_every_stream_in_the_folder(self, headerless_session):
        streams = NpmRecordingExtractor(
            str(headerless_session), num_ch=2, npm_time_unit="microseconds", npm_split_events=[False, False]
        ).decompose()

        # 700000 µs → 0.7 s, and the event file rides the same clock: 701500 µs → 0.7015 s.
        np.testing.assert_allclose(streams["file0_chev1"]["timestamps"], np.array([0.7, 0.701, 0.702]))
        np.testing.assert_allclose(streams["file0_chod1"]["timestamps"], np.array([0.7, 0.701, 0.702]))
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([0.7015]))
        # 3 samples spanning 0.702 - 0.7 = 0.002 s.
        np.testing.assert_allclose(streams["file0_chev1"]["sampling_rate"], np.array([1500.0]))

    def test_unrecognized_unit_raises(self, headerless_session):
        extractor = NpmRecordingExtractor(
            str(headerless_session), num_ch=2, npm_time_unit="minutes", npm_split_events=[False, False]
        )

        with pytest.raises(ValueError, match=r"npm_time_unit='minutes' is not a recognized timestamp unit"):
            extractor.decompose()


# ---------------------------------------------------------------------------
# event/photometry clock agreement
# ---------------------------------------------------------------------------


class TestNpmEventClockValidation:
    """A session offering several timestamp columns can only be read on the one its events ride.

    ``sampleData_NPM_1`` carries both ``SystemTimestamp`` (seconds) and ``ComputerTimestamp``
    (milliseconds); its stimuli file is on the latter.
    """

    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_1")

    @pytest.fixture
    def wrong_clock_extractor(self):
        return NpmRecordingExtractor(
            self.folder_path,
            num_ch=2,
            npm_timestamp_column_name="SystemTimestamp",
            npm_time_unit="seconds",
            npm_split_events=[False, True],
        )

    @pytest.fixture
    def matching_clock_extractor(self):
        return NpmRecordingExtractor(
            self.folder_path,
            num_ch=2,
            npm_timestamp_column_name="ComputerTimestamp",
            npm_time_unit="milliseconds",
            npm_split_events=[False, True],
        )

    def test_off_clock_event_read_reports_both_spans(self, wrong_clock_extractor, tmp_path):
        with pytest.raises(ValueError, match=r"lies entirely outside the photometry timespan"):
            wrong_clock_extractor.read(
                events=["file0_chev1", "file0_chod1", "eventpinknoise"], outputPath=str(tmp_path)
            )

    def test_off_clock_event_read_names_the_column_the_events_ride(self, wrong_clock_extractor, tmp_path):
        with pytest.raises(ValueError, match=r"Set Timestamp column to 'ComputerTimestamp'"):
            wrong_clock_extractor.read(
                events=["file0_chev1", "file0_chod1", "eventpinknoise"], outputPath=str(tmp_path)
            )

    def test_photometry_only_read_is_not_checked(self, wrong_clock_extractor, tmp_path):
        # A mixed-modality session takes its traces from another format and selects no NPM
        # channel, so the NPM photometry span is not the reference for its events.
        output_dicts = wrong_clock_extractor.read(events=["file0_chev1", "file0_chod1"], outputPath=str(tmp_path))

        assert [output_dict["store_id"] for output_dict in output_dicts] == ["file0_chev1", "file0_chod1"]

    def test_matching_clock_read_succeeds(self, matching_clock_extractor, tmp_path):
        output_dicts = matching_clock_extractor.read(events=["file0_chev1", "eventpinknoise"], outputPath=str(tmp_path))

        # 49956358.72 ms is the stub's single pinknoise stimulus, in seconds.
        np.testing.assert_allclose(output_dicts[1]["timestamps"], np.array([49956.35872]))
        assert output_dicts[1]["store_id"] == "eventpinknoise"

    def test_timestamp_column_spans_reports_each_columns_raw_range(self):
        column_spans = NpmRecordingExtractor._timestamp_column_spans(self.folder_path)

        assert list(column_spans) == ["SystemTimestamp", "ComputerTimestamp"]
        np.testing.assert_allclose(column_spans["SystemTimestamp"], (1891.312544, 2011.607936))
        np.testing.assert_allclose(column_spans["ComputerTimestamp"], (49884931.93, 50005222.1))

    @pytest.fixture
    def two_data_file_session(self, tmp_path):
        """Two data files sharing one timestamp column name, on non-overlapping stretches."""
        session_folder = tmp_path / "two_files"
        session_folder.mkdir()
        header = "FrameCounter,Timestamp,LedState,Region0G\n"
        for name, first in (("a_signals.csv", 500.0), ("b_signals.csv", 900.0)):
            rows = "".join(f"{i},{first + 0.5 * i},{1 if i % 2 == 0 else 2},{i}\n" for i in range(12))
            (session_folder / name).write_text(header + rows)
        return session_folder

    def test_timestamp_column_spans_merges_a_shared_column_across_files(self, two_data_file_session):
        column_spans = NpmRecordingExtractor._timestamp_column_spans(str(two_data_file_session))

        # a_signals covers 500.0-505.5 and b_signals 900.0-905.5, reported as one span.
        assert column_spans == {"Timestamp": (500.0, 905.5)}

    @pytest.fixture
    def events_off_every_column_session(self, tmp_path):
        """Two timestamp columns, and an event file on neither of them."""
        session_folder = tmp_path / "off_every_column"
        session_folder.mkdir()
        rows = "".join(
            f"{i},{500.0 + 0.5 * i},{1 if i % 2 == 0 else 2},{900000.0 + 500.0 * i},{i}\n" for i in range(12)
        )
        (session_folder / "a_signals.csv").write_text(
            "FrameCounter,SystemTimestamp,LedState,ComputerTimestamp,Region0G\n" + rows
        )
        (session_folder / "b_events.csv").write_text("5000000.0,1\n5000001.0,1\n")
        return session_folder

    def test_events_matching_no_column_report_every_column_offered(self, events_off_every_column_session, tmp_path):
        extractor = NpmRecordingExtractor(
            str(events_off_every_column_session),
            num_ch=2,
            npm_timestamp_column_name="SystemTimestamp",
            npm_time_unit="seconds",
            npm_split_events=[False, False],
        )

        with pytest.raises(ValueError, match=r"none of this session's timestamp columns") as excinfo:
            extractor.read(events=["file0_chev1", "event0"], outputPath=str(tmp_path))

        assert "'SystemTimestamp' [500, 505.5]" in str(excinfo.value)
        assert "'ComputerTimestamp' [900000, 905500]" in str(excinfo.value)

    @pytest.fixture
    def headerless_off_clock_session(self, tmp_path):
        """A headerless session, so no timestamp column has a name to recommend."""
        session_folder = tmp_path / "headerless_off_clock"
        session_folder.mkdir()
        (session_folder / "a_data.csv").write_text(
            "".join(f"{700000.0 + 500.0 * i},{i},{10 + i},{20 + i}\n" for i in range(6))
        )
        (session_folder / "z_events.csv").write_text("999999.0,1\n")
        return session_folder

    def test_headerless_session_advice_falls_back_to_the_time_unit(self, headerless_off_clock_session, tmp_path):
        extractor = NpmRecordingExtractor(
            str(headerless_off_clock_session), num_ch=2, npm_time_unit="seconds", npm_split_events=[False, False]
        )

        with pytest.raises(ValueError, match=r"Check the Time unit in the Label Stores NPM configuration") as excinfo:
            extractor.read(events=["file0_chev1", "event0"], outputPath=str(tmp_path))

        # Nothing to recommend without column names, so no column is named.
        assert "Set Timestamp column to" not in str(excinfo.value)
