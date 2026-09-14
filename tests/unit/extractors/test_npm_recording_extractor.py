"""Contract tests for NpmRecordingExtractor."""

import io
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guppy.extractors.npm_recording_extractor import NpmRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# The literal header of the issue #337 reproducer, PhAT's Sample2_NPM_1fiber.csv: four blank
# header cells and two timestamp columns, one of them named exactly "Timestamp". Read through
# pandas so the tests pin the real "Unnamed: N" names rather than hand-typed ones.
BLANK_HEADER_CSV = (
    ",Timestamp,msTimestamp,,Region0R,Region1G,,,LedState\n"
    ",57932.73362,57932733.62,182798927,2607.950814,2108.730754,0,,0\n"
    "0,57932.75772,57932757.72,183620416,3630.180108,1425.609443,2,0,4\n"
)


# ---------------------------------------------------------------------------
# _is_region_column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column_name",
    ["Region0G", "Region2R", "G0", "G3", "R1", "Channel0G", "left_DMS"],
)
def test_is_region_column_accepts_region_names(column_name):
    # Regions are recognized by subtracting the columns NPM writes around them, never by a
    # name pattern, so an unseen spelling is still read as a region.
    assert NpmRecordingExtractor._is_region_column(column_name) is True


@pytest.mark.parametrize(
    "column_name",
    [
        "FrameCounter",
        "Timestamp",
        "SystemTimestamp",
        "ComputerTimestamp",
        "msTimestamp",
        "TriggerEvents",
        "Flags",
        "LedState",
        "Stimulation",
        "Output0",
        "Output1",
        "Input0",
        "Input1",
        "SomeOtherCounter",
        "Unnamed: 0",
        "Unnamed: 7",
    ],
)
def test_is_region_column_rejects_everything_npm_writes_around_the_regions(column_name):
    assert NpmRecordingExtractor._is_region_column(column_name) is False


@pytest.mark.parametrize("column_name", ["flags", "FLAGS", "ledstate", "LEDSTATE", "timestamp"])
def test_is_region_column_matches_case_insensitively(column_name):
    # _detect_state_column accepts case variants of Flags/LedState, so a case-sensitive test
    # here would offer the state column as a region.
    assert NpmRecordingExtractor._is_region_column(column_name) is False


# ---------------------------------------------------------------------------
# _detect_state_column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_column", ["Flags", "flags", "FLAGS", "LedState", "ledstate", "LEDSTATE"])
def test_detect_state_column_resolves_case_insensitively(state_column):
    # Issue #381: the actual, possibly mixed-case, column name has to come back out.
    dataframe = pd.DataFrame({"Timestamp": [0.0, 0.1], state_column: [1, 2], "Region0G": [1.0, 2.0]})
    assert NpmRecordingExtractor._detect_state_column(dataframe, "a_data.csv") == state_column


def test_detect_state_column_prefers_flags_when_both_are_present():
    dataframe = pd.DataFrame({"Flags": [17, 18], "LedState": [1, 2], "Region0G": [1.0, 2.0]})
    assert NpmRecordingExtractor._detect_state_column(dataframe, "a_data.csv") == "Flags"


def test_detect_state_column_raises_when_neither_column_is_present():
    dataframe = pd.DataFrame({"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "Region0G": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"do not contain a 'Flags' or 'LedState' column"):
        NpmRecordingExtractor._detect_state_column(dataframe, "a_data.csv")


# ---------------------------------------------------------------------------
# _startup_row_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_state, expected_startup_rows",
    [
        # 7 = 0b111: every excitation bit set, so it would land in all three channels at once.
        (7, 1),
        # 23 = 0b10111: the same three bits with a digital line high alongside.
        (23, 1),
        # 0 and 16 set no excitation bit, so they match no wavelength and need no skipping.
        (0, 0),
        (16, 0),
        # A single-excitation frame is a measurement, not an initialization frame.
        (1, 0),
        (2, 0),
        (4, 0),
        # 6 = 0b110 is 470 nm and 560 nm strobed together -- a real frame, not a startup one.
        (6, 0),
    ],
)
def test_startup_row_count_only_skips_an_all_excitation_first_row(first_state, expected_startup_rows):
    state = np.array([first_state, 1, 2, 1, 2])
    assert NpmRecordingExtractor._startup_row_count(state) == expected_startup_rows


def test_startup_row_count_ignores_an_all_excitation_row_later_in_the_recording():
    # Only row 0 is tested, so a genuine simultaneous-excitation frame mid-recording is kept
    # and reaches every channel it belongs to.
    state = np.array([1, 2, 7, 1, 2])
    assert NpmRecordingExtractor._startup_row_count(state) == 0


# ---------------------------------------------------------------------------
# _resolve_columns
# ---------------------------------------------------------------------------


@pytest.fixture
def two_timestamp_column_frame():
    return pd.DataFrame(
        {
            "FrameCounter": [1, 2],
            "Timestamp_ms": [0.1, 0.2],
            "Timestamp_s": [0.0001, 0.0002],
            "LedState": [1, 2],
            "Region0G": [10.0, 20.0],
        }
    )


def test_resolve_columns_headerless_file_leads_with_the_timestamps():
    dataframe = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    timestamp_column, data_columns = NpmRecordingExtractor._resolve_columns(
        dataframe, columns_are_strings=False, timestamp_column_name=None, source_path="a_data.csv"
    )
    assert timestamp_column == 0
    assert data_columns == [1, 2]


def test_resolve_columns_multiple_timestamps_uses_first_by_default(two_timestamp_column_frame):
    timestamp_column, data_columns = NpmRecordingExtractor._resolve_columns(
        two_timestamp_column_frame, columns_are_strings=True, timestamp_column_name=None, source_path="a_data.csv"
    )
    assert timestamp_column == "Timestamp_ms"
    assert data_columns == ["Region0G"]


def test_resolve_columns_explicit_column_name_used(two_timestamp_column_frame):
    timestamp_column, _ = NpmRecordingExtractor._resolve_columns(
        two_timestamp_column_frame,
        columns_are_strings=True,
        timestamp_column_name="Timestamp_s",
        source_path="a_data.csv",
    )
    assert timestamp_column == "Timestamp_s"


def test_resolve_columns_raises_for_missing_name(two_timestamp_column_frame):
    with pytest.raises(ValueError, match=r"'BogusTimestamp' not found in columns"):
        NpmRecordingExtractor._resolve_columns(
            two_timestamp_column_frame,
            columns_are_strings=True,
            timestamp_column_name="BogusTimestamp",
            source_path="a_data.csv",
        )


def test_resolve_columns_requested_name_is_ignored_when_the_file_offers_one_column():
    # The timestamp column is chosen once for the whole session, so a file offering only one
    # column takes it regardless of the name the session settled on.
    dataframe = pd.DataFrame({"Timestamp": [0.1, 0.2], "LedState": [1, 2], "Region0G": [10.0, 20.0]})
    timestamp_column, _ = NpmRecordingExtractor._resolve_columns(
        dataframe, columns_are_strings=True, timestamp_column_name="ComputerTimestamp", source_path="a_data.csv"
    )
    assert timestamp_column == "Timestamp"


def test_resolve_columns_raises_when_a_headered_file_has_no_timestamp_column():
    dataframe = pd.DataFrame({"FrameCounter": [1, 2], "LedState": [1, 2], "Region0G": [10.0, 20.0]})
    with pytest.raises(ValueError, match=r"carries a text header but no timestamp column"):
        NpmRecordingExtractor._resolve_columns(
            dataframe, columns_are_strings=True, timestamp_column_name=None, source_path="a_data.csv"
        )


def test_resolve_columns_raises_when_a_headered_file_has_no_region_columns():
    dataframe = pd.DataFrame({"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2]})
    with pytest.raises(ValueError, match=r"has no region columns"):
        NpmRecordingExtractor._resolve_columns(
            dataframe, columns_are_strings=True, timestamp_column_name=None, source_path="a_data.csv"
        )


def test_resolve_columns_excludes_the_digital_line_columns():
    # sampleData_NPM_2's layout: five digital-line columns that used to be demultiplexed into
    # channels of their own, documented as "Garbage output" in stubbed_testing_data/README.md.
    dataframe = pd.DataFrame(
        {
            "FrameCounter": [0, 1],
            "Timestamp": [7849.631552, 7849.648192],
            "LedState": [7, 1],
            "Stimulation": [0, 0],
            "Output0": [0, 0],
            "Output1": [0, 0],
            "Input0": [0, 0],
            "Input1": [0, 0],
            "Region0G": [0.0039215686, 0.0234150205],
            "Region1G": [0.0039215686, 0.0039215686],
        }
    )
    _, data_columns = NpmRecordingExtractor._resolve_columns(
        dataframe, columns_are_strings=True, timestamp_column_name=None, source_path="FiberData415.csv"
    )
    assert data_columns == ["Region0G", "Region1G"]


def test_resolve_columns_excludes_blank_header_columns():
    # Issue #337: the file's own "Timestamp" column used to collide with the canonical one the
    # extractor inserted, and its blank-header columns would otherwise have become channels.
    dataframe = pd.read_csv(io.StringIO(BLANK_HEADER_CSV), index_col=False)
    timestamp_column, data_columns = NpmRecordingExtractor._resolve_columns(
        dataframe, columns_are_strings=True, timestamp_column_name="Timestamp", source_path="Sample2_NPM_1fiber.csv"
    )
    assert timestamp_column == "Timestamp"
    assert data_columns == ["Region0R", "Region1G"]


def test_resolve_columns_warns_naming_the_blank_header_columns(caplog):
    dataframe = pd.read_csv(io.StringIO(BLANK_HEADER_CSV), index_col=False)
    with caplog.at_level(logging.WARNING, logger="guppy.extractors.npm_recording_extractor"):
        NpmRecordingExtractor._resolve_columns(
            dataframe, columns_are_strings=True, timestamp_column_name=None, source_path="Sample2_NPM_1fiber.csv"
        )
    assert "Sample2_NPM_1fiber.csv" in caplog.text
    for blank_header_column_name in ["Unnamed: 0", "Unnamed: 3", "Unnamed: 6", "Unnamed: 7"]:
        assert blank_header_column_name in caplog.text


def test_resolve_columns_does_not_modify_the_callers_frame(two_timestamp_column_frame):
    NpmRecordingExtractor._resolve_columns(
        two_timestamp_column_frame, columns_are_strings=True, timestamp_column_name=None, source_path="a_data.csv"
    )
    assert list(two_timestamp_column_frame.columns) == [
        "FrameCounter",
        "Timestamp_ms",
        "Timestamp_s",
        "LedState",
        "Region0G",
    ]


# ---------------------------------------------------------------------------
# _decompose_by_excitation
# ---------------------------------------------------------------------------


def _streams_by_name(channel_groups):
    """Flatten the per-channel-group pairs _decompose_by_* returns into one dict."""
    return {name: stream for channel_group in channel_groups for name, stream in channel_group}


def test_decompose_by_excitation_splits_ledstate_into_one_group_per_wavelength():
    dataframe = pd.DataFrame(
        {
            "Timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "LedState": [7, 1, 2, 4, 1, 2],
            "Region0G": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    channel_groups = NpmRecordingExtractor._decompose_by_excitation(
        dataframe,
        name_prefix="signals_",
        timestamp_column="Timestamp",
        data_columns=["Region0G"],
        source_path="signals.csv",
    )
    # Groups come back in ascending wavelength order: 415, 470, 560.
    assert [name for channel_group in channel_groups for name, _ in channel_group] == [
        "signals_415nm_Region0G",
        "signals_470nm_Region0G",
        "signals_560nm_Region0G",
    ]
    streams = _streams_by_name(channel_groups)
    # Row 0 is the all-excitation startup frame and is dropped by position.
    np.testing.assert_allclose(streams["signals_415nm_Region0G"]["timestamps"], [0.1, 0.4])
    np.testing.assert_allclose(streams["signals_415nm_Region0G"]["data"], [1.0, 4.0])
    np.testing.assert_allclose(streams["signals_470nm_Region0G"]["timestamps"], [0.2, 0.5])
    np.testing.assert_allclose(streams["signals_470nm_Region0G"]["data"], [2.0, 5.0])
    np.testing.assert_allclose(streams["signals_560nm_Region0G"]["timestamps"], [0.3])
    np.testing.assert_allclose(streams["signals_560nm_Region0G"]["data"], [3.0])


def test_decompose_by_excitation_gathers_one_wavelength_written_under_several_state_values():
    # sampleData_NPM_4's real layout: a digital input goes high partway through, so 415 nm is
    # written as both 17 (0b000010001) and 273 (0b100010001), and 470 nm as both 18 and 274.
    dataframe = pd.DataFrame(
        {
            "Timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Flags": [16, 17, 18, 17, 18, 273, 274],
            "Region0G": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    streams = _streams_by_name(
        NpmRecordingExtractor._decompose_by_excitation(
            dataframe,
            name_prefix="fear_",
            timestamp_column="Timestamp",
            data_columns=["Region0G"],
            source_path="fear.csv",
        )
    )
    assert sorted(streams) == ["fear_415nm_Region0G", "fear_470nm_Region0G"]
    # Flags 16 sets no excitation bit, so row 0 belongs to no channel and needs no skipping.
    np.testing.assert_allclose(streams["fear_415nm_Region0G"]["data"], [1.0, 3.0, 5.0])
    np.testing.assert_allclose(streams["fear_415nm_Region0G"]["timestamps"], [0.1, 0.3, 0.5])
    np.testing.assert_allclose(streams["fear_470nm_Region0G"]["data"], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(streams["fear_470nm_Region0G"]["timestamps"], [0.2, 0.4, 0.6])


def test_decompose_by_excitation_gives_a_strobed_frame_to_every_wavelength_it_carries():
    # A rig can strobe two LEDs in one frame: LedState 6 is 470 nm and 560 nm together, their
    # emission landing in different region columns, so that frame belongs to both channels.
    # Selecting on the whole masked word instead of the single bit would drop it from both.
    dataframe = pd.DataFrame(
        {
            "Timestamp": [0.0, 0.1, 0.2, 0.3],
            "LedState": [1, 6, 1, 6],
            "G0": [10.0, 11.0, 12.0, 13.0],
            "R1": [20.0, 21.0, 22.0, 23.0],
        }
    )
    streams = _streams_by_name(
        NpmRecordingExtractor._decompose_by_excitation(
            dataframe,
            name_prefix="strobed_",
            timestamp_column="Timestamp",
            data_columns=["G0", "R1"],
            source_path="strobed.csv",
        )
    )
    assert sorted(streams) == [
        "strobed_415nm_G0",
        "strobed_415nm_R1",
        "strobed_470nm_G0",
        "strobed_470nm_R1",
        "strobed_560nm_G0",
        "strobed_560nm_R1",
    ]
    np.testing.assert_allclose(streams["strobed_415nm_G0"]["data"], [10.0, 12.0])
    # Rows 1 and 3 reach 470 nm and 560 nm alike, on the one timestamp they share.
    np.testing.assert_allclose(streams["strobed_470nm_G0"]["timestamps"], [0.1, 0.3])
    np.testing.assert_allclose(streams["strobed_470nm_G0"]["data"], [11.0, 13.0])
    np.testing.assert_allclose(streams["strobed_560nm_R1"]["timestamps"], [0.1, 0.3])
    np.testing.assert_allclose(streams["strobed_560nm_R1"]["data"], [21.0, 23.0])


def test_decompose_by_excitation_raises_when_no_state_value_names_a_wavelength():
    dataframe = pd.DataFrame(
        {"Timestamp": [0.0, 0.1], "LedState": [0, 16], "Region0G": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match=r"set no excitation bit"):
        NpmRecordingExtractor._decompose_by_excitation(
            dataframe,
            name_prefix="dark_",
            timestamp_column="Timestamp",
            data_columns=["Region0G"],
            source_path="dark.csv",
        )


# ---------------------------------------------------------------------------
# _decompose_by_stride
# ---------------------------------------------------------------------------


def test_decompose_by_stride_partitions_rows_by_cycle_position():
    dataframe = pd.DataFrame(np.arange(18, dtype=float).reshape(6, 3))
    channel_groups = NpmRecordingExtractor._decompose_by_stride(
        dataframe, name_prefix="legacy_", timestamp_column=0, data_columns=[1, 2], num_ch=2
    )
    streams = _streams_by_name(channel_groups)
    assert list(streams) == ["legacy_chev1", "legacy_chev2", "legacy_chod1", "legacy_chod2"]
    # Column 0 holds 0, 3, 6, 9, 12, 15; chev takes rows 0, 2, 4 and chod rows 1, 3, 5.
    np.testing.assert_allclose(streams["legacy_chev1"]["timestamps"], [0.0, 6.0, 12.0])
    np.testing.assert_allclose(streams["legacy_chev1"]["data"], [1.0, 7.0, 13.0])
    np.testing.assert_allclose(streams["legacy_chod2"]["timestamps"], [3.0, 9.0, 15.0])
    np.testing.assert_allclose(streams["legacy_chod2"]["data"], [5.0, 11.0, 17.0])


def test_decompose_by_stride_raises_when_num_ch_exceeds_three():
    dataframe = pd.DataFrame(np.arange(16, dtype=float).reshape(8, 2))
    with pytest.raises(ValueError, match=r"set to 4, which exceeds the maximum of 3 channels"):
        NpmRecordingExtractor._decompose_by_stride(
            dataframe, name_prefix="legacy_", timestamp_column=0, data_columns=[1], num_ch=4
        )


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
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_1"
    # This session offers two timestamp columns and its stimuli file rides ComputerTimestamp;
    # the defaults would put the events on a different clock than the photometry. See the
    # sampleData_NPM_1 entry in stubbed_testing_data/README.md.
    clock_kwargs = {"npm_timestamp_column_name": "ComputerTimestamp", "npm_time_unit": "milliseconds"}
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, **clock_kwargs)
    expected_events = ["bl72bl82_12feb2024_fp_415nm_G0", "bl72bl82_12feb2024_fp_470nm_G0", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": clock_kwargs}
    stub_extractor_kwargs = {"num_ch": 2, **clock_kwargs}
    control_event = "bl72bl82_12feb2024_fp_415nm_G0"
    signal_event = "bl72bl82_12feb2024_fp_470nm_G0"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0


class TestNpmRecordingExtractorSession2(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_2"
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = ["FiberData415_415nm_Region0G", "FiberData470_470nm_Region0G"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "FiberData415_415nm_Region0G"
    signal_event = "FiberData470_470nm_Region0G"
    ttl_event = None


class TestNpmRecordingExtractorSession3(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_3"
    # Same two-column shape as sampleData_NPM_1: ttls.csv rides ComputerTimestamp.
    clock_kwargs = {"npm_timestamp_column_name": "ComputerTimestamp", "npm_time_unit": "milliseconds"}
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, **clock_kwargs)
    expected_events = ["signals_415nm_G0", "signals_470nm_G0", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": clock_kwargs}
    stub_extractor_kwargs = {"num_ch": 2, **clock_kwargs}
    control_event = "signals_415nm_G0"
    signal_event = "signals_470nm_G0"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 600.0


class TestNpmRecordingExtractorSession4(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_4"
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, npm_split_events=[True, True])
    expected_events = [
        "PagCeAVgatFear_14421_415nm_Region0G",
        "PagCeAVgatFear_14421_470nm_Region0G",
        "eventTrue",
    ]
    # npm_split_events=[True, True] splits the boolean event stream into eventTrue/eventFalse.
    discover_kwargs = {"num_ch": 2, "inputParameters": {"npm_split_events": [True, True]}}
    stub_extractor_kwargs = {"num_ch": 2, "npm_split_events": [True, True]}
    control_event = "PagCeAVgatFear_14421_415nm_Region0G"
    signal_event = "PagCeAVgatFear_14421_470nm_Region0G"
    ttl_event = "eventTrue"
    stub_ttl_test_duration_in_seconds = 100.0


class TestNpmRecordingExtractorSession5(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_5"
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, npm_time_unit="milliseconds")
    expected_events = ["PagCeAVgatFear_1512_1_chev1", "PagCeAVgatFear_1512_1_chod1", "event0"]
    # npm_split_events=None means no splitting: the event stream becomes event0.
    discover_kwargs = {"num_ch": 2, "inputParameters": {"npm_time_unit": "milliseconds"}}
    stub_extractor_kwargs = {"num_ch": 2, "npm_time_unit": "milliseconds"}
    control_event = "PagCeAVgatFear_1512_1_chev1"
    signal_event = "PagCeAVgatFear_1512_1_chod1"
    ttl_event = "event0"
    stub_ttl_test_duration_in_seconds = 100.0


class TestNpmRecordingExtractorSession6(NpmRecordingExtractorTestMixin):
    """PhAT's Sample2_NPM_1fiber session: blank header cells and two timestamp columns,
    one of them named exactly ``Timestamp`` (issue #337). Photometry only, no TTL file.

    LedState cycles 1/2/4, so the three channels are the three excitation wavelengths,
    each crossed with Region0R and Region1G. See the sampleData_NPM_6 entry in
    stubbed_testing_data/README.md.
    """

    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_6"
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2)
    expected_events = [
        "Sample2_NPM_1fiber_415nm_Region1G",
        "Sample2_NPM_1fiber_470nm_Region1G",
        "Sample2_NPM_1fiber_560nm_Region0R",
    ]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    stub_extractor_kwargs = {"num_ch": 2}
    control_event = "Sample2_NPM_1fiber_415nm_Region1G"
    signal_event = "Sample2_NPM_1fiber_470nm_Region1G"
    ttl_event = None


class TestNpmBlankHeaderSession:
    """A session whose photometry file carries blank header cells (issue #337).

    The contract class above compares ``read()`` against ``read()``, so it holds whatever
    columns the extractor decides are channels. These tests pin which ones they are.
    """

    folder_path = str(Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_6")

    def test_timestamp_column_options_offers_both_named_columns(self):
        assert NpmRecordingExtractor.timestamp_column_options(self.folder_path) == ["Timestamp", "msTimestamp"]

    def test_blank_header_columns_do_not_become_channels(self):
        # The file has 9 columns, 4 of them blank-headered. Only Region0R and Region1G are
        # regions, crossed with the three LED states — not the twelve streams the blank
        # columns would otherwise add.
        streams = NpmRecordingExtractor(self.folder_path, num_ch=2).decompose()
        assert sorted(streams) == [
            "Sample2_NPM_1fiber_415nm_Region0R",
            "Sample2_NPM_1fiber_415nm_Region1G",
            "Sample2_NPM_1fiber_470nm_Region0R",
            "Sample2_NPM_1fiber_470nm_Region1G",
            "Sample2_NPM_1fiber_560nm_Region0R",
            "Sample2_NPM_1fiber_560nm_Region1G",
        ]

    def test_channel_data_comes_from_the_named_regions(self):
        # Row 0 is LedState 0, which sets no excitation bit and so belongs to no channel.
        # Rows 1/2/3 are LedState 4/1/2, so each wavelength's first sample comes from that
        # row of Region0R and Region1G.
        streams = NpmRecordingExtractor(self.folder_path, num_ch=2).decompose()
        assert streams["Sample2_NPM_1fiber_415nm_Region0R"]["data"][0] == pytest.approx(1989.005425)  # row 2
        assert streams["Sample2_NPM_1fiber_415nm_Region1G"]["data"][0] == pytest.approx(1959.94859)  # row 2
        assert streams["Sample2_NPM_1fiber_470nm_Region0R"]["data"][0] == pytest.approx(2607.829295)  # row 3
        assert streams["Sample2_NPM_1fiber_560nm_Region0R"]["data"][0] == pytest.approx(3630.180108)  # row 1

    def test_selecting_the_millisecond_column_switches_the_clock(self):
        # msTimestamp is the same clock as Timestamp at 1000x, so picking it with the
        # matching unit has to land on the same seconds. This is the popup's own choice
        # reaching the data.
        default_streams = NpmRecordingExtractor(self.folder_path, num_ch=2).decompose()
        millisecond_streams = NpmRecordingExtractor(
            self.folder_path,
            num_ch=2,
            npm_timestamp_column_name="msTimestamp",
            npm_time_unit="milliseconds",
        ).decompose()
        np.testing.assert_allclose(
            millisecond_streams["Sample2_NPM_1fiber_415nm_Region0R"]["timestamps"],
            default_streams["Sample2_NPM_1fiber_415nm_Region0R"]["timestamps"],
            atol=1e-6,
        )
        # And the clock itself is the acquisition's, not a re-zeroed one.
        assert default_streams["Sample2_NPM_1fiber_415nm_Region0R"]["timestamps"][0] == pytest.approx(57932.78284)


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
        np.testing.assert_allclose(streams["a_signals_415nm_Region0G"]["timestamps"], expected)
        # 470 nm rides the 415 nm axis, so it is absolute too.
        np.testing.assert_allclose(streams["a_signals_470nm_Region0G"]["timestamps"], expected)
        # 6 samples spanning 505.0 - 500.0 = 5.0 s.
        np.testing.assert_allclose(streams["a_signals_415nm_Region0G"]["sampling_rate"], np.array([1.2]))

    def test_headered_event_timestamps_are_absolute(self, headered_session):
        streams = NpmRecordingExtractor(
            str(headered_session),
            num_ch=2,
            npm_time_unit="seconds",
            npm_split_events=[False, False],
        ).decompose()

        # Raw event values, not shifted by the 500.0 s photometry start (which gave [2.0, 4.0]).
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([502.0, 504.0]))

    def test_headerless_timestamps_are_absolute_and_converted_to_seconds(self, headerless_session):
        streams = NpmRecordingExtractor(
            str(headerless_session), num_ch=2, npm_time_unit="milliseconds", npm_split_events=[False, False]
        ).decompose()

        # Headerless files are milliseconds: rows 0,2,4 → 700000/701000/702000 ms.
        # The divisor still applies; only the re-zeroing is gone ([0, 1, 2] before).
        np.testing.assert_allclose(streams["a_data_chev1"]["timestamps"], np.array([700.0, 701.0, 702.0]))
        np.testing.assert_allclose(streams["a_data_chev1"]["sampling_rate"], np.array([1.5]))
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
        np.testing.assert_allclose(streams["a_data_chev1"]["timestamps"], np.array([700000.0, 701000.0, 702000.0]))
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([701500.0]))

    def test_unit_applies_to_every_stream_in_the_folder(self, headerless_session):
        streams = NpmRecordingExtractor(
            str(headerless_session), num_ch=2, npm_time_unit="microseconds", npm_split_events=[False, False]
        ).decompose()

        # 700000 µs → 0.7 s, and the event file rides the same clock: 701500 µs → 0.7015 s.
        np.testing.assert_allclose(streams["a_data_chev1"]["timestamps"], np.array([0.7, 0.701, 0.702]))
        np.testing.assert_allclose(streams["a_data_chod1"]["timestamps"], np.array([0.7, 0.701, 0.702]))
        np.testing.assert_allclose(streams["event0"]["timestamps"], np.array([0.7015]))
        # 3 samples spanning 0.702 - 0.7 = 0.002 s.
        np.testing.assert_allclose(streams["a_data_chev1"]["sampling_rate"], np.array([1500.0]))

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

    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_1"

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
                events=[
                    "bl72bl82_12feb2024_fp_415nm_G0",
                    "bl72bl82_12feb2024_fp_470nm_G0",
                    "eventpinknoise",
                ],
                outputPath=str(tmp_path),
            )

    def test_off_clock_event_read_names_the_column_the_events_ride(self, wrong_clock_extractor, tmp_path):
        with pytest.raises(ValueError, match=r"Set Timestamp column to 'ComputerTimestamp'"):
            wrong_clock_extractor.read(
                events=[
                    "bl72bl82_12feb2024_fp_415nm_G0",
                    "bl72bl82_12feb2024_fp_470nm_G0",
                    "eventpinknoise",
                ],
                outputPath=str(tmp_path),
            )

    def test_photometry_only_read_is_not_checked(self, wrong_clock_extractor, tmp_path):
        # A mixed-modality session takes its traces from another format and selects no NPM
        # channel, so the NPM photometry span is not the reference for its events.
        output_dicts = wrong_clock_extractor.read(
            events=["bl72bl82_12feb2024_fp_415nm_G0", "bl72bl82_12feb2024_fp_470nm_G0"], outputPath=str(tmp_path)
        )

        assert [output_dict["store_id"] for output_dict in output_dicts] == [
            "bl72bl82_12feb2024_fp_415nm_G0",
            "bl72bl82_12feb2024_fp_470nm_G0",
        ]

    def test_matching_clock_read_succeeds(self, matching_clock_extractor, tmp_path):
        output_dicts = matching_clock_extractor.read(
            events=["bl72bl82_12feb2024_fp_415nm_G0", "eventpinknoise"], outputPath=str(tmp_path)
        )

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
            extractor.read(events=["a_signals_415nm_Region0G", "event0"], outputPath=str(tmp_path))

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
            extractor.read(events=["a_data_chev1", "event0"], outputPath=str(tmp_path))

        # Nothing to recommend without column names, so no column is named.
        assert "Set Timestamp column to" not in str(excinfo.value)
