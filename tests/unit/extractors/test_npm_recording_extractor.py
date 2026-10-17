"""Contract tests for NpmRecordingExtractor."""

import io
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from guppy.extractors.npm_recording_extractor import (
    ChannelStore,
    EventStore,
    NpmRecordingExtractor,
)
from guppy_test_data import STUBBED_TESTING_DATA

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# The literal header of the issue #337 reproducer, PhAT's Sample2_NPM_1fiber.csv: four blank
# header cells and two timestamp columns, one of them named exactly "Timestamp". Read through
# pandas so the tests pin the real "Unnamed: N" names rather than hand-typed ones.
BLANK_HEADER_CSV = (
    ",Timestamp,msTimestamp,,Region0R,Region1G,,,LedState\n"
    ",57932.73362,57932733.62,182798927,2607.950814,2108.730754,0,,0\n"
    "0,57932.75772,57932757.72,183620416,3630.180108,1425.609443,2,0,4\n"
)

# A headered session with a state column. Row 0 is an initialization frame (every excitation
# bit set), 415 nm is written both plain (1) and with a TTL line high (17, 273), and row 4 strobes
# 470 nm and 560 nm together (6).
EXCITATION_SIGNALS_CSV = (
    "FrameCounter,Timestamp,LedState,Region0G,Region1G\n"
    "0,10.0,7,0.0,10.0\n"
    "1,10.1,1,1.0,11.0\n"
    "2,10.2,2,2.0,12.0\n"
    "3,10.3,17,3.0,13.0\n"
    "4,10.4,6,4.0,14.0\n"
    "5,10.5,273,5.0,15.0\n"
    "6,10.6,2,6.0,16.0\n"
    "7,10.7,4,7.0,17.0\n"
)
EXCITATION_EVENTS_CSV = "10.15,3\n10.35,5\n10.55,3\n"

# A header-less session in milliseconds: timestamps lead, then three data columns, with two
# channels interleaved row by row.
STRIDE_DATA_CSV = "".join(f"{1000 + 10 * row},{row},{100 + row},{200 + row}\n" for row in range(6))


@pytest.fixture
def excitation_session(tmp_path):
    session_folder = tmp_path / "excitation_session"
    session_folder.mkdir()
    (session_folder / "a_signals.csv").write_text(EXCITATION_SIGNALS_CSV)
    (session_folder / "b_events.csv").write_text(EXCITATION_EVENTS_CSV)
    return session_folder


@pytest.fixture
def stride_session(tmp_path):
    session_folder = tmp_path / "stride_session"
    session_folder.mkdir()
    (session_folder / "a_data.csv").write_text(STRIDE_DATA_CSV)
    return session_folder


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


def _outputs_by_store_id(output_dicts):
    return {output_dict["store_id"]: output_dict for output_dict in output_dicts}


# ---------------------------------------------------------------------------
# Reading the session's files
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorSourceFiles:
    def test_only_npm_files_are_read(self, excitation_session):
        # Every other format sharing the folder belongs to its own extractor.
        (excitation_session / "session.doric").write_bytes(b"\x00")
        (excitation_session / "doric_shaped.csv").write_text("Time(s),Region0/Values\nUnit,V\n0.0,0.5\n0.1,0.51\n")
        (excitation_session / "standard.csv").write_text("timestamps,data,sampling_rate\n0.1,1.0,250\n0.2,1.1,250\n")
        (excitation_session / "external_ttl.csv").write_text("timestamps\n0.1\n0.2\n")

        dataframes = NpmRecordingExtractor._read_source_files(excitation_session)

        assert list(dataframes) == ["a_signals.csv", "b_events.csv"]

    def test_derived_per_channel_files_are_not_read(self, excitation_session):
        (excitation_session / "file0_chev1.csv").write_text("0.0,1.0\n0.1,2.0\n")

        assert list(NpmRecordingExtractor._read_source_files(excitation_session)) == ["a_signals.csv", "b_events.csv"]

    def test_a_header_less_file_is_labeled_by_position(self, stride_session):
        dataframe = NpmRecordingExtractor._read_source_files(stride_session)["a_data.csv"]

        assert list(dataframe.columns) == [0, 1, 2, 3]
        np.testing.assert_array_equal(dataframe[0], [1000, 1010, 1020, 1030, 1040, 1050])

    def test_a_malformed_single_column_csv_raises(self, excitation_session):
        # The single-column CSV format requires its column to be named "timestamps".
        (excitation_session / "single.csv").write_text("Timestamp\n0.1\n0.2\n")

        with pytest.raises(ValueError, match="only-supported single-column CSV format"):
            NpmRecordingExtractor._read_source_files(excitation_session)


# ---------------------------------------------------------------------------
# Columns and layout
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorColumns:
    @pytest.mark.parametrize("column_name", ["Region0G", "Region2R", "G0", "G3", "R1", "Channel0G", "left_DMS"])
    def test_is_region_column_accepts_region_names(self, column_name):
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
    def test_is_region_column_rejects_everything_npm_writes_around_the_regions(self, column_name):
        assert NpmRecordingExtractor._is_region_column(column_name) is False

    @pytest.mark.parametrize("column_name", ["flags", "FLAGS", "ledstate", "LEDSTATE", "timestamp"])
    def test_is_region_column_matches_case_insensitively(self, column_name):
        assert NpmRecordingExtractor._is_region_column(column_name) is False

    def test_region_columns_exclude_the_digital_line_columns(self):
        # sampleData_NPM_2's layout: five digital-line columns that are not regions.
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

        assert NpmRecordingExtractor._region_columns(dataframe) == ["Region0G", "Region1G"]

    def test_region_columns_exclude_blank_header_columns(self):
        dataframe = pd.read_csv(io.StringIO(BLANK_HEADER_CSV), index_col=False)

        assert NpmRecordingExtractor._region_columns(dataframe) == ["Region0R", "Region1G"]

    def test_has_text_header_distinguishes_the_two_layouts(self, excitation_session, stride_session):
        headered = NpmRecordingExtractor._read_source_files(excitation_session)["a_signals.csv"]
        header_less = NpmRecordingExtractor._read_source_files(stride_session)["a_data.csv"]

        assert NpmRecordingExtractor._has_text_header(headered) is True
        assert NpmRecordingExtractor._has_text_header(header_less) is False

    @pytest.mark.parametrize("state_column", ["Flags", "flags", "FLAGS", "LedState", "ledstate", "LEDSTATE"])
    def test_has_state_column_matches_case_insensitively(self, state_column):
        dataframe = pd.DataFrame({"Timestamp": [0.0, 0.1], state_column: [1, 2], "Region0G": [1.0, 2.0]})

        assert NpmRecordingExtractor._has_state_column(dataframe) is True

    def test_has_state_column_is_false_without_one(self):
        dataframe = pd.DataFrame({"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "Region0G": [1.0, 2.0]})

        assert NpmRecordingExtractor._has_state_column(dataframe) is False

    def test_is_event_file_is_a_two_column_file(self, excitation_session):
        dataframes = NpmRecordingExtractor._read_source_files(excitation_session)

        assert NpmRecordingExtractor._is_event_file(dataframes["b_events.csv"]) is True
        assert NpmRecordingExtractor._is_event_file(dataframes["a_signals.csv"]) is False

    def test_timestamp_column_names_match_case_insensitively(self, two_timestamp_column_frame):
        assert NpmRecordingExtractor._timestamp_column_names(two_timestamp_column_frame) == [
            "Timestamp_ms",
            "Timestamp_s",
        ]


class TestNpmRecordingExtractorExcitationWavelengths:
    def test_only_wavelengths_whose_bit_is_set_are_recorded(self):
        dataframe = pd.DataFrame({"LedState": [1, 2, 1, 2], "Region0G": [0.0, 1.0, 2.0, 3.0]})

        assert NpmRecordingExtractor._excitation_wavelengths(dataframe) == [415, 470]

    @pytest.mark.parametrize("first_state", [7, 23])
    def test_an_opening_initialization_frame_counts_for_no_wavelength(self, first_state):
        # 7 has every excitation bit set, and 23 is 7 with a digital line high; were it a
        # measurement it would add 560 nm.
        dataframe = pd.DataFrame({"LedState": [first_state, 1, 2, 1, 2], "Region0G": [0.0, 1.0, 2.0, 3.0, 4.0]})

        assert NpmRecordingExtractor._excitation_wavelengths(dataframe) == [415, 470]

    def test_an_all_excitation_frame_later_in_the_recording_counts_for_every_wavelength(self):
        dataframe = pd.DataFrame({"LedState": [1, 2, 7, 1, 2], "Region0G": [0.0, 1.0, 2.0, 3.0, 4.0]})

        assert NpmRecordingExtractor._excitation_wavelengths(dataframe) == [415, 470, 560]

    def test_a_strobed_frame_counts_for_both_its_wavelengths(self):
        # 6 = 0b110 is 470 nm and 560 nm lit together.
        dataframe = pd.DataFrame({"LedState": [1, 6, 1, 6], "Region0G": [0.0, 1.0, 2.0, 3.0]})

        assert NpmRecordingExtractor._excitation_wavelengths(dataframe) == [415, 470, 560]


class TestNpmRecordingExtractorResolveTimestampColumn:
    def test_a_header_less_file_is_timed_by_its_leading_column(self):
        dataframe = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))

        timestamp_column = NpmRecordingExtractor._resolve_timestamp_column(
            dataframe, npm_timestamp_column_name=None, file_name="a_data.csv"
        )

        assert timestamp_column == 0

    def test_the_first_timestamp_column_is_used_by_default(self, two_timestamp_column_frame):
        timestamp_column = NpmRecordingExtractor._resolve_timestamp_column(
            two_timestamp_column_frame, npm_timestamp_column_name=None, file_name="a_data.csv"
        )

        assert timestamp_column == "Timestamp_ms"

    def test_a_named_timestamp_column_is_used(self, two_timestamp_column_frame):
        timestamp_column = NpmRecordingExtractor._resolve_timestamp_column(
            two_timestamp_column_frame, npm_timestamp_column_name="Timestamp_s", file_name="a_data.csv"
        )

        assert timestamp_column == "Timestamp_s"

    def test_a_named_column_the_file_lacks_raises(self, two_timestamp_column_frame):
        with pytest.raises(ValueError, match=r"'BogusTimestamp' not found in columns"):
            NpmRecordingExtractor._resolve_timestamp_column(
                two_timestamp_column_frame, npm_timestamp_column_name="BogusTimestamp", file_name="a_data.csv"
            )

    def test_a_file_offering_one_timestamp_column_ignores_the_named_one(self):
        # The column is chosen once for the whole session, so a file offering only one takes it
        # regardless of the name the session settled on.
        dataframe = pd.DataFrame({"Timestamp": [0.1, 0.2], "LedState": [1, 2], "Region0G": [10.0, 20.0]})

        timestamp_column = NpmRecordingExtractor._resolve_timestamp_column(
            dataframe, npm_timestamp_column_name="ComputerTimestamp", file_name="a_data.csv"
        )

        assert timestamp_column == "Timestamp"

    def test_a_headered_file_with_no_timestamp_column_raises(self):
        dataframe = pd.DataFrame({"FrameCounter": [1, 2], "LedState": [1, 2], "Region0G": [10.0, 20.0]})

        with pytest.raises(ValueError, match=r"'a_data.csv' carries a text header but no timestamp column"):
            NpmRecordingExtractor._resolve_timestamp_column(
                dataframe, npm_timestamp_column_name=None, file_name="a_data.csv"
            )


# ---------------------------------------------------------------------------
# Store IDs, records and provenance
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorStoreRecords:
    def test_discover_names_every_store_and_flags_every_file(self, excitation_session):
        store_ids, flags = NpmRecordingExtractor.discover_events_and_flags(
            folder_path=str(excitation_session), num_ch=2, inputParameters={}
        )

        assert store_ids == [
            "a_signals_415nm_Region0G",
            "a_signals_415nm_Region1G",
            "a_signals_470nm_Region0G",
            "a_signals_470nm_Region1G",
            "a_signals_560nm_Region0G",
            "a_signals_560nm_Region1G",
            "event0",
        ]
        assert flags == ["data_np_v2", "event_np"]

    def test_a_split_event_file_yields_one_store_per_value(self, excitation_session):
        store_ids, _ = NpmRecordingExtractor.discover_events_and_flags(
            folder_path=str(excitation_session),
            num_ch=2,
            inputParameters={"npm_split_events": [False, True]},
        )

        assert store_ids[-2:] == ["event3", "event5"]

    def test_discover_names_the_cycle_slots_of_a_header_less_file(self, stride_session):
        store_ids, flags = NpmRecordingExtractor.discover_events_and_flags(
            folder_path=str(stride_session), num_ch=2, inputParameters={}
        )

        assert store_ids == [
            "a_data_chev1",
            "a_data_chev2",
            "a_data_chev3",
            "a_data_chod1",
            "a_data_chod2",
            "a_data_chod3",
        ]
        assert flags == ["data_np"]

    def test_records_say_where_each_store_comes_from(self, excitation_session):
        dataframes = NpmRecordingExtractor._read_source_files(excitation_session)

        store_records = NpmRecordingExtractor._store_records(
            dataframes, num_ch=2, npm_timestamp_column_name=None, npm_split_events=[False, True]
        )

        assert store_records["a_signals_470nm_Region1G"] == ChannelStore(
            file="a_signals.csv",
            excitation_wavelength_in_nm=470,
            interleave_position=None,
            data_column="Region1G",
            timestamp_column="Timestamp",
        )
        assert store_records["event5"] == EventStore(file="b_events.csv", event_value=5)

    def test_an_unsplit_event_file_holds_every_value(self, excitation_session):
        dataframes = NpmRecordingExtractor._read_source_files(excitation_session)

        store_records = NpmRecordingExtractor._store_records(
            dataframes, num_ch=2, npm_timestamp_column_name=None, npm_split_events=None
        )

        assert store_records["event0"] == EventStore(file="b_events.csv", event_value=None)

    def test_provenance_covers_the_photometry_channels_only(self, stride_session):
        (stride_session / "b_events.csv").write_text("1015,1\n")

        provenance = NpmRecordingExtractor(str(stride_session), num_ch=2).get_store_provenance()

        assert list(provenance) == [
            "a_data_chev1",
            "a_data_chev2",
            "a_data_chev3",
            "a_data_chod1",
            "a_data_chod2",
            "a_data_chod3",
        ]
        assert provenance["a_data_chod2"] == {
            "file": "a_data.csv",
            "excitation_wavelength_in_nm": None,
            "interleave_position": 1,
            "data_column": 2,
            "timestamp_column": 0,
        }

    def test_a_state_column_that_sets_no_excitation_bit_raises(self, tmp_path):
        (tmp_path / "a_data.csv").write_text(
            "FrameCounter,LedState,Timestamp,Signal\n0,0,0.00,0.0\n1,16,0.01,1.0\n2,0,0.02,2.0\n"
        )

        with pytest.raises(ValueError, match=r"'a_data.csv' has a state column whose values set no excitation bit"):
            NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})

    def test_a_headered_file_with_no_region_column_raises(self, tmp_path):
        (tmp_path / "a_data.csv").write_text("FrameCounter,Timestamp,LedState,Output0\n0,0.00,1,0\n1,0.01,2,0\n")

        with pytest.raises(ValueError, match=r"'a_data.csv' has no region columns"):
            NpmRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path), num_ch=2, inputParameters={})

    def test_more_channels_than_cycle_slots_raises(self, stride_session):
        with pytest.raises(ValueError, match=r"Number of channels is set to 4, which exceeds the maximum of 3"):
            NpmRecordingExtractor.discover_events_and_flags(
                folder_path=str(stride_session), num_ch=4, inputParameters={}
            )


# ---------------------------------------------------------------------------
# read, count_samples and save
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorRead:
    @pytest.fixture
    def excitation_outputs(self, excitation_session):
        extractor = NpmRecordingExtractor(str(excitation_session), npm_split_events=[False, True])
        store_ids, _ = NpmRecordingExtractor.discover_events_and_flags(
            folder_path=str(excitation_session), num_ch=2, inputParameters={"npm_split_events": [False, True]}
        )
        return _outputs_by_store_id(extractor.read(events=store_ids, outputPath=""))

    def test_a_wavelength_gathers_every_state_value_carrying_its_bit(self, excitation_outputs):
        # 415 nm is written as 1, 17 and 273 (rows 1, 3, 5); row 0's initialization frame is left out.
        np.testing.assert_array_equal(excitation_outputs["a_signals_415nm_Region0G"]["data"], [1.0, 3.0, 5.0])

    def test_each_channel_is_timed_by_its_own_frames(self, excitation_outputs):
        np.testing.assert_allclose(excitation_outputs["a_signals_415nm_Region0G"]["timestamps"], [10.1, 10.3, 10.5])
        np.testing.assert_allclose(excitation_outputs["a_signals_470nm_Region0G"]["timestamps"], [10.2, 10.4, 10.6])

    def test_a_strobed_frame_reaches_both_its_wavelengths(self, excitation_outputs):
        # Row 4 (state 6) lights 470 nm and 560 nm together.
        np.testing.assert_array_equal(excitation_outputs["a_signals_470nm_Region1G"]["data"], [12.0, 14.0, 16.0])
        np.testing.assert_array_equal(excitation_outputs["a_signals_560nm_Region1G"]["data"], [14.0, 17.0])
        np.testing.assert_allclose(excitation_outputs["a_signals_560nm_Region1G"]["timestamps"], [10.4, 10.7])

    def test_the_sampling_rate_comes_from_the_channels_own_frames(self, excitation_outputs):
        # 3 samples spanning 10.5 - 10.1 = 0.4 s.
        np.testing.assert_allclose(excitation_outputs["a_signals_415nm_Region0G"]["sampling_rate"], [7.5])

    def test_a_split_event_store_holds_its_value_only(self, excitation_outputs):
        np.testing.assert_allclose(excitation_outputs["event3"]["timestamps"], [10.15, 10.55])
        np.testing.assert_allclose(excitation_outputs["event5"]["timestamps"], [10.35])
        assert list(excitation_outputs["event3"]) == ["store_id", "timestamps"]

    def test_an_unsplit_event_store_holds_every_event(self, excitation_session):
        extractor = NpmRecordingExtractor(str(excitation_session))

        (output_dict,) = extractor.read(events=["event0"], outputPath="")

        np.testing.assert_allclose(output_dict["timestamps"], [10.15, 10.35, 10.55])

    def test_a_header_less_file_is_read_by_cycle_slot_and_converted_to_seconds(self, stride_session):
        extractor = NpmRecordingExtractor(str(stride_session), num_ch=2, npm_time_unit="milliseconds")

        outputs = _outputs_by_store_id(extractor.read(events=["a_data_chev1", "a_data_chod2"], outputPath=""))

        # chev takes rows 0, 2, 4 and chod rows 1, 3, 5, each on its own frames' clock.
        np.testing.assert_allclose(outputs["a_data_chev1"]["timestamps"], [1.0, 1.02, 1.04])
        np.testing.assert_array_equal(outputs["a_data_chev1"]["data"], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(outputs["a_data_chod2"]["timestamps"], [1.01, 1.03, 1.05])
        np.testing.assert_array_equal(outputs["a_data_chod2"]["data"], [101.0, 103.0, 105.0])

    def test_an_unrecognized_time_unit_raises(self, stride_session):
        extractor = NpmRecordingExtractor(str(stride_session), num_ch=2, npm_time_unit="minutes")

        with pytest.raises(ValueError, match=r"npm_time_unit='minutes' is not a recognized timestamp unit"):
            extractor.read(events=["a_data_chev1"], outputPath="")

    @pytest.mark.parametrize(
        "store_id, expected_sample_count",
        [("a_signals_415nm_Region0G", 3), ("a_signals_560nm_Region0G", 2), ("event3", 2)],
    )
    def test_count_samples_counts_the_stores_rows(self, excitation_session, store_id, expected_sample_count):
        extractor = NpmRecordingExtractor(str(excitation_session), npm_split_events=[False, True])

        assert extractor.count_samples(event=store_id) == expected_sample_count

    def test_an_extractor_reads_its_session_once(self, excitation_session):
        extractor = NpmRecordingExtractor(str(excitation_session))
        extractor.count_samples(event="event0")

        # With the raw files gone, everything still comes from what the first call read.
        for csv_path in excitation_session.glob("*.csv"):
            csv_path.unlink()
        (output_dict,) = extractor.read(events=["a_signals_470nm_Region0G"], outputPath="")

        np.testing.assert_array_equal(output_dict["data"], [2.0, 4.0, 6.0])

    def test_save_writes_one_file_per_store(self, excitation_session, tmp_path):
        extractor = NpmRecordingExtractor(str(excitation_session))
        output_folder = tmp_path / "run"
        output_folder.mkdir()

        extractor.save(
            output_dicts=extractor.read(events=["a_signals_415nm_Region0G", "event0"], outputPath=str(output_folder)),
            outputPath=str(output_folder),
        )

        with h5py.File(output_folder / "a_signals_415nm_Region0G.hdf5", "r") as channel_file:
            assert sorted(channel_file) == ["data", "sampling_rate", "timestamps"]
            np.testing.assert_array_equal(channel_file["data"][:], [1.0, 3.0, 5.0])
        with h5py.File(output_folder / "event0.hdf5", "r") as event_file:
            assert sorted(event_file) == ["timestamps"]


# ---------------------------------------------------------------------------
# Label Stores helpers
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorHasMultipleEventTtls:
    def test_a_data_file_is_false(self, tmp_path):
        pd.DataFrame(
            {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Signal": [0.1, 0.2]}
        ).to_csv(tmp_path / "data.csv", index=False)

        assert NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path)) == [False]

    def test_an_event_file_with_one_value_is_false(self, tmp_path):
        pd.DataFrame({"timestamp": [0.1, 0.2, 0.3], "value": [1, 1, 1]}).to_csv(tmp_path / "stimuli.csv", index=False)

        assert NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path)) == [False]

    def test_an_event_file_with_several_values_is_true(self, tmp_path):
        pd.DataFrame({"timestamp": [0.1, 0.2, 0.3], "value": [1, 3, 1]}).to_csv(tmp_path / "stimuli.csv", index=False)

        assert NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(tmp_path)) == [True]

    def test_an_external_csv_event_file_is_left_out(self, tmp_path):
        # Mixed intra-session folder: the single-column CSV belongs to the CSV extractor.
        session_folder = tmp_path / "sampleData_NPM_1"
        shutil.copytree(STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1", session_folder)
        (session_folder / "csv_event.csv").write_text("timestamps\n20.0\n40.0\n")

        assert NpmRecordingExtractor.has_multiple_event_ttls(folder_path=str(session_folder)) == [False, True]


class TestNpmRecordingExtractorGetTimestampColumnOptions:
    def test_an_event_file_offers_no_options(self, tmp_path):
        pd.DataFrame({"timestamp": [0.1, 0.2], "value": [1, 1]}).to_csv(tmp_path / "stimuli.csv", index=False)

        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(tmp_path)) == []

    def test_a_single_timestamp_column(self, tmp_path):
        pd.DataFrame(
            {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Signal": [0.5, 0.6]}
        ).to_csv(tmp_path / "data.csv", index=False)

        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(tmp_path)) == ["Timestamp"]

    def test_several_timestamp_columns_in_file_order(self, tmp_path):
        pd.DataFrame(
            {
                "FrameCounter": [1, 2],
                "SystemTimestamp": [0.1, 0.2],
                "LedState": [1, 2],
                "ComputerTimestamp": [0.001, 0.002],
                "Signal": [0.5, 0.6],
            }
        ).to_csv(tmp_path / "data.csv", index=False)

        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(tmp_path)) == [
            "SystemTimestamp",
            "ComputerTimestamp",
        ]

    def test_a_header_less_session_offers_no_options(self, stride_session):
        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(stride_session)) == []

    def test_a_column_shared_by_several_files_is_offered_once(self, tmp_path):
        for file_name in ("FiberData415.csv", "FiberData470.csv"):
            pd.DataFrame(
                {"FrameCounter": [1, 2], "Timestamp": [0.1, 0.2], "LedState": [1, 2], "Region0G": [0.5, 0.6]}
            ).to_csv(tmp_path / file_name, index=False)

        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(tmp_path)) == ["Timestamp"]

    def test_an_external_csv_event_file_is_left_out(self, tmp_path):
        session_folder = tmp_path / "sampleData_NPM_1"
        shutil.copytree(STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1", session_folder)
        (session_folder / "csv_event.csv").write_text("timestamps\n20.0\n40.0\n")

        assert NpmRecordingExtractor.get_timestamp_column_options(folder_path=str(session_folder)) == [
            "SystemTimestamp",
            "ComputerTimestamp",
        ]


# ---------------------------------------------------------------------------
# stub
# ---------------------------------------------------------------------------


class TestNpmRecordingExtractorStub:
    @pytest.fixture
    def session_with_late_events(self, tmp_path):
        """Ten seconds of photometry, with events at 2.5, 5.5 and 8.5 s on the same clock."""
        session_folder = tmp_path / "session"
        session_folder.mkdir()
        rows = "".join(
            f"{second},{float(second)},{1 if second % 2 == 0 else 2},{10.0 + second}\n" for second in range(10)
        )
        (session_folder / "a_signals.csv").write_text("FrameCounter,Timestamp,LedState,Region0G\n" + rows)
        (session_folder / "b_events.csv").write_text("2.5,1\n5.5,1\n8.5,1\n")
        return session_folder

    def test_data_is_cut_at_its_first_timestamp_plus_the_duration(self, session_with_late_events, tmp_path):
        stub_folder = tmp_path / "stubbed"

        NpmRecordingExtractor(str(session_with_late_events)).stub(folder_path=stub_folder, duration_in_seconds=4.0)

        stubbed_data = pd.read_csv(stub_folder / "a_signals.csv")
        np.testing.assert_array_equal(stubbed_data["Timestamp"], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_events_are_kept_only_inside_the_retained_photometry(self, session_with_late_events, tmp_path):
        stub_folder = tmp_path / "stubbed"

        NpmRecordingExtractor(str(session_with_late_events)).stub(folder_path=stub_folder, duration_in_seconds=4.0)

        # The photometry keeps [0, 4] s, so only the event at 2.5 s survives.
        assert (stub_folder / "b_events.csv").read_text() == "2.5,1\n"

    def test_an_event_file_left_with_no_events_is_removed(self, session_with_late_events, tmp_path):
        stub_folder = tmp_path / "stubbed"

        NpmRecordingExtractor(str(session_with_late_events)).stub(folder_path=stub_folder, duration_in_seconds=1.0)

        # The photometry keeps [0, 1] s, before the first event at 2.5 s.
        assert sorted(path.name for path in stub_folder.iterdir()) == ["a_signals.csv"]


# ---------------------------------------------------------------------------
# Contract tests on the stubbed sessions
# ---------------------------------------------------------------------------


class NpmRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides the ``expected_*`` fixtures for the NPM contract test classes."""

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

    def test_store_provenance_covers_exactly_the_data_channels(self, isolated_extractor_instance):
        store_ids, _ = self.extractor_class.discover_events_and_flags(
            str(isolated_extractor_instance.folder_path), **self.discover_kwargs
        )
        output_dicts = isolated_extractor_instance.read(events=store_ids, outputPath="")
        store_provenance = isolated_extractor_instance.get_store_provenance()

        assert list(store_provenance) == [
            output_dict["store_id"] for output_dict in output_dicts if "data" in output_dict
        ]
        for store_id, record in store_provenance.items():
            # A header-less file labels its columns by position, which is also how its columns
            # are recorded.
            header = None if isinstance(record["data_column"], int) else "infer"
            columns = pd.read_csv(
                Path(isolated_extractor_instance.folder_path) / record["file"], header=header, nrows=1
            ).columns
            assert record["timestamp_column"] in columns, (
                f"Store {store_id!r} records timestamp column {record['timestamp_column']!r}, which "
                f"'{record['file']}' does not have: {list(columns)}"
            )
            if record["excitation_wavelength_in_nm"] is None:
                assert record["interleave_position"] is not None
            else:
                assert record["excitation_wavelength_in_nm"] in (415, 470, 560)
                assert record["interleave_position"] is None


class TestNpmRecordingExtractor(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_1"
    # This session offers two timestamp columns and its stimuli file rides ComputerTimestamp. See
    # the sampleData_NPM_1 entry in stubbed_testing_data/README.md.
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
    # Splitting the boolean event file yields eventTrue and eventFalse.
    split_kwargs = {"npm_split_events": [False, True]}
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, **split_kwargs)
    expected_events = [
        "PagCeAVgatFear_14421_415nm_Region0G",
        "PagCeAVgatFear_14421_470nm_Region0G",
        "eventTrue",
    ]
    discover_kwargs = {"num_ch": 2, "inputParameters": split_kwargs}
    stub_extractor_kwargs = {"num_ch": 2, **split_kwargs}
    control_event = "PagCeAVgatFear_14421_415nm_Region0G"
    signal_event = "PagCeAVgatFear_14421_470nm_Region0G"
    ttl_event = "eventTrue"
    # The first two events fall 127 s and 155 s into the photometry.
    stub_ttl_test_duration_in_seconds = 200.0


class TestNpmRecordingExtractorSession5(NpmRecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = Path(STUBBED_TESTING_DATA) / "npm" / "sampleData_NPM_5"
    extractor_instance = NpmRecordingExtractor(folder_path, num_ch=2, npm_time_unit="milliseconds")
    expected_events = ["PagCeAVgatFear_1512_1_chev1", "PagCeAVgatFear_1512_1_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {"npm_time_unit": "milliseconds"}}
    stub_extractor_kwargs = {"num_ch": 2, "npm_time_unit": "milliseconds"}
    control_event = "PagCeAVgatFear_1512_1_chev1"
    signal_event = "PagCeAVgatFear_1512_1_chod1"
    ttl_event = "event0"
    # The first two events fall 173 s and 201 s into the photometry.
    stub_ttl_test_duration_in_seconds = 250.0


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
        assert NpmRecordingExtractor.get_timestamp_column_options(self.folder_path) == ["Timestamp", "msTimestamp"]

    def test_blank_header_columns_do_not_become_channels(self):
        # The file has 9 columns, 4 of them blank-headered. Only Region0R and Region1G are
        # regions, crossed with the three LED states.
        store_ids, _ = NpmRecordingExtractor.discover_events_and_flags(self.folder_path, 2, {})

        assert store_ids == [
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
        store_ids, _ = NpmRecordingExtractor.discover_events_and_flags(self.folder_path, 2, {})
        outputs = _outputs_by_store_id(
            NpmRecordingExtractor(self.folder_path, num_ch=2).read(events=store_ids, outputPath="")
        )

        assert outputs["Sample2_NPM_1fiber_415nm_Region0R"]["data"][0] == pytest.approx(1989.005425)  # row 2
        assert outputs["Sample2_NPM_1fiber_415nm_Region1G"]["data"][0] == pytest.approx(1959.94859)  # row 2
        assert outputs["Sample2_NPM_1fiber_470nm_Region0R"]["data"][0] == pytest.approx(2607.829295)  # row 3
        assert outputs["Sample2_NPM_1fiber_560nm_Region0R"]["data"][0] == pytest.approx(3630.180108)  # row 1

    def test_selecting_the_millisecond_column_switches_the_clock(self):
        # msTimestamp is the same clock as Timestamp at 1000x, so picking it with the
        # matching unit has to land on the same seconds.
        store_id = "Sample2_NPM_1fiber_415nm_Region0R"
        (default_output,) = NpmRecordingExtractor(self.folder_path, num_ch=2).read(events=[store_id], outputPath="")
        (millisecond_output,) = NpmRecordingExtractor(
            self.folder_path,
            num_ch=2,
            npm_timestamp_column_name="msTimestamp",
            npm_time_unit="milliseconds",
        ).read(events=[store_id], outputPath="")

        np.testing.assert_allclose(millisecond_output["timestamps"], default_output["timestamps"], atol=1e-6)
        # And the clock itself is the acquisition's, not a re-zeroed one.
        assert default_output["timestamps"][0] == pytest.approx(57932.78284)


# ---------------------------------------------------------------------------
# Absolute clock and timestamp unit
# ---------------------------------------------------------------------------


@pytest.fixture
def headerless_session(tmp_path):
    """A header-less session whose clock starts at 700000 ms."""
    session_folder = tmp_path / "headerless"
    session_folder.mkdir()
    rows = "".join(f"{700000.0 + 500.0 * row},{row},{10 + row},{20 + row}\n" for row in range(6))
    (session_folder / "a_data.csv").write_text(rows)
    (session_folder / "z_events.csv").write_text("701500.0,1\n")
    return session_folder


class TestNpmAbsoluteTime:
    """NPM keeps the acquisition's own clock (issue #407): neither the continuous channels
    nor the event streams are re-zeroed to start at 0. The conversion to seconds still applies.
    """

    @pytest.fixture
    def headered_session(self, tmp_path):
        """A session with a state column whose clock starts at 500 s."""
        session_folder = tmp_path / "headered"
        session_folder.mkdir()
        rows = "".join(f"{row},{500.0 + 0.5 * row},{1 if row % 2 == 0 else 2},{row}\n" for row in range(12))
        (session_folder / "a_signals.csv").write_text("FrameCounter,Timestamp,LedState,Region0G\n" + rows)
        (session_folder / "b_events.csv").write_text("502.0,1\n504.0,1\n")
        return session_folder

    def test_headered_channel_timestamps_are_absolute(self, headered_session):
        outputs = _outputs_by_store_id(
            NpmRecordingExtractor(str(headered_session), num_ch=2, npm_time_unit="seconds").read(
                events=["a_signals_415nm_Region0G", "a_signals_470nm_Region0G"], outputPath=""
            )
        )

        # LedState 1 selects rows 0, 2, …, 10 and LedState 2 rows 1, 3, …, 11, at 500.0 + 0.5 * row.
        np.testing.assert_allclose(
            outputs["a_signals_415nm_Region0G"]["timestamps"], [500.0, 501.0, 502.0, 503.0, 504.0, 505.0]
        )
        np.testing.assert_allclose(
            outputs["a_signals_470nm_Region0G"]["timestamps"], [500.5, 501.5, 502.5, 503.5, 504.5, 505.5]
        )
        # 6 samples spanning 505.0 - 500.0 = 5.0 s.
        np.testing.assert_allclose(outputs["a_signals_415nm_Region0G"]["sampling_rate"], [1.2])

    def test_headered_event_timestamps_are_absolute(self, headered_session):
        (output_dict,) = NpmRecordingExtractor(str(headered_session), num_ch=2, npm_time_unit="seconds").read(
            events=["event0"], outputPath=""
        )

        np.testing.assert_allclose(output_dict["timestamps"], [502.0, 504.0])

    def test_headerless_timestamps_are_absolute_and_converted_to_seconds(self, headerless_session):
        outputs = _outputs_by_store_id(
            NpmRecordingExtractor(str(headerless_session), num_ch=2, npm_time_unit="milliseconds").read(
                events=["a_data_chev1", "event0"], outputPath=""
            )
        )

        # Rows 0, 2, 4 → 700000/701000/702000 ms.
        np.testing.assert_allclose(outputs["a_data_chev1"]["timestamps"], [700.0, 701.0, 702.0])
        np.testing.assert_allclose(outputs["a_data_chev1"]["sampling_rate"], [1.5])
        np.testing.assert_allclose(outputs["event0"]["timestamps"], [701.5])


class TestNpmTimeUnit:
    """The timestamp unit is one value per session folder, supplied as a parameter."""

    def test_unit_defaults_to_seconds(self, headerless_session):
        outputs = _outputs_by_store_id(
            NpmRecordingExtractor(str(headerless_session), num_ch=2).read(
                events=["a_data_chev1", "event0"], outputPath=""
            )
        )

        np.testing.assert_allclose(outputs["a_data_chev1"]["timestamps"], [700000.0, 701000.0, 702000.0])
        np.testing.assert_allclose(outputs["event0"]["timestamps"], [701500.0])

    def test_unit_applies_to_every_stream_in_the_folder(self, headerless_session):
        outputs = _outputs_by_store_id(
            NpmRecordingExtractor(str(headerless_session), num_ch=2, npm_time_unit="microseconds").read(
                events=["a_data_chev1", "a_data_chod1", "event0"], outputPath=""
            )
        )

        np.testing.assert_allclose(outputs["a_data_chev1"]["timestamps"], [0.7, 0.701, 0.702])
        np.testing.assert_allclose(outputs["a_data_chod1"]["timestamps"], [0.7005, 0.7015, 0.7025])
        np.testing.assert_allclose(outputs["event0"]["timestamps"], [0.7015])
        # 3 samples spanning 0.702 - 0.7 = 0.002 s.
        np.testing.assert_allclose(outputs["a_data_chev1"]["sampling_rate"], [1500.0])
