"""Tests for NwbRecordingExtractor and its private helper functions."""

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
from pynwb import TimeSeries, read_nwb

from guppy.extractors.nwb_recording_extractor import (
    NwbRecordingExtractor,
    _core_events_split_column,
    _find_nwb_file,
    _parse_event_name,
    _read_ndx_event,
    _register_unique_name,
    _resolve_timing,
)
from guppy_test_data import STUBBED_TESTING_DATA

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

MOCK_NWB_FOLDER = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
MOCK_NWB_FILE = MOCK_NWB_FOLDER / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb"

MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FOLDER = (
    STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"
)
MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE = (
    MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FOLDER / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.nwb"
)

MOCK_NWB_CORE_EVENTS_FOLDER = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"
MOCK_NWB_CORE_EVENTS_FILE = MOCK_NWB_CORE_EVENTS_FOLDER / "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.nwb"

_NUM_SAMPLES = 3000
_SAMPLING_RATE = 30.0


class TestFindNwbFile:
    def test_finds_single_nwb_file(self, tmp_path):
        nwb_file = tmp_path / "test.nwb"
        nwb_file.touch()
        result = _find_nwb_file(tmp_path)
        assert result == str(nwb_file)

    def test_raises_when_no_nwb_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No NWB file found"):
            _find_nwb_file(tmp_path)

    def test_raises_when_two_nwb_files(self, tmp_path):
        (tmp_path / "first.nwb").touch()
        (tmp_path / "second.nwb").touch()
        with pytest.raises(ValueError) as exception_info:
            _find_nwb_file(tmp_path)
        message = str(exception_info.value)
        assert "Multiple NWB files" in message
        assert "first.nwb" in message
        assert "second.nwb" in message


class TestParseEventName:
    def test_resolves_single_channel_series(self):
        series_name_to_object = {"my_series": object()}
        series_name, column_index = _parse_event_name("my_series", series_name_to_object)
        assert series_name == "my_series"
        assert column_index is None

    def test_resolves_multi_channel_event(self):
        series_name_to_object = {"my_series": object()}
        series_name, column_index = _parse_event_name("my_series_1", series_name_to_object)
        assert series_name == "my_series"
        assert column_index == 1

    def test_raises_for_unknown_event(self):
        series_name_to_object = {"my_series": object(), "another_series": object()}
        with pytest.raises(ValueError) as exception_info:
            _parse_event_name("unknown_event", series_name_to_object)
        message = str(exception_info.value)
        assert "could not be resolved" in message
        assert "Available series:" in message
        assert "my_series" in message
        assert "another_series" in message


class TestRegisterUniqueName:
    def test_registers_first_occurrence(self):
        seen = set()
        _register_unique_name(seen, "my_series", "FiberPhotometryResponseSeries")
        assert "my_series" in seen

    def test_raises_on_duplicate(self):
        seen = {"my_series"}
        with pytest.raises(ValueError, match="must be unique within the file"):
            _register_unique_name(seen, "my_series", "FiberPhotometryResponseSeries")


class _FakeEventsColumn:
    """Minimal stand-in for a VectorData column supporting ``column[:]`` indexing."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def __getitem__(self, key):
        return self._values[key]


class _FakeEventsTable:
    """Minimal stand-in for a core ``EventsTable`` supporting ``table[column_name]`` access."""

    def __init__(self, name="events", **columns):
        self.name = name
        self.colnames = tuple(columns)
        self._columns = {column_name: _FakeEventsColumn(values) for column_name, values in columns.items()}

    def __getitem__(self, name):
        return self._columns[name]


class TestCoreEventsSplitColumn:
    def test_timestamp_only_returns_none(self):
        table = _FakeEventsTable(timestamp=[1.0, 2.0, 3.0])
        assert _core_events_split_column(table) is None

    def test_ignores_duration_column(self):
        table = _FakeEventsTable(timestamp=[1.0, 2.0], duration=[0.1, 0.1])
        assert _core_events_split_column(table) is None

    def test_single_value_column_is_split_column(self):
        table = _FakeEventsTable(timestamp=[1.0, 2.0], strobe=["16", "2064"])
        assert _core_events_split_column(table) == "strobe"

    def test_multiple_value_columns_raise_not_implemented(self):
        table = _FakeEventsTable(
            name="merged",
            timestamp=[1.0, 2.0],
            event_type=["a", "b"],
            strobe=["16", "2064"],
        )
        with pytest.raises(NotImplementedError, match=r"multiple value columns.*raise an issue"):
            _core_events_split_column(table)


class TestReadNdxEvent:
    def test_raises_for_unknown_tag(self):
        with pytest.raises(ValueError, match=r"Expected one of 'annotated', 'labeled', 'events', 'core'"):
            _read_ndx_event(event_name="bogus", source_info=("invalid_tag",))

    def test_core_event_without_value_column_returns_all_timestamps(self):
        table = _FakeEventsTable(timestamp=[1.0, 2.0, 3.0])
        result = _read_ndx_event(event_name="simple_events", source_info=("core", table, None, None))
        assert result["store_id"] == "simple_events"
        np.testing.assert_array_equal(result["timestamps"], np.array([1.0, 2.0, 3.0]))

    def test_core_event_filters_by_annotation_value(self):
        table = _FakeEventsTable(
            timestamp=[41.0, 42.0, 55.0, 56.0],
            annotation=["Reward", "Reward", "Punishment", "Punishment"],
        )
        result = _read_ndx_event(
            event_name="annotated_events_Reward", source_info=("core", table, "annotation", "Reward")
        )
        assert result["store_id"] == "annotated_events_Reward"
        np.testing.assert_array_equal(result["timestamps"], np.array([41.0, 42.0]))

    def test_core_event_filters_by_strobe_value(self):
        table = _FakeEventsTable(
            timestamp=[160.7, 165.7, 165.8, 210.8, 215.8],
            strobe=["16", "2064", "0", "16", "2064"],
        )
        result = _read_ndx_event(event_name="PAB_16", source_info=("core", table, "strobe", "16"))
        assert result["store_id"] == "PAB_16"
        np.testing.assert_array_equal(result["timestamps"], np.array([160.7, 210.8]))


class TestResolveTiming:
    def test_rate_with_starting_time(self):
        series = TimeSeries(name="test", data=np.zeros(5), unit="n.a.", rate=30.0, starting_time=10.0)
        sampling_rate, timestamps = _resolve_timing(series, 5)
        assert sampling_rate == 30.0
        np.testing.assert_allclose(timestamps, 10.0 + np.arange(5) / 30.0)

    def test_rate_with_no_starting_time_defaults_to_zero(self):
        series = TimeSeries(name="test", data=np.zeros(5), unit="n.a.", rate=30.0)
        sampling_rate, timestamps = _resolve_timing(series, 5)
        assert sampling_rate == 30.0
        np.testing.assert_allclose(timestamps, np.arange(5) / 30.0)

    def test_explicit_timestamps(self):
        series = TimeSeries(name="test", data=np.zeros(3), unit="n.a.", timestamps=[0.0, 0.1, 0.2])
        sampling_rate, timestamps = _resolve_timing(series, 3)
        assert sampling_rate == pytest.approx(10.0, rel=1e-3)
        np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2])


# ---------------------------------------------------------------------------
# Shared mixin for all NWB contract test classes
# ---------------------------------------------------------------------------


class NwbRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides shared fixtures and stub overrides for all NWB extractor test classes.

    NWB's ``stub()`` always raises ``NotImplementedError``, so all four stub tests
    from ``RecordingExtractorTestMixin`` are overridden here. All three concrete
    test classes use the same FiberPhotometryResponseSeries as control and signal,
    so those expected-value fixtures are also centralised here.
    """

    extractor_class = NwbRecordingExtractor
    folder_path = str(MOCK_NWB_FOLDER)
    file_path = str(MOCK_NWB_FILE)
    expected_events = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
        "AnnotatedEventsTable_Reward",
        "AnnotatedEventsTable_Punishment",
        "labeled_events_label_1",
        "labeled_events_label_2",
        "labeled_events_label_3",
        "events",
    ]
    discover_kwargs = {}
    extractor_instance = NwbRecordingExtractor(folder_path=str(MOCK_NWB_FOLDER))
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"

    @pytest.fixture
    def expected_control_timestamps(self):
        # rate=30 Hz, starting_time=0.0 → timestamps are exactly arange(3000)/30
        return np.arange(_NUM_SAMPLES) / _SAMPLING_RATE

    @pytest.fixture
    def expected_control_data(self):
        nwbfile = read_nwb(self.file_path)
        return np.array(nwbfile.acquisition["fiber_photometry_response_series"].data[:, 0])

    @pytest.fixture
    def expected_signal_timestamps(self):
        return np.arange(_NUM_SAMPLES) / _SAMPLING_RATE

    @pytest.fixture
    def expected_signal_data(self):
        nwbfile = read_nwb(self.file_path)
        return np.array(nwbfile.acquisition["fiber_photometry_response_series"].data[:, 1])

    # --- override stub tests (stub() raises NotImplementedError for NWB) ---

    def test_stub_data_matches_original(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    def test_stub_idempotent(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    def test_stub_ttl_timestamps_within_duration(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")

    def test_stub_duration(self, tmp_path, isolated_extractor_instance):
        with pytest.raises(NotImplementedError):
            isolated_extractor_instance.stub(folder_path=tmp_path / "stubbed")


# ---------------------------------------------------------------------------
# Contract test classes — one per TTL event type
# ---------------------------------------------------------------------------


class TestNwbRecordingExtractorEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using a plain ndx-events ``Events`` object as the TTL channel.

    Also hosts general NWB-specific tests (discover count, flags, per-type reads)
    that apply to the extractor as a whole rather than a specific TTL type.
    """

    ttl_event = "events"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # Events timestamps: 45, 46, ..., 54
        return np.arange(45, 55, dtype=np.float64)


class TestNwbExtractorAnnotatedEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using an ``AnnotatedEventsTable`` row as the TTL channel."""

    ttl_event = "AnnotatedEventsTable_Reward"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        return np.array([41.0, 42.0, 43.0, 44.0, 45.0])


class TestNwbRecordingExtractorLabeledEvents(NwbRecordingExtractorTestMixin):
    """Contract tests using a ``LabeledEvents`` label as the TTL channel."""

    ttl_event = "labeled_events_label_1"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # label_1 is index 0: timestamps at positions where data == 0 → 40, 43, 46, 49, 52
        return np.array([40.0, 43.0, 46.0, 49.0, 52.0])


# ---------------------------------------------------------------------------
# Contract tests for ndx-fiber-photometry v0.1.0 mock NWB file
# ---------------------------------------------------------------------------

# The v0.1.0 file carries its own cached namespace, which hdmf refuses to load once a different
# version of ndx-fiber-photometry is registered in the process-wide type map: it then builds the
# file's FiberPhotometryTable against the installed 0.2.x spec and raises ConstructError. Importing
# the package is what registers it, so any test sharing this process with one that imports neuroconv
# decides the outcome. The extractor therefore runs in a fresh interpreter, as
# tests/integration/test_integration_ndx_events_import_state.py does for the same reason.
V010_PROBE_SCRIPT = """
import json, sys
import numpy as np
from pynwb import read_nwb
from guppy.extractors.nwb_recording_extractor import NwbRecordingExtractor

folder_path, file_path, output_path, events_json = sys.argv[1:5]
expected_events = json.loads(events_json)

events, flags = NwbRecordingExtractor.discover_events_and_flags(folder_path)
extractor = NwbRecordingExtractor(folder_path=folder_path)
output_dicts = extractor.read(events=expected_events, outputPath=output_path)
extractor.save(output_dicts=output_dicts, outputPath=output_path)

# The source columns come from the same interpreter, since reading this file is what needs isolating.
nwbfile = read_nwb(file_path)
source_data = np.asarray(nwbfile.acquisition["fiber_photometry_response_series"].data)
np.save(output_path + "/source_data.npy", source_data)

print(json.dumps({
    "events": sorted(events),
    "flags": flags,
    "store_ids": [output_dict["store_id"] for output_dict in output_dicts],
    "control_sample_count": extractor.count_samples(event="fiber_photometry_response_series_0"),
    "signal_sample_count": extractor.count_samples(event="fiber_photometry_response_series_1"),
    "ndx_fiber_photometry_imported": "ndx_fiber_photometry" in sys.modules,
}))
"""


class TestNwbRecordingExtractorNdxFiberPhotometryV010Events:
    """Contract tests for the ndx-fiber-photometry v0.1.0 mock file, read in an isolated interpreter.

    The v0.1.0 mock holds identical data to the current mock — same FiberPhotometryResponseSeries
    shape, same events — but was produced with the older ndx-fiber-photometry API (devices in
    ndx_fiber_photometry directly, no virus/injection/indicator containers in FiberPhotometry).
    """

    expected_events = NwbRecordingExtractorTestMixin.expected_events
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    ttl_event = "events"

    @pytest.fixture(scope="class")
    def probe(self, tmp_path_factory) -> dict:
        """Read, save and inspect the v0.1.0 file once, in a process that never imports the extension."""
        output_path = tmp_path_factory.mktemp("ndx_fiber_photometry_v010")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                V010_PROBE_SCRIPT,
                str(MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FOLDER),
                str(MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE),
                str(output_path),
                json.dumps(self.expected_events),
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return {"output_path": output_path, **json.loads(completed.stdout)}

    def test_the_extension_stays_unimported(self, probe):
        # The premise of the isolation: with the package imported, every read below raises
        # ConstructError instead.
        assert probe["ndx_fiber_photometry_imported"] is False

    def test_discover_returns_events_and_flags_as_lists_of_strings(self, probe):
        assert all(isinstance(event, str) for event in probe["events"])
        assert all(isinstance(flag, str) for flag in probe["flags"])

    def test_discover_includes_expected_events(self, probe):
        assert set(self.expected_events) <= set(probe["events"])

    def test_count_samples_matches_the_series_length(self, probe):
        assert probe["control_sample_count"] == _NUM_SAMPLES
        assert probe["signal_sample_count"] == _NUM_SAMPLES

    def test_read_reports_a_store_id_per_event(self, probe):
        assert probe["store_ids"] == self.expected_events

    def test_save_produces_one_hdf5_per_store_with_timestamps(self, probe):
        for store_id in probe["store_ids"]:
            store_path = probe["output_path"] / f"{store_id}.hdf5"
            assert store_path.exists()
            with h5py.File(store_path, "r") as file:
                assert "timestamps" in file

    @pytest.mark.parametrize("event_name, column_index", [("control_event", 0), ("signal_event", 1)])
    def test_roundtrip_preserves_the_series(self, probe, event_name, column_index):
        source_data = np.load(probe["output_path"] / "source_data.npy")
        with h5py.File(probe["output_path"] / f"{getattr(self, event_name)}.hdf5", "r") as file:
            # rate=30 Hz, starting_time=0.0 -> timestamps are exactly arange(3000)/30
            np.testing.assert_array_equal(file["timestamps"][:], np.arange(_NUM_SAMPLES) / _SAMPLING_RATE)
            np.testing.assert_array_equal(file["data"][:], source_data[:, column_index])

    def test_roundtrip_preserves_the_ttl_timestamps(self, probe):
        with h5py.File(probe["output_path"] / f"{self.ttl_event}.hdf5", "r") as file:
            np.testing.assert_array_equal(file["timestamps"][:], np.arange(45, 55, dtype=np.float64))


# ---------------------------------------------------------------------------
# Contract tests for core (pynwb 4.0) EventsTable mock NWB file
# ---------------------------------------------------------------------------


class TestNwbRecordingExtractorCoreEventsSimple(NwbRecordingExtractorTestMixin):
    """Contract tests for the core-events mock file using a plain ``EventsTable`` as the TTL channel."""

    extractor_class = NwbRecordingExtractor
    folder_path = str(MOCK_NWB_CORE_EVENTS_FOLDER)
    file_path = str(MOCK_NWB_CORE_EVENTS_FILE)
    extractor_instance = NwbRecordingExtractor(folder_path=str(MOCK_NWB_CORE_EVENTS_FOLDER))
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    ttl_event = "simple_events"
    expected_events = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
        "simple_events",
        "annotated_events_Reward",
        "annotated_events_Punishment",
        "strobe_events_0",
        "strobe_events_16",
        "strobe_events_2064",
    ]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # simple_events: timestamps 45, 46, ..., 54
        return np.arange(45, 55, dtype=np.float64)


class TestNwbRecordingExtractorCoreEventsAnnotated(NwbRecordingExtractorTestMixin):
    """Contract tests for the core-events mock file using an annotated ``EventsTable`` as the TTL channel."""

    extractor_class = NwbRecordingExtractor
    folder_path = str(MOCK_NWB_CORE_EVENTS_FOLDER)
    file_path = str(MOCK_NWB_CORE_EVENTS_FILE)
    extractor_instance = NwbRecordingExtractor(folder_path=str(MOCK_NWB_CORE_EVENTS_FOLDER))
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    ttl_event = "annotated_events_Reward"
    expected_events = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
        "simple_events",
        "annotated_events_Reward",
        "annotated_events_Punishment",
        "strobe_events_0",
        "strobe_events_16",
        "strobe_events_2064",
    ]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # Reward timestamps: 41, 42, 43, 44, 45
        return np.array([41.0, 42.0, 43.0, 44.0, 45.0])


class TestNwbRecordingExtractorCoreEventsStrobe(NwbRecordingExtractorTestMixin):
    """Contract tests for the core-events mock file using a strobe-coded ``EventsTable``.

    Mirrors NeuroConv's TDTEventsInterface output: a single "strobe" value column (not
    "annotation"), which the reader splits per code.
    """

    extractor_class = NwbRecordingExtractor
    folder_path = str(MOCK_NWB_CORE_EVENTS_FOLDER)
    file_path = str(MOCK_NWB_CORE_EVENTS_FILE)
    extractor_instance = NwbRecordingExtractor(folder_path=str(MOCK_NWB_CORE_EVENTS_FOLDER))
    control_event = "fiber_photometry_response_series_0"
    signal_event = "fiber_photometry_response_series_1"
    ttl_event = "strobe_events_16"
    expected_events = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
        "simple_events",
        "annotated_events_Reward",
        "annotated_events_Punishment",
        "strobe_events_0",
        "strobe_events_16",
        "strobe_events_2064",
    ]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        # strobe codes [16, 2064, 0, 16, 2064] at timestamps 60..64, so code 16 -> 60, 63.
        return np.array([60.0, 63.0])
