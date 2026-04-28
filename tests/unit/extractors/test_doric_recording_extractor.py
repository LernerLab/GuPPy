"""Contract tests for DoricRecordingExtractor."""

import os

import h5py
import numpy as np
import pytest

from guppy.extractors.doric_recording_extractor import DoricRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# _separate_last_element
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr, expected_remaining, expected_last",
    [
        ([1, 2, 3], [1, 2], 3),
        ([42], [], 42),
        (["a", "b", "c"], ["a", "b"], "c"),
    ],
)
def test_separate_last_element(arr, expected_remaining, expected_last):
    remaining, last = DoricRecordingExtractor._separate_last_element(arr)
    assert remaining == expected_remaining
    assert last == expected_last


# ---------------------------------------------------------------------------
# _access_keys_doricV1
# ---------------------------------------------------------------------------


def test_access_keys_doricV1_excludes_time_column(tmp_path):
    hdf5_path = tmp_path / "doric_v1.doric"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        console = hdf5_file.require_group("Traces/Console")
        console.create_dataset("Time(s)", data=np.array([0.0, 0.1, 0.2]))
        console.create_dataset("AIn-1", data=np.array([1.0, 2.0, 3.0]))
        console.create_dataset("AIn-2", data=np.array([4.0, 5.0, 6.0]))

    with h5py.File(hdf5_path, "r") as hdf5_file:
        keys = DoricRecordingExtractor._access_keys_doricV1(hdf5_file)

    assert "Time(s)" not in keys
    assert set(keys) == {"AIn-1", "AIn-2"}


# ---------------------------------------------------------------------------
# _access_keys_doricV6
# ---------------------------------------------------------------------------


def test_access_keys_doricV6_returns_formatted_paths(tmp_path):
    hdf5_path = tmp_path / "doric_v6.doric"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Simulate a Values/Time pair under a region group (typical V6 structure)
        region = hdf5_file.require_group("DataAcquisition/FiberPhotometry/Signals/Region0G")
        region.create_dataset("Values", data=np.array([1.0, 2.0]))
        region.create_dataset("Time", data=np.array([0.0, 0.1]))

    with h5py.File(hdf5_path, "r") as hdf5_file:
        keys = DoricRecordingExtractor._access_keys_doricV6(hdf5_file)

    # Time dataset is excluded; Values dataset formats as "Signals/Region0G"
    assert "Signals/Region0G" in keys
    assert len(keys) == 1


def test_access_keys_doricV6_excludes_time_datasets(tmp_path):
    hdf5_path = tmp_path / "doric_v6_multi.doric"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        region_a = hdf5_file.require_group("DataAcquisition/FiberPhotometry/Signals/Region0G")
        region_a.create_dataset("Values", data=np.array([1.0]))
        region_a.create_dataset("Time", data=np.array([0.0]))
        region_b = hdf5_file.require_group("DataAcquisition/FiberPhotometry/Signals/Region1G")
        region_b.create_dataset("Values", data=np.array([2.0]))
        region_b.create_dataset("Time", data=np.array([0.0]))

    with h5py.File(hdf5_path, "r") as hdf5_file:
        keys = DoricRecordingExtractor._access_keys_doricV6(hdf5_file)

    assert set(keys) == {"Signals/Region0G", "Signals/Region1G"}


from conftest import STUBBED_TESTING_DATA

# ---------------------------------------------------------------------------
# Shared fixtures for all Doric test classes
# ---------------------------------------------------------------------------


class DoricRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides ``expected_*`` fixtures for all Doric extractor test classes.

    Doric's ``read()`` does not write intermediate files, so ``outputPath=""``
    is used throughout. TTL fixtures return ``None`` when ``ttl_event is None``.
    """

    @pytest.fixture
    def expected_control_timestamps(self):
        result = self.extractor_instance.read(events=[self.control_event], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self):
        result = self.extractor_instance.read(events=[self.control_event], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self):
        result = self.extractor_instance.read(events=[self.signal_event], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self):
        result = self.extractor_instance.read(events=[self.signal_event], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        if self.ttl_event is None:
            return None
        result = self.extractor_instance.read(events=[self.ttl_event], outputPath="")
        return result[0]["timestamps"]


# ---------------------------------------------------------------------------
# Contract test classes
# ---------------------------------------------------------------------------

_EVENT_NAME_TO_EVENT_TYPE = {
    "AIn-1 - Raw": "control",
    "AIn-2 - Raw": "signal",
    "DI--O-1": "ttl",
}


class TestDoricRecordingExtractor(DoricRecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "doric", "sample_doric_1")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE)
    expected_events = ["AIn-1 - Raw", "AIn-2 - Raw", "DI--O-1"]
    discover_kwargs = {}
    control_event = "AIn-1 - Raw"
    signal_event = "AIn-2 - Raw"
    ttl_event = "DI--O-1"
    stub_ttl_test_duration_in_seconds = 100.0
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE}


_EVENT_NAME_TO_EVENT_TYPE_SAMPLE2 = {
    "AIn-1 - Dem (ref)": "control",
    "AIn-1 - Dem (da)": "signal",
    "DI/O-1": "ttl",
}


class TestDoricRecordingExtractorSample2(DoricRecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "doric", "sample_doric_2")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE_SAMPLE2)
    expected_events = ["AIn-1 - Dem (ref)", "AIn-1 - Dem (da)", "DI/O-1"]
    discover_kwargs = {}
    control_event = "AIn-1 - Dem (ref)"
    signal_event = "AIn-1 - Dem (da)"
    ttl_event = "DI/O-1"
    stub_ttl_test_duration_in_seconds = 100.0
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE_SAMPLE2}


_EVENT_NAME_TO_EVENT_TYPE_SAMPLE4 = {
    "Series0001/AIN01xAOUT01-LockIn": "control",
    "Series0001/AIN01xAOUT02-LockIn": "signal",
}


class TestDoricRecordingExtractorSample4(DoricRecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "doric", "sample_doric_4")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE_SAMPLE4)
    expected_events = ["Series0001/AIN01xAOUT01-LockIn", "Series0001/AIN01xAOUT02-LockIn"]
    discover_kwargs = {}
    control_event = "Series0001/AIN01xAOUT01-LockIn"
    signal_event = "Series0001/AIN01xAOUT02-LockIn"
    ttl_event = None
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE_SAMPLE4}


_EVENT_NAME_TO_EVENT_TYPE_SAMPLE5 = {
    "Series0001/AIN01xAOUT01-LockIn": "control",
    "Series0001/AIN01xAOUT02-LockIn": "signal",
}


class TestDoricRecordingExtractorSample5(DoricRecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "doric", "sample_doric_5")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE_SAMPLE5)
    expected_events = ["Series0001/AIN01xAOUT01-LockIn", "Series0001/AIN01xAOUT02-LockIn"]
    discover_kwargs = {}
    control_event = "Series0001/AIN01xAOUT01-LockIn"
    signal_event = "Series0001/AIN01xAOUT02-LockIn"
    ttl_event = None
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE_SAMPLE5}


_EVENT_NAME_TO_EVENT_TYPE_V6 = {
    "CAM1_EXC1/ROI01": "control",
    "CAM1_EXC2/ROI01": "signal",
    "DigitalIO/CAM1": "ttl",
}


class TestDoricRecordingExtractorV6(DoricRecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "doric", "sample_doric_3")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE_V6)
    expected_events = ["CAM1_EXC1/ROI01", "CAM1_EXC2/ROI01", "DigitalIO/CAM1"]
    discover_kwargs = {}
    control_event = "CAM1_EXC1/ROI01"
    signal_event = "CAM1_EXC2/ROI01"
    ttl_event = "DigitalIO/CAM1"
    stub_ttl_test_duration_in_seconds = 100.0
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE_V6}


# ---------------------------------------------------------------------------
# Validation of signal/control data (issue #270)
# ---------------------------------------------------------------------------


class TestValidateSignalControlData:
    """Unit tests for ``_validate_signal_control_data``."""

    def test_accepts_finite_non_constant_data(self):
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        DoricRecordingExtractor._validate_signal_control_data("AIn-1 - Raw", data, "signal")

    def test_rejects_empty_data(self):
        with pytest.raises(ValueError, match="is empty"):
            DoricRecordingExtractor._validate_signal_control_data("ch", np.array([]), "signal")

    def test_rejects_data_with_any_nan(self):
        data = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        with pytest.raises(ValueError, match="1 non-finite value"):
            DoricRecordingExtractor._validate_signal_control_data("AIn-1 - Dem (AOut-2)", data, "signal")

    def test_rejects_all_nan_data(self):
        data = np.full(10, np.nan)
        with pytest.raises(ValueError, match="10 non-finite value"):
            DoricRecordingExtractor._validate_signal_control_data("ch", data, "control")

    def test_rejects_inf_values(self):
        data = np.array([0.1, np.inf, 0.3])
        with pytest.raises(ValueError, match="non-finite"):
            DoricRecordingExtractor._validate_signal_control_data("ch", data, "signal")

    def test_rejects_constant_data(self):
        data = np.ones(100, dtype=float)
        with pytest.raises(ValueError, match="constant.*std=0.*AOut"):
            DoricRecordingExtractor._validate_signal_control_data("AOut-1", data, "signal")

    def test_error_message_includes_event_type(self):
        data = np.ones(5)
        with pytest.raises(ValueError, match="control_DMS"):
            DoricRecordingExtractor._validate_signal_control_data("AOut-1", data, "control_DMS")


# ---------------------------------------------------------------------------
# Doric read paths: existence guards and data validation (issue #270)
# ---------------------------------------------------------------------------


def _make_doric_v1(hdf5_path, channels):
    """Build a synthetic Doric V1 file. ``channels`` maps name -> 1-D array."""
    with h5py.File(hdf5_path, "w") as f:
        console = f.require_group("Traces/Console")
        time_length = max(len(arr) for arr in channels.values())
        console.require_group("Time(s)").create_dataset("Console_time(s)", data=np.linspace(0.0, 1.0, time_length))
        for name, arr in channels.items():
            console.require_group(name).create_dataset(name, data=arr)


class TestDoricV1Validation:
    """V1 read path must surface missing channels and bad data as ValueError."""

    def test_missing_channel_raises_with_available_list(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        _make_doric_v1(hdf5_path, {"AIn-1": np.arange(10.0), "AIn-2": np.arange(10.0)})
        extractor = DoricRecordingExtractor(str(tmp_path), {"Raw": "signal"})
        with pytest.raises(ValueError, match="'Raw' not found.*Available channels"):
            extractor.read(events=["Raw"], outputPath="")

    def test_nan_signal_raises_before_polyfit(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        data_with_nan = np.arange(10.0)
        data_with_nan[0] = np.nan
        _make_doric_v1(
            hdf5_path,
            {"AIn-1 - Dem (AOut-2)": data_with_nan, "DI--O-1": np.ones(10)},
        )
        extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1 - Dem (AOut-2)": "signal", "DI--O-1": "ttl"})
        with pytest.raises(ValueError, match="non-finite value"):
            extractor.read(events=["AIn-1 - Dem (AOut-2)"], outputPath="")

    def test_constant_aout_channel_raises(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        _make_doric_v1(
            hdf5_path,
            {"AOut-1": np.ones(10), "AIn-2 - Raw": np.arange(10.0), "DI--O-1": np.ones(10)},
        )
        extractor = DoricRecordingExtractor(
            str(tmp_path),
            {"AOut-1": "signal", "AIn-2 - Raw": "control", "DI--O-1": "ttl"},
        )
        with pytest.raises(ValueError, match="'AOut-1' is constant.*AOut"):
            extractor.read(events=["AOut-1"], outputPath="")

    def test_valid_signal_still_works(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        signal = np.linspace(0.0, 1.0, 10) + 0.1
        _make_doric_v1(hdf5_path, {"AIn-1 - Raw": signal, "DI--O-1": np.ones(10)})
        extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1 - Raw": "signal", "DI--O-1": "ttl"})
        result = extractor.read(events=["AIn-1 - Raw"], outputPath="")
        np.testing.assert_array_equal(result[0]["data"], signal)


class TestDoricV6Validation:
    """V6 read path must surface missing channels and bad data as ValueError."""

    def _make_v6(self, hdf5_path, channels):
        """``channels``: dict of {group_path: 1-D array}. Each gets sibling Time."""
        with h5py.File(hdf5_path, "w") as f:
            f.require_group("Configurations")
            group_to_children = {}
            for path, arr in channels.items():
                parent, leaf = path.rsplit("/", 1)
                group_to_children.setdefault(parent, {})[leaf] = arr
            for parent, leaves in group_to_children.items():
                grp = f.require_group("DataAcquisition/" + parent)
                length = max(len(a) for a in leaves.values())
                grp.create_dataset("Time", data=np.linspace(0.0, 1.0, length))
                for leaf, arr in leaves.items():
                    grp.create_dataset(leaf, data=arr)

    def test_constant_aout_channel_raises(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        self._make_v6(
            hdf5_path,
            {"Signals/Series0001/AnalogOut/AOUT01": np.ones(10)},
        )
        extractor = DoricRecordingExtractor(str(tmp_path), {"AnalogOut/AOUT01": "signal"})
        with pytest.raises(ValueError, match="constant.*AOut"):
            extractor.read(events=["AnalogOut/AOUT01"], outputPath="")

    def test_missing_channel_raises_with_available_list(self, tmp_path):
        hdf5_path = tmp_path / "session.doric"
        self._make_v6(
            hdf5_path,
            {"Signals/Series0001/AnalogIn/AIN01": np.arange(10.0)},
        )
        extractor = DoricRecordingExtractor(str(tmp_path), {"AnalogIn/NotAChannel": "signal"})
        with pytest.raises(ValueError, match="not found.*Available channels"):
            extractor.read(events=["AnalogIn/NotAChannel"], outputPath="")


class TestDoricCsvDiscovery:
    """CSV discovery must filter junk columns that break the read path (issue #270)."""

    def test_discovery_drops_trailing_all_nan_unnamed_column(self, tmp_path):
        csv_path = tmp_path / "session.csv"
        csv_path.write_text(
            "---,Analog In. | Ch.1,Digital I/O | Ch.1,\n"
            "Time(s),AIn-1 - Dem (ref),DI/O-1,\n"
            "0.0,0.5,0,\n"
            "0.1,0.6,1,\n"
            "0.2,0.7,0,\n"
        )
        events, _ = DoricRecordingExtractor.discover_events_and_flags(str(tmp_path))
        assert "AIn-1 - Dem (ref)" in events
        assert "DI/O-1" in events
        assert not any(e.startswith("Unnamed") for e in events)


class TestDoricCsvValidation:
    """CSV read path must surface missing columns and bad data as ValueError."""

    def _write_csv(self, path, content):
        path.write_text(content)

    def test_missing_column_raises_with_available_list(self, tmp_path):
        csv_path = tmp_path / "session.csv"
        self._write_csv(
            csv_path,
            "---,Analog In. | Ch.1,Digital I/O | Ch.1,\n"
            "Time(s),AIn-1 - Dem (ref),DI/O-1,\n"
            "0.0,0.5,0,\n"
            "0.1,0.6,1,\n",
        )
        extractor = DoricRecordingExtractor(str(tmp_path), {"Unnamed: 7": "signal"})
        with pytest.raises(ValueError, match="'Unnamed: 7' not found.*Available channels"):
            extractor.read(events=["Unnamed: 7"], outputPath="")

    def test_constant_aout_column_raises(self, tmp_path):
        csv_path = tmp_path / "session.csv"
        self._write_csv(
            csv_path,
            "---,Analog Out. | Ch.1,Digital I/O | Ch.1,\n"
            "Time(s),AOut-1,DI/O-1,\n"
            "0.0,1,0,\n"
            "0.1,1,1,\n"
            "0.2,1,0,\n",
        )
        extractor = DoricRecordingExtractor(str(tmp_path), {"AOut-1": "signal", "DI/O-1": "ttl"})
        with pytest.raises(ValueError, match="'AOut-1' is constant"):
            extractor.read(events=["AOut-1"], outputPath="")


# ---------------------------------------------------------------------------
# Multi-file detection and read-path file-not-found errors
# ---------------------------------------------------------------------------


def _write_minimal_doric_v1(path):
    with h5py.File(path, "w") as f:
        console = f.require_group("Traces/Console")
        console.require_group("Time(s)").create_dataset("Console_time(s)", data=np.linspace(0.0, 1.0, 5))
        console.require_group("AIn-1").create_dataset("AIn-1", data=np.arange(5.0))


def test_check_doric_raises_for_multiple_doric_files(tmp_path):
    _write_minimal_doric_v1(tmp_path / "session_a.doric")
    _write_minimal_doric_v1(tmp_path / "session_b.doric")
    extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1": "signal"})
    with pytest.raises(ValueError, match="Multiple Doric data files"):
        extractor._check_doric()


def test_read_doric_csv_raises_for_multiple_csv_files(tmp_path):
    csv_a = tmp_path / "session_a.csv"
    csv_b = tmp_path / "session_b.csv"
    contents = "---,Analog In. | Ch.1,\n" "Time(s),AIn-1 - Dem (ref),\n" "0.0,0.5,\n" "0.1,0.6,\n"
    csv_a.write_text(contents)
    csv_b.write_text(contents)
    extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1 - Dem (ref)": "signal"})
    with pytest.raises(ValueError, match=r"Multiple Doric .csv files"):
        extractor._read_doric_csv(events=["AIn-1 - Dem (ref)"])


def test_read_doric_doric_raises_for_multiple_doric_files(tmp_path):
    _write_minimal_doric_v1(tmp_path / "session_a.doric")
    _write_minimal_doric_v1(tmp_path / "session_b.doric")
    extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1": "signal"})
    with pytest.raises(ValueError, match=r"Multiple Doric .doric files"):
        extractor._read_doric_doric(events=["AIn-1"])


def test_read_raises_when_no_doric_file_present(tmp_path):
    extractor = DoricRecordingExtractor(str(tmp_path), {"AIn-1": "signal"})
    with pytest.raises(FileNotFoundError, match="No Doric file"):
        extractor.read(events=["AIn-1"], outputPath="")


def test_v6_signal_event_with_multiple_matching_paths_raises(tmp_path):
    """Two HDF5 datasets at different parents but ending in ``Series0001/AnalogIn/Values``
    both register against the same event key, triggering the duplicate-match guard."""
    hdf5_path = tmp_path / "session.doric"
    with h5py.File(hdf5_path, "w") as f:
        f.require_group("Configurations")
        # Same final segments ("Series0001/AnalogIn") under two different ancestor paths,
        # so _access_data_doricV6 registers both into decide_path for event "Series0001/AnalogIn".
        for parent in ("Signals", "Other"):
            grp = f.require_group(f"DataAcquisition/{parent}/Series0001/AnalogIn")
            grp.create_dataset("Time", data=np.linspace(0.0, 1.0, 5))
            grp.create_dataset("Values", data=np.arange(5.0) + 0.1)
    extractor = DoricRecordingExtractor(str(tmp_path), {"Series0001/AnalogIn": "signal"})
    with pytest.raises(ValueError, match="matches multiple internal HDF5 paths"):
        extractor.read(events=["Series0001/AnalogIn"], outputPath="")


def test_discover_raises_for_standard_csv_in_doric_extractor(tmp_path):
    """A standard (non-Doric) CSV with numeric data in the first 2 rows triggers the
    'appears to be a standard .csv' guard inside discover_events_and_flags."""
    csv_path = tmp_path / "standard.csv"
    csv_path.write_text("timestamps,data,sampling_rate\n0.1,1.0,250\n0.2,1.1,250\n")
    with pytest.raises(ValueError, match=r"appears to be a standard .csv"):
        DoricRecordingExtractor.discover_events_and_flags(str(tmp_path))
