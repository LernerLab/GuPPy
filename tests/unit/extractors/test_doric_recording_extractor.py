"""Contract tests for DoricRecordingExtractor."""

import os

import h5py
import numpy as np
import pandas as pd
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
# Informative error tests
# ---------------------------------------------------------------------------


def test_access_data_doricV6_raises_on_empty_channel_control(tmp_path):
    """_access_data_doricV6 raises ValueError with helpful message when a signal channel is empty."""
    hdf5_path = tmp_path / "empty_channel.doric"
    with h5py.File(hdf5_path, "w") as f:
        region = f.require_group("DataAcquisition/FiberPhotometry/Signals/AIN01xAOUT01-LockIn")
        region.create_dataset("Values", data=np.array([1.0, 2.0, 3.0]))
        region.create_dataset("Time", data=np.array([0.0, 0.1, 0.2]))

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"AIN01xAOUT01-LockIn": "control", "AOut1": "signal"},
    )
    with h5py.File(hdf5_path, "r") as f:
        with pytest.raises(ValueError, match="AOut1"):
            extractor._access_data_doricV6(f, ["AOut1"])


def test_access_data_doricV6_raises_on_empty_channel_ttl(tmp_path):
    """_access_data_doricV6 raises ValueError with helpful message when a TTL channel is empty."""
    hdf5_path = tmp_path / "empty_ttl.doric"
    with h5py.File(hdf5_path, "w") as f:
        region = f.require_group("DataAcquisition/FiberPhotometry/Signals/AIN01xAOUT01-LockIn")
        region.create_dataset("Values", data=np.array([1.0, 2.0]))
        region.create_dataset("Time", data=np.array([0.0, 0.1]))

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"AOut2": "ttl"},
    )
    with h5py.File(hdf5_path, "r") as f:
        with pytest.raises(ValueError, match="AOut2"):
            extractor._access_data_doricV6(f, ["AOut2"])


def test_access_data_doricV1_raises_on_missing_channel(tmp_path):
    """_access_data_doricV1 raises ValueError with helpful message when a channel is missing."""
    hdf5_path = tmp_path / "missing_channel.doric"
    with h5py.File(hdf5_path, "w") as f:
        console = f.require_group("Traces/Console")
        time_group = console.require_group("Time(s)")
        time_group.create_dataset("Console_time(s)", data=np.array([0.0, 0.1, 0.2]))
        ain1 = console.require_group("AIn-1")
        ain1.create_dataset("AIn-1", data=np.array([1.0, 2.0, 3.0]))

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"AOut1": "control"},
    )
    with h5py.File(hdf5_path, "r") as f:
        with pytest.raises(ValueError, match="AOut1"):
            extractor._access_data_doricV1(f, ["AOut1"])


def test_read_doric_csv_raises_on_missing_column(tmp_path):
    """_read_doric_csv raises ValueError with file path and available columns when a column is missing."""
    csv_path = tmp_path / "doric_data.csv"
    header_row = "AIn-1,AIn-2\n"
    col_row = "Time(s),AIn-1,AIn-2\n"
    data_rows = "0.0,1.0,2.0\n0.1,1.1,2.1\n"
    csv_path.write_text(header_row + col_row + data_rows)

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"MissingChannel": "signal"},
    )
    with pytest.raises(ValueError, match="MissingChannel") as exc_info:
        extractor._read_doric_csv(["MissingChannel"])
    assert "AIn-1" in str(exc_info.value)
    assert str(csv_path) in str(exc_info.value)


def test_read_doric_csv_raises_targeted_message_for_Raw(tmp_path):
    """_read_doric_csv raises ValueError with artifact-specific message for 'Raw' column."""
    csv_path = tmp_path / "doric_data.csv"
    header_row = "AIn-1,AIn-2\n"
    col_row = "Time(s),AIn-1,AIn-2\n"
    data_rows = "0.0,1.0,2.0\n0.1,1.1,2.1\n"
    csv_path.write_text(header_row + col_row + data_rows)

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"Raw": "signal"},
    )
    with pytest.raises(ValueError, match="export artifact") as exc_info:
        extractor._read_doric_csv(["Raw"])
    assert "step 2" in str(exc_info.value)


def test_read_doric_csv_raises_targeted_message_for_Unknown7(tmp_path):
    """_read_doric_csv raises ValueError with artifact-specific message for 'Unknown 7' column."""
    csv_path = tmp_path / "doric_data.csv"
    header_row = "AIn-1,AIn-2\n"
    col_row = "Time(s),AIn-1,AIn-2\n"
    data_rows = "0.0,1.0,2.0\n0.1,1.1,2.1\n"
    csv_path.write_text(header_row + col_row + data_rows)

    extractor = DoricRecordingExtractor(
        str(tmp_path),
        {"Unknown 7": "signal"},
    )
    with pytest.raises(ValueError, match="export artifact") as exc_info:
        extractor._read_doric_csv(["Unknown 7"])
    assert "step 2" in str(exc_info.value)
