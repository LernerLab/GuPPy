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


_STUBBED_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "stubbed_testing_data")

_EVENT_NAME_TO_EVENT_TYPE = {
    "AIn-1 - Raw": "control",
    "AIn-2 - Raw": "signal",
    "DI--O-1": "ttl",
}


class TestDoricRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(_STUBBED_TESTING_DATA, "doric", "sample_doric_1")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE)
    expected_events = ["AIn-1 - Raw", "AIn-2 - Raw", "DI--O-1"]
    discover_kwargs = {}
    control_event = "AIn-1 - Raw"
    signal_event = "AIn-2 - Raw"
    ttl_event = "DI--O-1"
    stub_ttl_test_duration_in_seconds = 100.0
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE}

    @pytest.fixture
    def expected_control_timestamps(self):
        result = self.extractor_instance.read(events=["AIn-1 - Raw"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self):
        result = self.extractor_instance.read(events=["AIn-1 - Raw"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self):
        result = self.extractor_instance.read(events=["AIn-2 - Raw"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self):
        result = self.extractor_instance.read(events=["AIn-2 - Raw"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        result = self.extractor_instance.read(events=["DI--O-1"], outputPath="")
        return result[0]["timestamps"]


_EVENT_NAME_TO_EVENT_TYPE_V6 = {
    "CAM1_EXC1/ROI01": "control",
    "CAM1_EXC2/ROI01": "signal",
    "DigitalIO/CAM1": "ttl",
}


class TestDoricRecordingExtractorV6(RecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(_STUBBED_TESTING_DATA, "doric", "sample_doric_3")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE_V6)
    expected_events = ["CAM1_EXC1/ROI01", "CAM1_EXC2/ROI01", "DigitalIO/CAM1"]
    discover_kwargs = {}
    control_event = "CAM1_EXC1/ROI01"
    signal_event = "CAM1_EXC2/ROI01"
    ttl_event = "DigitalIO/CAM1"
    stub_ttl_test_duration_in_seconds = 100.0
    stub_extractor_kwargs = {"event_name_to_event_type": _EVENT_NAME_TO_EVENT_TYPE_V6}

    @pytest.fixture
    def expected_control_timestamps(self):
        result = self.extractor_instance.read(events=["CAM1_EXC1/ROI01"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self):
        result = self.extractor_instance.read(events=["CAM1_EXC1/ROI01"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self):
        result = self.extractor_instance.read(events=["CAM1_EXC2/ROI01"], outputPath="")
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self):
        result = self.extractor_instance.read(events=["CAM1_EXC2/ROI01"], outputPath="")
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self):
        result = self.extractor_instance.read(events=["DigitalIO/CAM1"], outputPath="")
        return result[0]["timestamps"]
