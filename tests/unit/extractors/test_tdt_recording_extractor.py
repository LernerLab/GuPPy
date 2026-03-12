"""Contract tests for TdtRecordingExtractor."""

import os

import numpy as np
import pytest

from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# _ismember
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr, element, expected",
    [
        (["a", "b", "a", "c"], "a", [1, 0, 1, 0]),
        (["x", "y", "z"], "x", [1, 0, 0]),
        (["x", "y", "z"], "w", [0, 0, 0]),
        ([], "a", []),
    ],
)
def test_ismember_returns_indicator_array(arr, element, expected):
    result = TdtRecordingExtractor._ismember(arr, element)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _event_needs_splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, sampling_rate, expected",
    [
        (np.array([1.0, 2.0, 3.0]), 0, False),  # uniform diffs at sampling_rate=0 → no split
        (np.array([1.0, 2.0, 5.0]), 0, True),  # non-uniform diffs at sampling_rate=0 → split
        (np.array([1.0, 2.0, 5.0]), 250.0, False),  # nonzero sampling_rate → never split
        (np.array([1.0]), 0, False),  # single element → no diff → no split
        (np.array([]), 0, False),  # empty → no diff → no split
    ],
)
def test_event_needs_splitting(data, sampling_rate, expected):
    assert TdtRecordingExtractor._event_needs_splitting(data=data, sampling_rate=sampling_rate) == expected


_STUBBED_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "stubbed_testing_data")


class TestTdtRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(_STUBBED_TESTING_DATA, "tdt", "Photo_63_207-181030-103332")
    extractor_instance = TdtRecordingExtractor(folder_path)
    expected_events = ["Dv1A", "Dv2A", "PrtN"]
    discover_kwargs = {}
    control_event = "Dv1A"
    signal_event = "Dv2A"
    ttl_event = "PrtN"
    stub_ttl_test_duration_in_seconds = 100.0

    @pytest.fixture
    def expected_control_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=["Dv1A"], outputPath=str(tmp_path))
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self, tmp_path):
        result = self.extractor_instance.read(events=["Dv1A"], outputPath=str(tmp_path))
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=["Dv2A"], outputPath=str(tmp_path))
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self, tmp_path):
        result = self.extractor_instance.read(events=["Dv2A"], outputPath=str(tmp_path))
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=["PrtN"], outputPath=str(tmp_path))
        return result[0]["timestamps"]
