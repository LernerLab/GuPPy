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


_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestTdtRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Clean", "Photo_63_207-181030-103332")
    extractor_instance = TdtRecordingExtractor(folder_path)
    expected_events = ["Dv1A", "Dv2A", "PrtN"]
    discover_kwargs = {}
    control_event = "Dv1A"
    signal_event = "Dv2A"
    ttl_event = "PrtN"

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

    def test_stub_data_matches_original(self, tmp_path):
        original_result = self.extractor_instance.read(events=["Dv1A"], outputPath=str(tmp_path))
        original_data = original_result[0]["data"]
        original_timestamps = original_result[0]["timestamps"]

        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(stub_folder_path=stub_folder_path)
        stubbed_extractor = TdtRecordingExtractor(folder_path=stub_folder_path)
        stubbed_result = stubbed_extractor.read(events=["Dv1A"], outputPath=str(tmp_path))
        stubbed_data = stubbed_result[0]["data"]
        stubbed_timestamps = stubbed_result[0]["timestamps"]

        np.testing.assert_array_equal(stubbed_data, original_data[: len(stubbed_data)])
        np.testing.assert_array_equal(stubbed_timestamps, original_timestamps[: len(stubbed_timestamps)])

    def test_stub_idempotent(self, tmp_path):
        stub_folder_path = tmp_path / "stubbed"

        self.extractor_instance.stub(stub_folder_path=stub_folder_path)
        first_result = TdtRecordingExtractor(folder_path=stub_folder_path).read(
            events=["Dv1A"], outputPath=str(tmp_path)
        )
        first_data = first_result[0]["data"]

        self.extractor_instance.stub(stub_folder_path=stub_folder_path)
        second_result = TdtRecordingExtractor(folder_path=stub_folder_path).read(
            events=["Dv1A"], outputPath=str(tmp_path)
        )
        second_data = second_result[0]["data"]

        np.testing.assert_array_equal(first_data, second_data)

    def test_stub_ttl_timestamps_within_duration(self, tmp_path):
        stub_duration_in_seconds = 100.0
        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(
            stub_folder_path=stub_folder_path,
            stub_duration_in_seconds=stub_duration_in_seconds,
        )
        stubbed_extractor = TdtRecordingExtractor(folder_path=stub_folder_path)

        original_control_result = self.extractor_instance.read(events=[self.control_event], outputPath=str(tmp_path))
        first_continuous_timestamp = original_control_result[0]["timestamps"][0]
        cutoff_timestamp = first_continuous_timestamp + stub_duration_in_seconds

        original_ttl_result = self.extractor_instance.read(events=[self.ttl_event], outputPath=str(tmp_path))
        original_ttl_timestamps = original_ttl_result[0]["timestamps"]

        stubbed_ttl_result = stubbed_extractor.read(events=[self.ttl_event], outputPath=str(tmp_path))
        stubbed_ttl_timestamps = stubbed_ttl_result[0]["timestamps"]

        assert np.all(stubbed_ttl_timestamps <= cutoff_timestamp)
        assert len(stubbed_ttl_timestamps) < len(original_ttl_timestamps)
        assert np.all(np.isin(stubbed_ttl_timestamps, original_ttl_timestamps))

    @pytest.mark.parametrize("stub_duration_in_seconds", [0.5, 1.0, 2.0])
    def test_stub_duration(self, tmp_path, stub_duration_in_seconds):
        stub_folder_path = tmp_path / "stubbed"
        self.extractor_instance.stub(
            stub_folder_path=stub_folder_path,
            stub_duration_in_seconds=stub_duration_in_seconds,
        )
        stubbed_extractor = TdtRecordingExtractor(folder_path=stub_folder_path)
        stubbed_result = stubbed_extractor.read(events=["Dv1A"], outputPath=str(tmp_path))
        stubbed_data = stubbed_result[0]["data"]
        stubbed_timestamps = stubbed_result[0]["timestamps"]
        sampling_rate = stubbed_result[0]["sampling_rate"]

        duration_in_seconds = stubbed_timestamps[-1] - stubbed_timestamps[0]
        assert duration_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)

        duration_from_samples_in_seconds = len(stubbed_data) / sampling_rate
        assert duration_from_samples_in_seconds == pytest.approx(stub_duration_in_seconds, abs=0.2)
