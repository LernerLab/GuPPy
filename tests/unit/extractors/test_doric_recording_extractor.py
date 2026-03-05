"""Contract tests for DoricRecordingExtractor."""

import os

from guppy.extractors.doric_recording_extractor import DoricRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")

_EVENT_NAME_TO_EVENT_TYPE = {
    "AIn-1 - Raw": "control",
    "AIn-2 - Raw": "signal",
    "DI--O-1": "ttl",
}


class TestDoricRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Doric", "sample_doric_1")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE)
    expected_events = ["AIn-1 - Raw", "AIn-2 - Raw", "DI--O-1"]
    discover_kwargs = {}
    control_event = "AIn-1 - Raw"
    signal_event = "AIn-2 - Raw"
    ttl_event = "DI--O-1"

    @property
    def expected_control_timestamps(self):
        result = self.extractor_instance.read(events=["AIn-1 - Raw"], outputPath="")
        return result[0]["timestamps"]

    @property
    def expected_control_data(self):
        result = self.extractor_instance.read(events=["AIn-1 - Raw"], outputPath="")
        return result[0]["data"]

    @property
    def expected_signal_timestamps(self):
        result = self.extractor_instance.read(events=["AIn-2 - Raw"], outputPath="")
        return result[0]["timestamps"]

    @property
    def expected_signal_data(self):
        result = self.extractor_instance.read(events=["AIn-2 - Raw"], outputPath="")
        return result[0]["data"]

    @property
    def expected_ttl_timestamps(self):
        result = self.extractor_instance.read(events=["DI--O-1"], outputPath="")
        return result[0]["timestamps"]
