"""Contract tests for DoricRecordingExtractor."""

import os

from guppy.extractors.doric_recording_extractor import DoricRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")

_EVENT_NAME_TO_EVENT_TYPE = {
    "AIn-1 - Raw": "control",
    "AIn-2 - Raw": "signal",
    "DI--O-1": "DI--O-1",
}


class TestDoricRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = DoricRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Doric", "sample_doric_1")
    extractor_instance = DoricRecordingExtractor(folder_path, _EVENT_NAME_TO_EVENT_TYPE)
    expected_events = ["AIn-1 - Raw", "AIn-2 - Raw"]
    discover_kwargs = {}
