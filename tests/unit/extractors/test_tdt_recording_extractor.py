"""Contract tests for TdtRecordingExtractor."""

import os
import tempfile

from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

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

    @property
    def expected_control_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.extractor_instance.read(events=["Dv1A"], outputPath=tmp)
        return result[0]["timestamps"]

    @property
    def expected_control_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.extractor_instance.read(events=["Dv1A"], outputPath=tmp)
        return result[0]["data"]

    @property
    def expected_signal_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.extractor_instance.read(events=["Dv2A"], outputPath=tmp)
        return result[0]["timestamps"]

    @property
    def expected_signal_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.extractor_instance.read(events=["Dv2A"], outputPath=tmp)
        return result[0]["data"]

    @property
    def expected_ttl_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.extractor_instance.read(events=["PrtN"], outputPath=tmp)
        return result[0]["timestamps"]
