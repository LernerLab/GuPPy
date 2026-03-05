"""Contract tests for NpmRecordingExtractor."""

import os

from guppy.extractors.npm_recording_extractor import NpmRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestNpmRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Neurophotometrics", "sampleData_NPM_1")
    extractor_instance = NpmRecordingExtractor(folder_path)
    expected_events = ["file0_chev1", "file0_chod1"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
