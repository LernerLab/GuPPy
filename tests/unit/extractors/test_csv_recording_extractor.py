"""Contract tests for CsvRecordingExtractor."""

import os

from guppy.extractors.csv_recording_extractor import CsvRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestCsvRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = CsvRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_csv", "sample_data_csv_1")
    extractor_instance = CsvRecordingExtractor(folder_path)
    expected_events = ["Sample_Control_Channel", "Sample_Signal_Channel"]
    discover_kwargs = {}
