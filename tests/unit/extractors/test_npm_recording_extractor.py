"""Contract tests for NpmRecordingExtractor."""

import os

import numpy as np

from guppy.extractors.npm_recording_extractor import NpmRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

_TESTING_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_data")


class TestNpmRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = NpmRecordingExtractor
    folder_path = os.path.join(_TESTING_DATA, "SampleData_Neurophotometrics", "sampleData_NPM_1")
    extractor_instance = NpmRecordingExtractor(folder_path)
    expected_events = ["file0_chev1", "file0_chod1", "event0"]
    discover_kwargs = {"num_ch": 2, "inputParameters": {}}
    control_event = "file0_chod1"
    signal_event = "file0_chev1"
    ttl_event = "event0"

    @property
    def expected_control_timestamps(self):
        # discover must run first to create the intermediate CSV files that read() depends on
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chod1"], outputPath="")
        return np.array(list(result[0]["timestamps"].values()))

    @property
    def expected_control_data(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chod1"], outputPath="")
        return np.array(list(result[0]["data"].values()))

    @property
    def expected_signal_timestamps(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chev1"], outputPath="")
        return np.array(list(result[0]["timestamps"].values()))

    @property
    def expected_signal_data(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["file0_chev1"], outputPath="")
        return np.array(list(result[0]["data"].values()))

    @property
    def expected_ttl_timestamps(self):
        NpmRecordingExtractor.discover_events_and_flags(self.folder_path, num_ch=2, inputParameters={})
        result = self.extractor_instance.read(events=["event0"], outputPath="")
        return np.array(list(result[0]["timestamps"].values()))
