"""Mixin of contract tests for BaseRecordingExtractor subclasses."""

import h5py
import numpy as np


class RecordingExtractorTestMixin:
    """
    Mixin of contract tests for BaseRecordingExtractor subclasses.

    Child test classes must define the following class-level attributes:

    extractor_class : type
        The concrete extractor class under test.
    folder_path : str
        Folder path passed to ``discover_events_and_flags`` and the constructor.
    expected_events : list of str
        At least one event name known to be discoverable in ``folder_path``.
    discover_kwargs : dict
        Extra keyword arguments for ``discover_events_and_flags()`` beyond
        ``folder_path``. Use ``{}`` for TDT/Doric/CSV; NPM needs ``{"num_ch": N}``.
    extractor_instance : BaseRecordingExtractor
        An initialized instance of the extractor under test. Can be a class
        variable when the extractor has no mutable state, or set in
        ``setup_method`` when a fresh instance per test is needed.

    Child test classes must also implement the ``expected_timestamps`` property,
    which returns the timestamps array for ``expected_events[0]`` as it should
    appear in the saved HDF5 file. Each format may require different logic (e.g.
    reading from a CSV, calling ``extractor_instance.read()``, etc.).
    """

    @property
    def expected_timestamps(self):
        raise NotImplementedError("Child test classes must implement the expected_timestamps property.")

    # --- discover tests ---

    def test_discover_returns_two_element_tuple(self):
        result = self.extractor_class.discover_events_and_flags(self.folder_path, **self.discover_kwargs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_discover_first_element_is_list_of_strings(self):
        events, _ = self.extractor_class.discover_events_and_flags(self.folder_path, **self.discover_kwargs)
        assert isinstance(events, list)
        assert all(isinstance(event, str) for event in events)

    def test_discover_second_element_is_list_of_strings(self):
        _, flags = self.extractor_class.discover_events_and_flags(self.folder_path, **self.discover_kwargs)
        assert isinstance(flags, list)
        assert all(isinstance(flag, str) for flag in flags)

    def test_discover_includes_expected_events(self):
        events, _ = self.extractor_class.discover_events_and_flags(self.folder_path, **self.discover_kwargs)
        for expected_event in self.expected_events:
            assert expected_event in events

    # --- read tests ---

    def test_read_returns_list_of_dicts(self, tmp_path):
        result = self.extractor_instance.read(events=self.expected_events, outputPath=str(tmp_path))
        assert isinstance(result, list)
        assert all(isinstance(output_dict, dict) for output_dict in result)

    def test_read_output_dicts_have_storename_string(self, tmp_path):
        result = self.extractor_instance.read(events=self.expected_events, outputPath=str(tmp_path))
        for output_dict in result:
            assert "storename" in output_dict
            assert isinstance(output_dict["storename"], str)
            assert len(output_dict["storename"]) > 0

    # --- save tests ---

    def test_save_produces_hdf5_files(self, tmp_path):
        output_dicts = self.extractor_instance.read(events=self.expected_events, outputPath=str(tmp_path))
        self.extractor_instance.save(output_dicts=output_dicts, outputPath=str(tmp_path))

        for output_dict in output_dicts:
            sanitized_storename = output_dict["storename"].replace("\\", "_").replace("/", "_")
            assert (tmp_path / f"{sanitized_storename}.hdf5").exists()

    def test_save_hdf5_has_timestamps_dataset(self, tmp_path):
        output_dicts = self.extractor_instance.read(events=self.expected_events, outputPath=str(tmp_path))
        self.extractor_instance.save(output_dicts=output_dicts, outputPath=str(tmp_path))

        for output_dict in output_dicts:
            sanitized_storename = output_dict["storename"].replace("\\", "_").replace("/", "_")
            with h5py.File(tmp_path / f"{sanitized_storename}.hdf5", "r") as file:
                assert "timestamps" in file

    # --- roundtrip test ---

    def test_roundtrip_timestamps_preserved(self, tmp_path):
        first_event = self.expected_events[0]
        output_dicts = self.extractor_instance.read(events=[first_event], outputPath=str(tmp_path))
        self.extractor_instance.save(output_dicts=output_dicts, outputPath=str(tmp_path))

        sanitized_storename = first_event.replace("\\", "_").replace("/", "_")
        with h5py.File(tmp_path / f"{sanitized_storename}.hdf5", "r") as file:
            np.testing.assert_array_equal(file["timestamps"][:], self.expected_timestamps)
