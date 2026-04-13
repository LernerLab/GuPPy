"""Mock recording extractor for unit testing."""

import numpy as np

from guppy.extractors.base_recording_extractor import BaseRecordingExtractor

_MOCK_EVENTS = ["mock_signal", "mock_control", "mock_ttl"]
_MOCK_SAMPLING_RATE = 100.0
_MOCK_DURATION_IN_SECONDS = 3.0


class MockRecordingExtractor(BaseRecordingExtractor):
    """
    Concrete recording extractor backed by deterministic numpy arrays.

    Intended for unit testing the BaseRecordingExtractor contract without
    requiring any real acquisition data files.

    Parameters
    ----------
    folder_path : str
        Ignored; present to match the constructor pattern of real extractors.
    """

    _stub_folder_path_to_duration: dict[str, float] = {}

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @classmethod
    def discover_events_and_flags(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Return a fixed set of mock event names and empty flags.

        Parameters
        ----------
        folder_path : str
            Ignored; present to match the signature of real extractors.

        Returns
        -------
        events : list of str
            Fixed list of mock event names.
        flags : list of str
            Always empty for the mock extractor.
        """
        return list(_MOCK_EVENTS), []

    def read(self, *, events: list[str], outputPath: str) -> list[dict]:
        """
        Return deterministic numpy arrays for each requested event.

        Parameters
        ----------
        events : list of str
            Event names to read. Each name becomes the ``storename`` in the
            returned dict.
        outputPath : str
            Ignored; present to match the BaseRecordingExtractor interface.

        Returns
        -------
        list of dict
            One dict per event, each containing:
            ``storename`` (str), ``timestamps`` (ndarray), ``data`` (ndarray),
            ``sampling_rate`` (float).
        """
        duration = MockRecordingExtractor._stub_folder_path_to_duration.get(str(self.folder_path))
        original_number_of_samples = int(_MOCK_DURATION_IN_SECONDS * _MOCK_SAMPLING_RATE)
        if duration is not None:
            number_of_samples = int(duration * _MOCK_SAMPLING_RATE)
        else:
            number_of_samples = original_number_of_samples
        output_dicts = []
        for event in events:
            output_dicts.append(
                {
                    "storename": event,
                    "timestamps": np.arange(number_of_samples, dtype=float) / _MOCK_SAMPLING_RATE,
                    "data": np.linspace(0.0, 1.0, original_number_of_samples)[:number_of_samples],
                    "sampling_rate": _MOCK_SAMPLING_RATE,
                }
            )
        return output_dicts

    def stub(self, *, folder_path, duration_in_seconds=1.0):
        """
        Register a stub duration for ``folder_path`` without writing any files.

        Subsequent ``read()`` calls on a ``MockRecordingExtractor`` initialised
        with ``folder_path`` will generate arrays of the appropriate length.

        Parameters
        ----------
        folder_path : str or Path
            The path that will be passed to the constructor of the stubbed
            extractor. The directory is not created.
        duration_in_seconds : float, optional
            Duration of mock data to generate on subsequent ``read()`` calls.
            Default is 1.0.
        """
        MockRecordingExtractor._stub_folder_path_to_duration[str(folder_path)] = duration_in_seconds

    def save(self, *, output_dicts: list[dict], outputPath: str) -> None:
        """
        Write each output dict to an HDF5 file via ``_write_hdf5``.

        Parameters
        ----------
        output_dicts : list of dict
            Dicts returned by ``read()``. Each dict must have a ``storename``
            key; all other keys are written as HDF5 datasets.
        outputPath : str
            Directory in which HDF5 files are written.
        """
        for output_dict in output_dicts:
            storename = output_dict["storename"]
            for key, value in output_dict.items():
                self._write_hdf5(value, storename, outputPath, key)
