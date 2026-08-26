import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from guppy.extractors import BaseRecordingExtractor
from guppy.extractors._ppd_file_reader import (
    PPDAnalogSignal,
    PPDDigitalSignal,
    PPDRecording,
    _get_layout,
    _read_header,
    read_ppd,
)
from guppy.utils._hdf5_io import write_hdf5

logger = logging.getLogger(__name__)


class PyPhotometryRecordingExtractor(BaseRecordingExtractor):
    """
    Extractor for fiber photometry data recorded with pyPhotometry.

    A session folder holds one ``.ppd`` file, which carries every signal the board recorded
    interleaved into a single stream of 16-bit words. It is separated back into stores here:

    * ``detector_<n>_excitation_<m>`` — one photometry trace per slot of the sampling cycle, in
      volts, named for the photodetector that was read and the excitation source that was lit.
      A shared ``detector_<n>`` prefix means a shared optical fiber, so those stores belong to one
      recording site and should carry ``signal_``/``control_`` labels for the same site. That is the
      decision Step 1 asks you to make, and the board's own ``analog_1``/``analog_2`` get it wrong:
      they read as two sockets, while the strobed one-emission modes read a single photodetector
      twice under different excitations.
    * ``digital_<n>`` — one store per digital line, holding the onset time of each pulse.

    The board has no simultaneous analog-to-digital converters, so its slots were never sampled at
    the same instants. In the strobed modes the offset between them is exactly one tick of the
    ``sampling_rate * analog_input_count`` timer, and each store's timestamps carry it.

    The four-colour fork of the acquisition software is refused rather than read, since its layout
    is stated nowhere the file or the firmware can be asked. The error names the fork.

    On header version 1.1 and later a strobed recording stores an LED-on sample and the LED-off
    baseline beside it. Only their difference is exposed as a store, since that is the trace every
    pipeline expects and the two raw measurements have no store label they could be mapped to in
    Label Stores.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the ``.ppd`` file.
    """

    @staticmethod
    def _find_ppd_file(folder_path: str | Path) -> str:
        """Return the single ``.ppd`` file in ``folder_path``, refusing zero or several."""
        ppd_paths = sorted(glob.glob(os.path.join(str(folder_path), "*.ppd")))
        if not ppd_paths:
            message = f"No pyPhotometry '.ppd' file found in '{folder_path}'."
            logger.error(message)
            raise FileNotFoundError(message)
        if len(ppd_paths) > 1:
            names = [os.path.basename(path) for path in ppd_paths]
            message = (
                f"Found {len(ppd_paths)} '.ppd' files in '{folder_path}': {names}. A pyPhotometry "
                "session folder must hold exactly one recording; put each recording in its own folder."
            )
            logger.error(message)
            raise ValueError(message)
        return ppd_paths[0]

    @staticmethod
    def _analog_store_id(detector_index: int, excitation_index: int) -> str:
        """Name one analog store by the two devices that produced it, counting from one."""
        return f"detector_{detector_index + 1}_excitation_{excitation_index + 1}"

    @staticmethod
    def _digital_store_id(signal: PPDDigitalSignal) -> str:
        return f"digital_{signal.digital_input + 1}"

    @classmethod
    def discover_events_and_flags(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Discover the stores held by the folder's ``.ppd`` file.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the ``.ppd`` file.

        Returns
        -------
        events : list of str
            Analog trace stores followed by digital line stores.
        flags : list of str
            Always empty: the format carries no feature flags the GUI branches on.
        """
        ppd_path = cls._find_ppd_file(folder_path)
        logger.debug(f"Discovering pyPhotometry stores in {ppd_path}.")

        # Only the header is needed to name the stores, and it decides the layout, so the words are
        # not de-interleaved here — discovery runs on every session folder in step 1.
        header, _payload = _read_header(Path(ppd_path).read_bytes())
        layout = _get_layout(header)

        events = [
            cls._analog_store_id(detector_index, excitation_index)
            for detector_index, excitation_index in layout.slot_mapping
        ]

        digital_count = min(int(header.get("n_digital_signals", 2)), layout.analog_input_count)
        for digital_input in range(digital_count):
            events.append(f"digital_{digital_input + 1}")

        logger.info(f"Discovered {len(events)} pyPhotometry stores: {events}.")
        return events, []

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self._recording = None

    def _read_recording(self) -> PPDRecording:
        """Read and cache the file. One recording is a few megabytes at most, so it is read once."""
        if self._recording is None:
            self._recording = read_ppd(self._find_ppd_file(self.folder_path))
        return self._recording

    def _store_id_to_signal(self) -> dict[str, PPDAnalogSignal | PPDDigitalSignal]:
        """Map every store id onto the signal it names."""
        recording = self._read_recording()
        store_id_to_signal: dict[str, PPDAnalogSignal | PPDDigitalSignal] = {}
        for signal in recording.analog_signals:
            store_id = self._analog_store_id(signal.detector_index, signal.excitation_index)
            store_id_to_signal[store_id] = signal
        for signal in recording.digital_signals:
            store_id_to_signal[self._digital_store_id(signal)] = signal
        return store_id_to_signal

    def _signal_for_event(self, event: str) -> PPDAnalogSignal | PPDDigitalSignal:
        store_id_to_signal = self._store_id_to_signal()
        if event not in store_id_to_signal:
            message = (
                f"No pyPhotometry store named '{event}' in '{self.folder_path}'. "
                f"Available stores: {sorted(store_id_to_signal)}."
            )
            logger.error(message)
            raise ValueError(message)
        return store_id_to_signal[event]

    @staticmethod
    def _sample_timestamps(signal: PPDAnalogSignal | PPDDigitalSignal, sample_count: int) -> np.ndarray:
        """Build a store's sample times from its own start and rate.

        Every signal is regular, so its times follow from the start instant rather than being stored.
        The start is what separates two signals on this board, and it is kept rather than re-zeroed,
        so the stores stay on the acquisition clock like every other GuPPy reader.
        """
        return signal.starting_time_in_seconds + np.arange(sample_count, dtype=np.float64) / signal.rate_in_hz

    @staticmethod
    def _detect_ttl_onsets(digital: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Return the onset time of each low->high transition in a digital line.

        Matches the convention the Doric reader uses: a rising edge is a high sample whose
        immediate predecessor is low, and its onset is that sample's timestamp. A line already high
        at the first sample has no observed transition and is not reported; a pulse still high at
        the final sample IS reported, because its rising edge was observed regardless of when the
        recording stopped.
        """
        high = np.asarray(digital) > 0
        rising_edges = np.flatnonzero(~high[:-1] & high[1:]) + 1
        return np.asarray(timestamps)[rising_edges]

    def count_samples(self, *, event: str) -> int:
        """Return the number of samples the store holds, for the step-2 progress bar."""
        signal = self._signal_for_event(event)
        if isinstance(signal, PPDAnalogSignal):
            return int(signal.data_in_volts.size)
        return int(signal.data.size)

    def read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]:
        """
        Read the requested stores out of the folder's ``.ppd`` file.

        Parameters
        ----------
        events : list of str
            Store names to extract, as returned by :meth:`discover_events_and_flags`.
        outputPath : str
            Path to the output directory (unused by this extractor; required by
            the base-class interface).

        Returns
        -------
        list of dict
            One dictionary per store. Analog stores carry ``store_id``, ``timestamps``, ``data``
            and ``sampling_rate``; digital stores carry ``store_id`` and ``timestamps``, the latter
            holding pulse onsets rather than a densely sampled line.
        """
        output_dicts = []
        for event in events:
            signal = self._signal_for_event(event)
            if isinstance(signal, PPDAnalogSignal):
                timestamps = self._sample_timestamps(signal, signal.data_in_volts.size)
                output_dicts.append(
                    {
                        "store_id": event,
                        "timestamps": timestamps,
                        "data": signal.data_in_volts,
                        "sampling_rate": np.array([signal.rate_in_hz]),
                    }
                )
            else:
                timestamps = self._sample_timestamps(signal, signal.data.size)
                output_dicts.append(
                    {
                        "store_id": event,
                        "timestamps": self._detect_ttl_onsets(signal.data, timestamps),
                    }
                )
        return output_dicts

    def save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None:
        """
        Save extracted data dictionaries to HDF5 files.

        Parameters
        ----------
        output_dicts : list of dict
            Data dictionaries as returned by :meth:`read`.
        outputPath : str
            Path to the output directory where HDF5 files are written.
        """
        for output_dict in output_dicts:
            store_id = output_dict["store_id"]
            for key, value in output_dict.items():
                if key == "store_id":
                    continue
                write_hdf5(value, store_id, outputPath, key)

    def stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None:
        """
        Create a stubbed copy of the session folder with the recording truncated.

        Copies the folder, then rewrites the ``.ppd`` file keeping its header byte for byte and
        cutting the word stream at a **cycle boundary**, so every store keeps the same number of
        samples as it would have had. Cutting anywhere else would leave the slots with different
        lengths.

        Parameters
        ----------
        folder_path : str or Path
            Destination directory for the stubbed folder. Created if it does not
            exist; overwritten if it already exists.
        duration_in_seconds : float, optional
            Approximate duration of data to retain in seconds. Default is 1.0.
        """
        folder_path = Path(folder_path)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        shutil.copytree(self.folder_path, folder_path)

        source_path = Path(self._find_ppd_file(self.folder_path))
        destination_path = folder_path / source_path.name

        raw = source_path.read_bytes()
        header_length = int.from_bytes(raw[:2], "little")
        header_and_length = raw[: 2 + header_length]
        payload = raw[2 + header_length :]

        header, _payload = _read_header(raw)
        layout = _get_layout(header)
        recording = self._read_recording()
        words_per_cycle = layout.analog_input_count * (2 if recording.has_paired_samples else 1)

        # A cycle carries one sample of every signal, and every slot is visited once per cycle, so each
        # signal samples at the rate the header states.
        signal_rate = float(header["sampling_rate"])
        cycles_to_keep = max(1, int(round(duration_in_seconds * signal_rate)))
        bytes_to_keep = cycles_to_keep * words_per_cycle * 2
        destination_path.write_bytes(header_and_length + payload[:bytes_to_keep])
