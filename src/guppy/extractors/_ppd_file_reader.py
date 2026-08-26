"""Reader for the pyPhotometry ``.ppd`` binary format.

A ``.ppd`` file is a two-byte little-endian header length, a header, and then interleaved unsigned
16-bit little-endian words whose top fifteen bits are an analog sample and whose bottom bit is a digital
line. Nothing else in the file says how those words are laid out: the ``mode`` string does, and the
mapping from mode to layout is the table below.

Six header generations exist and three of them spell the modes differently, so the reader dispatches on
``mode`` rather than on the presence of ``n_analog_signals``, which only the newest generation carries.
The strings are mutually unique across the three vocabularies, so one flat table resolves them all;
keying on the header version first would misread the files that carry a prose name under version 0.1,
since the rename that introduced those names did not bump it.

An unrecognized mode is refused rather than read with the two-signal default the format documents. A fork
of the acquisition software exists whose file is indistinguishable from an ordinary two-signal recording
except by its mode string, and reading it the documented way returns interleaved colours that look like a
trace and are not. That fork is named in ``_FORKED_MODES`` so its message can say what the file is instead
of calling it unknown; reading it is not supported.

Vendored from NeuroConv, where it lives at
``src/neuroconv/datainterfaces/fiber_photometry/pyphotometry/_file_reader.py``. It is copied rather
than imported because it is a private module there, and because every GuPPy extractor is otherwise
self-contained in its own parsing. Keep the mode table below in step with the NeuroConv copy; the
intent is to drop this file once the reader is public API upstream.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The header stores the LED-off baseline beside the LED-on sample from this version onward.
_PAIRED_SAMPLE_VERSION = (1, 1)

#: What the analog value is multiplied by when the pre-JSON header packs it as an integer.
_LEGACY_VOLTS_PER_DIVISION_SCALE = 1e9

#: Length of the fixed-layout header that predates the JSON one.
_LEGACY_HEADER_LENGTH = 42


@dataclass(frozen=True)
class _ModeLayout:
    """How one ``mode`` string lays its samples out.

    Attributes
    ----------
    slot_mapping : tuple of (int, int)
        One entry per slot of the sampling cycle, in the order the slots occupy it, giving the index of
        the photodetector read and of the excitation source lit. Read from the firmware's ``set_mode``
        and its interrupt service routines, which label every branch with the pair it handles. A slot is
        not a socket: the strobed one-emission modes read the same photodetector twice under different
        excitations, and ``3EX_2EM_pulsed`` visits detector 1, detector 2, detector 1 in that order.
    pulsed : bool
        Whether the excitation sources are strobed, which decides whether the signals are staggered in
        time.
    """

    slot_mapping: tuple[tuple[int, int], ...]
    pulsed: bool

    @property
    def analog_input_count(self) -> int:
        """How many slots one sampling cycle holds, which is the stride."""
        return len(self.slot_mapping)


# Every mode string the acquisition software has ever written. The modern symbolic names, the prose names
# of versions 0.2 and 0.3, and the indicator names that predate them all name the same handful of layouts,
# and the strings are mutually unique, so the table is keyed on the string alone: dispatching on the
# header version first would misread the files that carry a prose name under version 0.1, since the rename
# did not bump it.
#
# Each entry was read from `set_mode` and the interrupt service routines of the firmware, at the tag that
# introduced its vocabulary, and cross-checked against the recordings where any are held. `GCaMP/iso` is
# the one entry with no recording behind it anywhere: it is in the firmware and in the pre-JSON header
# writer's own mode table, but no file carrying it has been found.
#: The three slot layouts the format has ever used, as (photodetector, excitation source) per slot. Every
#: mode is one of these: two detectors each under their own source, one detector under two sources in
#: turn, or three sources across two detectors in the order the firmware's interrupt visits them.
_DIAGONAL = ((0, 0), (1, 1))
_ONE_DETECTOR = ((0, 0), (0, 1))
_THREE_SOURCES = ((0, 0), (1, 1), (0, 2))

_MODE_LAYOUTS = {
    # Modern, version 1.0 and later.
    "2EX_2EM_continuous": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=False),
    "2EX_1EM_pulsed": _ModeLayout(slot_mapping=_ONE_DETECTOR, pulsed=True),
    "2EX_2EM_pulsed": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=True),
    "3EX_2EM_pulsed": _ModeLayout(slot_mapping=_THREE_SOURCES, pulsed=True),
    # Prose names, versions 0.2 and 0.3.
    "1 colour time div.": _ModeLayout(slot_mapping=_ONE_DETECTOR, pulsed=True),
    "2 colour time div.": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=True),
    "2 colour continuous": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=False),
    # Indicator names, used by the software that predates the first tagged release. The mode carries the
    # layout here as well: the continuous one runs at 1 kHz and the strobed ones at 130 Hz.
    "GCaMP/RFP": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=False),
    "GCaMP/iso": _ModeLayout(slot_mapping=_ONE_DETECTOR, pulsed=True),
    "GCaMP/RFP_dif": _ModeLayout(slot_mapping=_DIAGONAL, pulsed=True),
}

#: What the pre-JSON header's mode byte indexes, from the header writer of that generation
#: (``photometry_host.py`` at commit 87c7d084, before commit c0182a88 replaced the layout with JSON).
_LEGACY_MODE_CODES = {1: "GCaMP/RFP", 2: "GCaMP/iso", 3: "GCaMP/RFP_dif"}

#: Modes written by a fork of the acquisition software rather than by any pyPhotometry release, mapped to
#: what to say when one turns up. They are listed rather than left to the unknown-mode path so that the
#: message names what the file is, which is the difference between a user knowing to ask and a user
#: assuming the format is unsupported.
_FORKED_MODES = {
    "4 colour time div.": (
        "This recording was written by the Wiegert-lab fork of the pyPhotometry acquisition software "
        "(Formozov, Dieter and Wiegert 2023, Cell Reports Methods 3:100418), whose analog lines each "
        "alternate two excitation sources, so it holds four signals at half the rate its header states. "
        "Reading it with the layout the format documents returns two traces of interleaved colours that "
        "look like signals and are not, so it is refused rather than guessed at. Support can be added: "
        "please open an issue at https://github.com/LernerLab/GuPPy/issues."
    ),
}


@dataclass
class PPDAnalogSignal:
    """One analog signal, de-interleaved from the file's words.

    Attributes
    ----------
    analog_input : int
        Which slot of the sampling cycle this signal occupies.
    detector_index : int
        Which photodetector was read for this slot, counting from zero.
    excitation_index : int
        Which excitation source was lit for this slot, counting from zero.
    data_in_volts : numpy.ndarray
        The signal, scaled by the header's volts per division. On a paired file this is the LED-on
        sample minus its baseline, which is what the acquisition system used to compute on the board.
    starting_time_in_seconds : float
        When this signal's first sample was taken. Signals are sampled one after another rather than at
        once, so this is what distinguishes them in time; the samples themselves are perfectly regular.
    rate_in_hz : float
        The signal's own sampling rate, which is the rate the header states.
    raw_led_on_in_volts : numpy.ndarray or None
        The LED-on measurement, on a paired file only.
    raw_baseline_in_volts : numpy.ndarray or None
        The LED-off measurement taken beside it, on a paired file only. This is a genuine measurement of
        ambient light and detector offset, so it is kept rather than discarded into the subtraction.
    """

    analog_input: int
    detector_index: int
    excitation_index: int
    data_in_volts: np.ndarray
    starting_time_in_seconds: float
    rate_in_hz: float
    raw_led_on_in_volts: np.ndarray | None = None
    raw_baseline_in_volts: np.ndarray | None = None


@dataclass
class PPDDigitalSignal:
    """One digital line, carried in the low bit of the words of its analog line's slot."""

    digital_input: int
    data: np.ndarray
    starting_time_in_seconds: float
    rate_in_hz: float


@dataclass
class PPDRecording:
    """Everything a ``.ppd`` file holds."""

    header: dict
    sampling_rate_in_hz: float
    analog_signals: list[PPDAnalogSignal]
    digital_signals: list[PPDDigitalSignal]
    has_paired_samples: bool
    pulsed: bool


def _parse_version(header: dict) -> tuple[int, ...]:
    """Read the header version as comparable integers.

    The field is a JSON number in versions 0.1 and 0.2 and a string from 0.3 onward, so ``"1.0" > "0.3"``
    style comparisons and float comparisons are both wrong on some part of the corpus.
    """
    version = header.get("version")
    if version is None:
        return ()
    parts = str(version).split(".")
    return tuple(int(part) for part in parts if part.isdigit())


def _read_legacy_header(header_bytes: bytes) -> dict:
    """Read the fixed-layout header that predates the JSON one.

    Two files out of 380 use it, both from the 2018 manuscript data. A failed JSON parse is a version
    signal rather than a corrupt file, so this is a fallback rather than an error path.

    The two text fields are fixed-width slices rather than a delimited pair: the writer of this
    generation packs the subject with ``ljust(12)`` and the timestamp straight after it, and the
    acquisition GUI caps that field at exactly twelve characters. So a twelve-character subject leaves no
    separator at all and an eleven-character one leaves a single space, and splitting on a run of spaces
    would swallow the timestamp on either.
    """
    subject_id = header_bytes[0:12].decode("utf-8").strip()
    date_time = header_bytes[12:31].decode("utf-8").strip()
    volts_per_division = [
        int.from_bytes(header_bytes[34:38], "little") / _LEGACY_VOLTS_PER_DIVISION_SCALE,
        int.from_bytes(header_bytes[38:42], "little") / _LEGACY_VOLTS_PER_DIVISION_SCALE,
    ]
    return {
        "subject_ID": subject_id,
        "date_time": date_time,
        "mode_code": header_bytes[31],
        "sampling_rate": int.from_bytes(header_bytes[32:34], "little"),
        "volts_per_division": volts_per_division,
    }


def _read_header(raw: bytes) -> tuple[dict, bytes]:
    """Split a file into its header and its words, whichever header generation it uses."""
    header_length = int.from_bytes(raw[:2], "little")
    header_bytes = raw[2 : 2 + header_length]
    payload = raw[2 + header_length :]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        if header_length != _LEGACY_HEADER_LENGTH:
            raise ValueError(
                f"Could not read the header: it is neither JSON nor the {_LEGACY_HEADER_LENGTH}-byte "
                f"fixed layout that precedes it (the declared header length is {header_length})."
            )
        header = _read_legacy_header(header_bytes)
    return header, payload


def _get_layout(header: dict) -> _ModeLayout:
    """Look the mode up, and refuse rather than guess when it is not in the table."""
    mode = header.get("mode")
    if mode is None and "mode_code" in header:
        # The pre-JSON header packs the mode as a byte indexing the three modes that generation offered,
        # so it resolves to the same layouts as every later vocabulary rather than needing a default.
        mode_code = header["mode_code"]
        mode = _LEGACY_MODE_CODES.get(mode_code)
        if mode is None:
            raise ValueError(
                f"Unknown pyPhotometry mode code {mode_code} in the pre-JSON header. That byte indexes "
                f"the acquisition modes of the software that wrote this generation, and only "
                f"{sorted(_LEGACY_MODE_CODES)} were ever assigned."
            )

    if mode in _FORKED_MODES:
        raise ValueError(_FORKED_MODES[mode])

    layout = _MODE_LAYOUTS.get(mode)
    if layout is None:
        raise ValueError(
            f"Unknown pyPhotometry acquisition mode '{mode}'. The layout of the samples is decided by "
            f"the mode, and reading an unknown one with the default two-signal layout can interleave "
            f"several colors into one trace without raising. Known modes: {sorted(_MODE_LAYOUTS)}."
        )

    return layout


def _volts_per_division(header: dict, analog_input: int) -> float:
    volts_per_division = header["volts_per_division"]
    if isinstance(volts_per_division, (int, float)):
        return float(volts_per_division)
    return float(volts_per_division[analog_input])


def read_ppd(file_path: Path | str) -> PPDRecording:
    """Read a pyPhotometry ``.ppd`` file.

    Parameters
    ----------
    file_path : path
        The ``.ppd`` file.

    Returns
    -------
    PPDRecording
        The header, one entry per analog signal with its own rate and starting time, and one per digital
        line.

    Notes
    -----
    Every signal is regular, so each carries a rate and a starting time rather than an array of
    timestamps, and what separates two signals is a start time alone. In the pulsed modes the sampling
    timer runs at ``sampling_rate * analog_input_count`` and the interrupt advances one line per tick, so
    signal *i* starts exactly *i* ticks in, where the upstream reader reports every signal as starting at
    zero. In the continuous modes the conversions are also sequential rather than simultaneous, but by an
    amount the format does not record and the firmware only implies, so those signals start at zero here
    until the figure can be sourced rather than derived.
    """
    raw = Path(file_path).read_bytes()
    header, payload = _read_header(raw)
    layout = _get_layout(header)

    words = np.frombuffer(payload, dtype="<u2")
    analog_words = (words >> 1).astype(np.float64)
    digital_bits = (words & 1).astype(np.uint8)

    sampling_rate = float(header["sampling_rate"])
    has_paired_samples = layout.pulsed and _parse_version(header) >= _PAIRED_SAMPLE_VERSION
    if has_paired_samples:
        # Say it rather than hand back traces that look like every other generation's: from version
        # 1.1 a strobed recording stores an LED-on sample and the baseline beside it, and no file of
        # that version was available when this was written, so the layout comes from the vendor's own
        # reader rather than from a recording.
        warnings.warn(
            "This recording states header version 1.1 or later, where a strobed mode stores an LED-on "
            "sample and the LED-off baseline beside it, and the trace written here is their difference. "
            "No file of that version was available when this reader was written, so this path is "
            "untested and its output is less certain than for the other recordings the format has. If "
            "you have such a recording, please open an issue at "
            "https://github.com/LernerLab/GuPPy/issues so we can test this path and improve it.",
            UserWarning,
            stacklevel=3,
        )
    words_per_cycle = layout.analog_input_count * (2 if has_paired_samples else 1)

    # In the pulsed modes the timer ticks once per slot per sample, so a slot's position in the cycle is
    # how far into the recording it starts. Every slot is visited once per cycle, so each signal samples
    # at the rate the header states.
    timer_frequency = sampling_rate * layout.analog_input_count
    signal_rate = sampling_rate

    analog_signals = []
    for slot in range(layout.analog_input_count):
        offset = slot * (2 if has_paired_samples else 1)
        word_indices = np.arange(offset, len(words), words_per_cycle)
        detector, excitation = layout.slot_mapping[slot]
        scale = _volts_per_division(header, detector)

        if has_paired_samples:
            # From version 1.1 the firmware stores the LED-on sample and the LED-off baseline it was
            # measured against, and the subtraction the board used to do moved into the reader.
            baseline_indices = word_indices + 1
            usable = baseline_indices < len(words)
            word_indices, baseline_indices = word_indices[usable], baseline_indices[usable]
            led_on = analog_words[word_indices] * scale
            baseline = analog_words[baseline_indices] * scale
            data = led_on - baseline
        else:
            led_on = baseline = None
            data = analog_words[word_indices] * scale

        analog_signals.append(
            PPDAnalogSignal(
                analog_input=slot,
                detector_index=detector,
                excitation_index=excitation,
                data_in_volts=data,
                starting_time_in_seconds=slot / timer_frequency if layout.pulsed else 0.0,
                rate_in_hz=signal_rate,
                raw_led_on_in_volts=led_on,
                raw_baseline_in_volts=baseline,
            )
        )

    # A digital line rides in the low bit of the words of the analog line it shares a slot with, so it
    # starts when that line does. Headers before 1.0 do not count the lines; both are present.
    digital_count = int(header.get("n_digital_signals", 2))
    digital_signals = []
    for digital_input in range(min(digital_count, layout.analog_input_count)):
        offset = digital_input * (2 if has_paired_samples else 1)
        word_indices = np.arange(offset, len(words), words_per_cycle)
        digital_signals.append(
            PPDDigitalSignal(
                digital_input=digital_input,
                data=digital_bits[word_indices],
                starting_time_in_seconds=digital_input / timer_frequency if layout.pulsed else 0.0,
                rate_in_hz=signal_rate,
            )
        )

    return PPDRecording(
        header=header,
        sampling_rate_in_hz=sampling_rate,
        analog_signals=analog_signals,
        digital_signals=digital_signals,
        has_paired_samples=has_paired_samples,
        pulsed=layout.pulsed,
    )
