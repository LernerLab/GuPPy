"""Reader for the pyPhotometry ``.ppd`` binary format.

A ``.ppd`` file is a two-byte little-endian header length, a header, and then interleaved unsigned
16-bit little-endian words whose top fifteen bits are an analog sample and whose bottom bit is a digital
line. Nothing else in the file says how those words are laid out: the ``mode`` string does, and the
mapping from mode to layout is the table below.

Six header generations exist and three of them spell the modes differently, so the reader dispatches on
``mode`` rather than on the presence of ``n_analog_signals``. It refuses a mode it does not know instead
of falling back to the two-signal default that the format documents, because a fork exists whose file is
indistinguishable from an ordinary two-signal recording except by its mode string, and reading it the
documented way returns interleaved colours that look like a trace and are not.

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
    analog_input_count : int
        How many analog lines the words interleave across, which is the stride.
    colors_per_input : int
        How many colors each analog line time-multiplexes. One for every upstream mode; two for the
        four-color fork, whose line alternates colors on consecutive samples.
    pulsed : bool
        Whether the LEDs are strobed, which decides whether the signals are staggered in time.
    """

    analog_input_count: int
    colors_per_input: int
    pulsed: bool


# Every mode string observed across 380 files spanning all six header generations. The modern symbolic
# names, the prose names of versions 0.2 and 0.3, and the indicator names of version 0.1 all name the
# same handful of layouts. Each entry's stride was checked against the recordings themselves rather than
# read off the documentation.
_MODE_LAYOUTS = {
    # Modern, version 1.0 and later.
    "2EX_2EM_continuous": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=False),
    "2EX_1EM_pulsed": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=True),
    "2EX_2EM_pulsed": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=True),
    "3EX_2EM_pulsed": _ModeLayout(analog_input_count=3, colors_per_input=1, pulsed=True),
    # Prose names, versions 0.2 and 0.3.
    "1 colour time div.": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=True),
    "2 colour time div.": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=True),
    "2 colour continuous": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=False),
    # The Wiegert-lab fused-fiber-coupler fork. Two analog lines, each alternating two colors, so the
    # per-color rate is half what the header advertises. Reading it as two signals is what produces a
    # trace that alternates every sample instead of a fluorescence signal.
    "4 colour time div.": _ModeLayout(analog_input_count=2, colors_per_input=2, pulsed=True),
    # Indicator names, version 0.1. The mode carries the layout here as well: the continuous one runs at
    # 1 kHz and the differential one at 130 Hz.
    "GCaMP/RFP": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=False),
    "GCaMP/RFP_dif": _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=True),
}


@dataclass
class PPDAnalogSignal:
    """One analog signal, de-interleaved from the file's words.

    Attributes
    ----------
    analog_input : int
        Which analog line this signal came off.
    color_index : int
        Which of that line's time-multiplexed colors this is; always zero outside the four-color fork.
    data_in_volts : numpy.ndarray
        The signal, scaled by the header's volts per division. On a paired file this is the LED-on
        sample minus its baseline, which is what the acquisition system used to compute on the board.
    starting_time_in_seconds : float
        When this signal's first sample was taken. Signals are sampled one after another rather than at
        once, so this is what distinguishes them in time; the samples themselves are perfectly regular.
    rate_in_hz : float
        The signal's own sampling rate, which is the header's rate divided by however many colors its
        analog line multiplexes.
    raw_led_on_in_volts : numpy.ndarray or None
        The LED-on measurement, on a paired file only.
    raw_baseline_in_volts : numpy.ndarray or None
        The LED-off measurement taken beside it, on a paired file only. This is a genuine measurement of
        ambient light and detector offset, so it is kept rather than discarded into the subtraction.
    """

    analog_input: int
    color_index: int
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
    """
    subject_and_time = header_bytes[:31].decode("utf-8").strip()
    subject_id, _, date_time = subject_and_time.partition(" " * 2)
    volts_per_division = [
        int.from_bytes(header_bytes[34:38], "little") / _LEGACY_VOLTS_PER_DIVISION_SCALE,
        int.from_bytes(header_bytes[38:42], "little") / _LEGACY_VOLTS_PER_DIVISION_SCALE,
    ]
    return {
        "subject_ID": subject_id.strip(),
        "date_time": date_time.strip(),
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
        # The pre-JSON header packs the mode as a code whose meaning is not published. Every file of this
        # generation held so far interleaves two analog lines, which is also what the format's default
        # would give, so it is read that way and stated here rather than being silently assumed. Whether
        # its LEDs were strobed is not recoverable from the code, so no stagger is claimed for it: an
        # invented offset would be worse than a missing one.
        return _ModeLayout(analog_input_count=2, colors_per_input=1, pulsed=False)

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
    signals_per_cycle = layout.analog_input_count * layout.colors_per_input
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
    words_per_cycle = signals_per_cycle * (2 if has_paired_samples else 1)

    # In the pulsed modes the timer ticks once per analog line per sample, so a signal's slot in the
    # cycle is how far into the recording it starts. A line that multiplexes colors visits each of them
    # once per cycle, so its own rate is the header's divided by that count.
    timer_frequency = sampling_rate * layout.analog_input_count
    signal_rate = sampling_rate / layout.colors_per_input

    analog_signals = []
    for color_index in range(layout.colors_per_input):
        for analog_input in range(layout.analog_input_count):
            slot = color_index * layout.analog_input_count + analog_input
            offset = slot * (2 if has_paired_samples else 1)
            word_indices = np.arange(offset, len(words), words_per_cycle)
            scale = _volts_per_division(header, analog_input)

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
                    analog_input=analog_input,
                    color_index=color_index,
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
