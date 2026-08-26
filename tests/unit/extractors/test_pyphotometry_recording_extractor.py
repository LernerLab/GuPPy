"""Contract and mode-table tests for PyPhotometryRecordingExtractor."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from guppy.extractors.pyphotometry_recording_extractor import (
    PyPhotometryRecordingExtractor,
)
from guppy_test_data import STUBBED_TESTING_DATA

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

PYPHOTOMETRY_DATA = os.path.join(STUBBED_TESTING_DATA, "pyphotometry")


def _session(name: str) -> str:
    return os.path.join(PYPHOTOMETRY_DATA, name)


def _deinterleave_two_signal_file(folder_path: str) -> dict:
    """Separate a two-analog-input ``.ppd`` file without going through the reader under test.

    Written out longhand from the format description rather than reused from
    ``_ppd_file_reader``, so the expected arrays below are an independent computation and the
    round-trip tests are not comparing the reader against itself.
    """
    ppd_path = next(Path(folder_path).glob("*.ppd"))
    raw = ppd_path.read_bytes()
    header_length = int.from_bytes(raw[:2], "little")
    header = json.loads(raw[2 : 2 + header_length].decode("utf-8"))
    words = np.frombuffer(raw[2 + header_length :], dtype="<u2")

    analog_words = (words >> 1).astype(np.float64)
    digital_bits = (words & 1).astype(np.uint8)
    volts_per_division = header["volts_per_division"]
    rate = float(header["sampling_rate"])
    # Two analog inputs, one tick of the rate*2 timer apart while the LEDs are strobed.
    tick = 1.0 / (rate * 2)

    result = {}
    for analog_input in (0, 1):
        data = analog_words[analog_input::2] * volts_per_division[analog_input]
        timestamps = analog_input * tick + np.arange(data.size, dtype=np.float64) / rate
        high = digital_bits[analog_input::2] > 0
        rising_edges = np.flatnonzero(~high[:-1] & high[1:]) + 1
        result[analog_input] = {
            "data": data,
            "timestamps": timestamps,
            "onsets": timestamps[rising_edges],
        }
    return result


class TestPyPhotometryRecordingExtractor(RecordingExtractorTestMixin):
    extractor_class = PyPhotometryRecordingExtractor
    folder_path = _session("two_excitation_two_emission_pulsed")
    extractor_instance = PyPhotometryRecordingExtractor(folder_path)
    expected_events = [
        "detector_1_excitation_1",
        "detector_2_excitation_2",
        "digital_1",
        "digital_2",
    ]
    discover_kwargs = {}
    control_event = "detector_2_excitation_2"
    signal_event = "detector_1_excitation_1"
    ttl_event = "digital_1"
    stub_ttl_test_duration_in_seconds = 8.0

    @pytest.fixture(scope="class")
    def reference(self):
        return _deinterleave_two_signal_file(self.folder_path)

    @pytest.fixture
    def expected_control_timestamps(self, reference):
        return reference[1]["timestamps"]

    @pytest.fixture
    def expected_control_data(self, reference):
        return reference[1]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self, reference):
        return reference[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self, reference):
        return reference[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self, reference):
        return reference[0]["onsets"]


# ---------------------------------------------------------------------------
# The mode table: every header generation names the same layouts differently,
# and the layout is the only thing that says how to separate the words.
# ---------------------------------------------------------------------------


TWO_DETECTORS = ["detector_1_excitation_1", "detector_2_excitation_2", "digital_1", "digital_2"]
ONE_DETECTOR = ["detector_1_excitation_1", "detector_1_excitation_2", "digital_1", "digital_2"]


@pytest.mark.parametrize(
    "session, expected_events, expected_rate",
    [
        # Modern symbolic names, header version 1.0 and later.
        ("two_excitation_two_emission_pulsed", TWO_DETECTORS, 130.0),
        # Prose names, header versions 0.2 and 0.3.
        ("two_colour_time_division", TWO_DETECTORS, 130.0),
        ("two_colour_continuous", TWO_DETECTORS, 1000.0),
        # One photodetector strobed under two excitation sources, which is the signal-plus-isosbestic
        # configuration and the case the store names exist to make visible.
        ("narrow_pulses_and_idle_line", ONE_DETECTOR, 130.0),
        # Indicator names, used by the software that predates the first tagged release.
        ("gcamp_rfp_dif", TWO_DETECTORS, 130.0),
        # The fixed-layout header that predates the JSON one, whose mode byte indexes the same modes.
        ("two_signals_200hz", TWO_DETECTORS, 200.0),
    ],
)
def test_discovery_and_rate_per_header_generation(session, expected_events, expected_rate):
    folder_path = _session(session)
    events, flags = PyPhotometryRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
    assert events == expected_events
    assert flags == []

    extractor = PyPhotometryRecordingExtractor(folder_path=folder_path)
    output_dicts = extractor.read(events=expected_events[:2], outputPath=folder_path)
    for output_dict in output_dicts:
        assert output_dict["sampling_rate"][0] == pytest.approx(expected_rate)


def test_four_colour_fork_is_refused_with_a_message_naming_the_fork():
    """The fork alternates two excitation sources on each analog line, at half the advertised rate.

    Its file is indistinguishable from an ordinary two-signal recording except by its mode string,
    and the layout is stated nowhere the file or the firmware can be asked, so it is refused. The
    message names the fork and its paper instead of calling the mode unknown, which is what tells a
    user with such a recording to ask rather than assume the format is unsupported.
    """
    folder_path = _session("four_colour_time_division")
    with pytest.raises(ValueError, match="Wiegert-lab fork"):
        PyPhotometryRecordingExtractor.discover_events_and_flags(folder_path=folder_path)


def test_unknown_mode_is_refused_rather_than_read_with_the_default_layout(tmp_path):
    """An unrecognized mode must raise. Guessing is how a fork's file is silently misread."""
    source_path = next(Path(_session("two_colour_time_division")).glob("*.ppd"))
    raw = source_path.read_bytes()
    header_length = int.from_bytes(raw[:2], "little")
    header = json.loads(raw[2 : 2 + header_length].decode("utf-8"))
    header["mode"] = "7 colour interpretive dance"

    header_bytes = json.dumps(header).encode("utf-8")
    session_folder = tmp_path / "unknown_mode"
    session_folder.mkdir()
    (session_folder / "unknown_mode.ppd").write_bytes(
        len(header_bytes).to_bytes(2, "little") + header_bytes + raw[2 + header_length :]
    )

    with pytest.raises(ValueError, match="Unknown pyPhotometry acquisition mode"):
        PyPhotometryRecordingExtractor.discover_events_and_flags(folder_path=str(session_folder))


# ---------------------------------------------------------------------------
# Digital lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session, event, expected_onset_count",
    [
        # An ordinary pulse train, beside a line that was recorded and never fired.
        ("narrow_pulses_and_idle_line", "digital_1", 37),
        ("narrow_pulses_and_idle_line", "digital_2", 0),
        # A pulse already high at the first sample has no observed onset and is not counted; a pulse
        # still high at the last sample IS counted, because its rising edge was observed.
        ("starts_and_ends_high", "digital_1", 33),
        # Wide pulses on both lines.
        ("wide_pulses_on_both_lines", "digital_1", 2),
        ("wide_pulses_on_both_lines", "digital_2", 2),
    ],
)
def test_digital_lines_report_pulse_onsets(session, event, expected_onset_count):
    folder_path = _session(session)
    extractor = PyPhotometryRecordingExtractor(folder_path=folder_path)
    output_dict = extractor.read(events=[event], outputPath=folder_path)[0]

    assert set(output_dict) == {"store_id", "timestamps"}
    assert output_dict["timestamps"].size == expected_onset_count


def test_digital_lines_are_staggered_like_the_slots_they_ride_on():
    """A digital line rides in the low bit of its slot's words, so it starts when that slot does."""
    folder_path = _session("wide_pulses_on_both_lines")
    extractor = PyPhotometryRecordingExtractor(folder_path=folder_path)
    first, second = (extractor._signal_for_event(event) for event in ("digital_1", "digital_2"))

    # One tick of the 260 Hz sampling timer: two slots at the header's 130 Hz.
    expected_offset = 1.0 / (130.0 * 2)
    assert second.starting_time_in_seconds - first.starting_time_in_seconds == pytest.approx(expected_offset)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_strobed_slots_are_staggered_by_one_timer_tick():
    """The board has no simultaneous converters; a strobed recording's slots are one tick apart."""
    folder_path = _session("two_colour_time_division")
    extractor = PyPhotometryRecordingExtractor(folder_path=folder_path)
    first, second = extractor.read(events=TWO_DETECTORS[:2], outputPath=folder_path)

    expected_offset = 1.0 / (130.0 * 2)
    assert second["timestamps"][0] - first["timestamps"][0] == pytest.approx(expected_offset)


def test_continuous_mode_shares_the_headers_timebase():
    """In continuous mode the conversions are still sequential, but by an amount the file does not
    record, so no offset is claimed rather than inventing one."""
    folder_path = _session("two_colour_continuous")
    extractor = PyPhotometryRecordingExtractor(folder_path=folder_path)
    first, second = extractor.read(events=TWO_DETECTORS[:2], outputPath=folder_path)

    assert first["timestamps"][0] == 0.0
    assert second["timestamps"][0] == 0.0


# ---------------------------------------------------------------------------
# Session-folder shape
# ---------------------------------------------------------------------------


def test_folder_without_a_ppd_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="No pyPhotometry '.ppd' file found"):
        PyPhotometryRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path))


def test_folder_with_several_ppd_files_is_refused(tmp_path):
    source_path = next(Path(_session("two_colour_time_division")).glob("*.ppd"))
    for name in ("first.ppd", "second.ppd"):
        (tmp_path / name).write_bytes(source_path.read_bytes())

    with pytest.raises(ValueError, match="must hold exactly one recording"):
        PyPhotometryRecordingExtractor.discover_events_and_flags(folder_path=str(tmp_path))


def test_unknown_store_name_names_the_available_stores():
    extractor = PyPhotometryRecordingExtractor(folder_path=_session("two_colour_time_division"))
    with pytest.raises(ValueError, match="No pyPhotometry store named 'detector_9_excitation_9'"):
        extractor.read(events=["detector_9_excitation_9"], outputPath=_session("two_colour_time_division"))
