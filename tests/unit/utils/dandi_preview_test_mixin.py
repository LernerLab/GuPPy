"""Mixin of contract tests for :mod:`guppy.utils.dandi_preview`.

Bound twice: once in ``test_dandi_preview.py`` against the mock NWB files in
``stubbed_testing_data/nwb/``, and once in ``test_dandi_preview_live.py`` against a real asset
streamed from the archive. The mock files are genuine ndx-fiber-photometry output rather than
hand-faked groups, so what the live binding adds is file diversity: that a conversion nobody here
wrote still lays its channels, table and traces out where the probe looks for them.

Child test classes must define the following class-level attributes:

expected_store_names : list of str
    Store names the probe reports, in column order.
expected_suggested_labels : list of str
    The signal/control label suggested for each of those channels.
expected_locations : tuple of str
    Recording sites the file's fiber photometry table names.
expected_indicators : tuple of str
    Indicators that table names.
expected_event_name : str
    One event container the file holds that a PSTH could be aligned to.
trace_duration_in_seconds : float
    Seconds of trace the contract reads.

and the following fixtures:

probe : PhotometryProbe
    What the file under test holds.
traces : ExampleTraces
    ``trace_duration_in_seconds`` of every channel of that file.
"""

import numpy as np
import pytest


class DandiPreviewTestMixin:
    """Contract tests for reporting what one NWB asset holds."""

    expected_store_names: list[str]
    expected_suggested_labels: list[str]
    expected_locations: tuple[str, ...]
    expected_indicators: tuple[str, ...]
    expected_event_name: str
    trace_duration_in_seconds: float = 10.0

    def test_the_file_is_reported_as_holding_photometry(self, probe):
        assert probe.has_photometry is True
        assert probe.series

    def test_channels_carry_the_store_names_guppy_will_show(self, probe):
        assert [channel.store_name for channel in probe.channels] == self.expected_store_names

    def test_channels_carry_the_recording_sites_and_indicators(self, probe):
        assert probe.locations == self.expected_locations
        assert probe.indicators == self.expected_indicators

    def test_isosbestic_excitation_suggests_a_control_label(self, probe):
        assert [channel.suggested_label for channel in probe.channels] == self.expected_suggested_labels

    def test_the_series_timing_is_reported(self, probe):
        (series,) = probe.series
        assert series.channel_count == len(self.expected_store_names)
        assert series.sampling_rate_in_hz > 0
        assert series.duration_in_seconds > 0

    def test_the_events_a_psth_could_align_to_are_found(self, probe):
        assert self.expected_event_name in probe.event_names

    def test_example_traces_cover_every_channel(self, traces):
        assert list(traces.traces) == self.expected_store_names

    def test_example_traces_cover_the_requested_window(self, traces):
        assert traces.timestamps[0] == pytest.approx(0.0)
        assert traces.timestamps[-1] == pytest.approx(self.trace_duration_in_seconds, rel=0.05)

    def test_every_trace_is_finite_and_matches_the_timestamps(self, traces):
        for values in traces.traces.values():
            assert values.shape == traces.timestamps.shape
            assert np.isfinite(values).all()
