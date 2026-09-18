"""Tests for reporting what one DANDI NWB asset holds.

These read HDF5, so they run against the real mock NWB files in ``stubbed_testing_data/nwb/`` --
one per supported ndx-fiber-photometry / events combination -- with no network access. The same
contract runs against a streamed archive asset in ``test_dandi_preview_live.py``.
"""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from guppy.utils.dandi_preview import (
    _read_fiber_photometry_table,
    probe_photometry,
    read_example_traces,
)
from guppy_test_data import STUBBED_TESTING_DATA

from .dandi_preview_test_mixin import DandiPreviewTestMixin

NWB_DATA = STUBBED_TESTING_DATA / "nwb"
MOCK_NWB_FILES = {
    name: NWB_DATA / name / f"{name}.nwb"
    for name in (
        "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events",
    )
}


@pytest.fixture(params=sorted(MOCK_NWB_FILES))
def mock_nwb_file(request):
    """Open each mock NWB file in turn, so the probe is checked on every supported layout."""
    with h5py.File(MOCK_NWB_FILES[request.param], "r") as file:
        yield file


@pytest.fixture
def sparse_nwb_file(tmp_path):
    """An NWB-shaped file whose series carries timestamps and whose fiber table is bare.

    Covers the layouts the mock files do not: a single-channel series with irregular
    timestamps instead of a starting_time/rate pair, a fiber photometry table with only a
    ``location`` column, and no ``/processing`` group at all.
    """
    path = tmp_path / "sparse.nwb"
    with h5py.File(path, "w") as file:
        series = file.create_group("acquisition/series_with_timestamps")
        series.attrs["neurodata_type"] = "FiberPhotometryResponseSeries"
        series.create_dataset("data", data=np.arange(10, dtype=float))
        series.create_dataset("timestamps", data=np.arange(10) * 0.5)
        series.create_dataset("fiber_photometry_table_region", data=np.array([0]))
        container = file.create_group("general/fiber_photometry")
        container.attrs["neurodata_type"] = "FiberPhotometry"
        table = container.create_group("fiber_photometry_table")
        table.attrs["neurodata_type"] = "FiberPhotometryTable"
        table.create_dataset("location", data=np.array([b"NAc"]))
    with h5py.File(path, "r") as file:
        yield file


class TestFindFiberPhotometryTable:
    def test_the_table_is_found_under_whatever_names_its_author_gave_it(self, tmp_path):
        # Dandiset 001038 names both levels after their types rather than in snake case.
        path = tmp_path / "camel_case.nwb"
        with h5py.File(path, "w") as file:
            container = file.create_group("general/FiberPhotometry")
            container.attrs["neurodata_type"] = "FiberPhotometry"
            table = container.create_group("FiberPhotometryTable")
            table.attrs["neurodata_type"] = "FiberPhotometryTable"
            table.create_dataset("location", data=np.array([b"NAc"]))
        with h5py.File(path, "r") as file:
            rows = _read_fiber_photometry_table(file)
        assert rows == [
            {
                "location": "NAc",
                "indicator": None,
                "excitation_wavelength_in_nm": None,
                "emission_wavelength_in_nm": None,
            }
        ]

    def test_a_file_without_the_container_has_no_table(self, tmp_path):
        path = tmp_path / "behavior.nwb"
        with h5py.File(path, "w") as file:
            file.create_group("general/devices")
        with h5py.File(path, "r") as file:
            assert _read_fiber_photometry_table(file) == []


class TestProbePhotometry:
    def test_finds_the_response_series_and_its_timing(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert probe.has_photometry is True
        (series,) = probe.series
        assert series.name == "fiber_photometry_response_series"
        assert series.sample_count == 3000
        assert series.channel_count == 2
        assert series.sampling_rate_in_hz == 30.0
        assert series.duration_in_seconds == 100.0

    def test_channels_carry_the_store_names_guppy_will_show(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert [channel.store_name for channel in probe.channels] == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert [channel.column_index for channel in probe.channels] == [0, 1]

    def test_channels_carry_the_fiber_photometry_table_details(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert probe.locations == ("VTA",)
        assert probe.indicators == ("GCamp6f",)
        assert [channel.excitation_wavelength_in_nm for channel in probe.channels] == [
            405.0,
            470.0,
        ]

    def test_session_fields_are_read(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert "Mock session for NWB extractor testing" in probe.session_description
        assert probe.identifier == Path(mock_nwb_file.filename).stem
        # The mock files are regenerated, so only the shape of the timestamp is fixed.
        assert datetime.fromisoformat(probe.session_start_time).year >= 2024
        # These mock files carry no Subject group, which the probe reports as no fields.
        assert probe.subject == {}

    def test_emission_wavelength_is_read_when_the_table_has_it(self):
        # The column exists only in ndx-fiber-photometry v0.2.
        with h5py.File(MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"], "r") as file:
            probe = probe_photometry(file=file)
        assert [channel.emission_wavelength_in_nm for channel in probe.channels] == [
            525.0,
            525.0,
        ]
        with h5py.File(
            MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"],
            "r",
        ) as file:
            probe = probe_photometry(file=file)
        assert [channel.emission_wavelength_in_nm for channel in probe.channels] == [
            None,
            None,
        ]

    def test_timestamps_are_used_when_the_series_has_no_rate(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (series,) = probe.series
        # Timestamps run 0, 0.5, ... 4.5, so the span is 4.5 s over 9 intervals: 2 Hz.
        assert series.sampling_rate_in_hz == 2.0
        assert series.duration_in_seconds == 4.5

    def test_a_single_channel_series_keeps_its_own_name_as_the_store(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (channel,) = probe.channels
        assert channel.store_name == "series_with_timestamps"
        assert channel.column_index is None

    def test_table_columns_the_file_omits_are_reported_as_unknown(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (channel,) = probe.channels
        assert channel.location == "NAc"
        assert channel.indicator is None
        assert channel.excitation_wavelength_in_nm is None
        assert channel.emission_wavelength_in_nm is None
        # Without a wavelength there is no basis for calling the channel signal or control.
        assert channel.suggested_label is None

    def test_a_file_without_photometry_reports_none(self, tmp_path):
        path = tmp_path / "empty.nwb"
        with h5py.File(path, "w") as file:
            file.create_group("acquisition")
            file.create_dataset("session_description", data="no photometry here")
        with h5py.File(path, "r") as file:
            probe = probe_photometry(file=file)
        assert probe.has_photometry is False
        assert probe.channels == ()
        assert probe.series == ()
        assert probe.session_description == "no photometry here"


class TestReadExampleTraces:
    def test_reads_the_requested_duration_of_every_channel(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=10.0, max_points=1000)
        # 10 s at 30 Hz is 300 samples, which is under max_points, so nothing is strided away.
        assert list(traces.traces) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert traces.timestamps.shape == (300,)
        np.testing.assert_allclose(traces.timestamps[:3], [0.0, 1 / 30, 2 / 30])
        for values in traces.traces.values():
            assert values.shape == (300,)

    def test_decimates_down_to_max_points(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=100.0, max_points=100)
        # 100 s at 30 Hz is the full 3000 samples, strided by ceil(3000/100) = 30.
        assert traces.timestamps.shape == (100,)
        np.testing.assert_allclose(traces.timestamps[:3], [0.0, 1.0, 2.0])

    def test_traces_hold_the_series_columns(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=1.0, max_points=1000)
        expected = mock_nwb_file["acquisition/fiber_photometry_response_series/data"][:30]
        np.testing.assert_allclose(traces.traces["fiber_photometry_response_series_0"], expected[:, 0])
        np.testing.assert_allclose(traces.traces["fiber_photometry_response_series_1"], expected[:, 1])

    def test_reads_a_timestamped_series_from_its_own_timestamps(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        traces = read_example_traces(file=sparse_nwb_file, probe=probe, max_points=4)
        # All 10 samples fall inside the default 60 s window, strided by ceil(10/4) = 3.
        np.testing.assert_allclose(traces.timestamps, [0.0, 1.5, 3.0, 4.5])
        np.testing.assert_allclose(traces.traces["series_with_timestamps"], [0.0, 3.0, 6.0, 9.0])

    def test_never_returns_more_than_max_points(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=100.0, max_points=7)
        assert traces.timestamps.shape == (7,)

    def test_named_series_is_honored(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(
            file=mock_nwb_file,
            probe=probe,
            series_name="fiber_photometry_response_series",
            duration_in_seconds=1.0,
        )
        assert traces.series_name == "fiber_photometry_response_series"


class MockNwbPreviewContract(DandiPreviewTestMixin):
    """The preview contract bound to one mock file; a subclass per layout names its own file."""

    mock_name: str
    expected_store_names = [
        "fiber_photometry_response_series_0",
        "fiber_photometry_response_series_1",
    ]
    expected_suggested_labels = ["control_VTA", "signal_VTA"]
    expected_locations = ("VTA",)
    expected_indicators = ("GCamp6f",)

    @pytest.fixture
    def mock_file(self):
        with h5py.File(MOCK_NWB_FILES[self.mock_name], "r") as file:
            yield file

    @pytest.fixture
    def probe(self, mock_file):
        return probe_photometry(file=mock_file)

    @pytest.fixture
    def traces(self, mock_file, probe):
        return read_example_traces(
            file=mock_file,
            probe=probe,
            duration_in_seconds=self.trace_duration_in_seconds,
            max_points=1000,
        )


class TestDandiPreviewFiberPhotometryV01NdxEvents(MockNwbPreviewContract):
    mock_name = "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"
    expected_event_name = "labeled_events"


class TestDandiPreviewFiberPhotometryV02NdxEvents(MockNwbPreviewContract):
    mock_name = "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
    expected_event_name = "labeled_events"


class TestDandiPreviewFiberPhotometryV02CoreEvents(MockNwbPreviewContract):
    mock_name = "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"
    expected_event_name = "simple_events"
