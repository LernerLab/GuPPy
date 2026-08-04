"""End-to-end test of NWB export over GuPPy's native stubbed data, one acquisition format at a time.

Uses the session-scoped ``step5_output_*`` fixtures (tests/integration/conftest.py), which copy a
stubbed session to a temp dir and run the real pipeline to produce a populated
``<session>_output_<run>`` dir. Everything the export does is format-independent except which
interfaces read the raw folder, so each format runs the same assertions against its own topology.
"""

from datetime import datetime
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO

from guppy.orchestration.export_nwb import export_session_to_nwb
from guppy.utils._hdf5_io import read_hdf5
from guppy.utils.acquisition_format import (
    acquisition_supplies_session_start_time,
    resolve_acquisition_format,
)
from guppy.utils.nwb_metadata import (
    build_metadata_dict,
    derive_channels,
    dump_yaml,
    load_yaml,
    parse_metadata_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_METADATA = PROJECT_ROOT / "data" / "fiber_photometry_metadata_example.yaml"

# Supplied by the form for the formats whose raw files record no recording start.
SUPPLIED_SESSION_START_TIME = datetime.fromisoformat("2020-05-04T09:00:00+00:00")

# The topology each stubbed session was labeled with in REPRESENTATIVE_SESSIONS
# (tests/integration/integration_helpers.py): one recording site, one behavioral event store.
EXPECTED_TOPOLOGY = {
    "tdt": {"recording_sites": ["dms"], "event_types": {"port_entries_dms"}},
    "csv": {"recording_sites": ["region"], "event_types": {"ttl"}},
    "npm": {"recording_sites": ["region1"], "event_types": {"ttl_region1"}},
    "doric": {"recording_sites": ["region"], "event_types": {"ttl"}},
}

# Per-role annotations for the derived channels, using the example library's device names.
COMMON_CHANNEL_ANNOTATIONS = {
    "emission_wavelength_in_nm": 525.0,
    "optical_fiber": "optical_fiber",
    "photodetector": "photodetector",
    "indicator": "dms_green_fluorophore",
    "dichroic_mirror": "dichroic_mirror",
    "emission_filter": "emission_filter",
}
ROLE_CHANNEL_ANNOTATIONS = {
    "control": {  # isosbestic
        "excitation_wavelength_in_nm": 405.0,
        "excitation_source": "excitation_source_isosbestic_control",
        "excitation_filter": "isosbestic_excitation_filter",
    },
    "signal": {  # calcium
        "excitation_wavelength_in_nm": 465.0,
        "excitation_source": "excitation_source_calcium_signal",
        "excitation_filter": "excitation_filter",
    },
}


class TestExportSessionToNwb:
    @pytest.fixture(params=sorted(EXPECTED_TOPOLOGY))
    def acquisition_format(self, request) -> str:
        return request.param

    @pytest.fixture
    def pipeline_state(self, request, acquisition_format) -> dict:
        return request.getfixturevalue(f"step5_output_{acquisition_format}")

    @pytest.fixture
    def metadata_yaml_path(self, pipeline_state, acquisition_format, tmp_path) -> str:
        """Build a session metadata YAML the way the form does: device library + per-channel annotations."""
        channels = derive_channels(pipeline_state["output_directory"])
        # Reuse the example's hardware/biology library (recombined into merged device entries).
        devices, _channel_rows, _scalars = parse_metadata_dict(load_yaml(EXAMPLE_METADATA), channels)
        channel_rows = [
            {**COMMON_CHANNEL_ANNOTATIONS, **ROLE_CHANNEL_ANNOTATIONS[channel.role]} for channel in channels
        ]
        scalars = {
            "session_description": "RI30 photometry session",
            "identifier": f"{acquisition_format}_run1",
            "lab": "Lerner Lab",
            "institution": "Northwestern University",
            "subject_id": "63_207",
            "sex": "M",
            "species": "Mus musculus",
        }
        # The form only asks for a start time where the raw files carry none.
        if not acquisition_supplies_session_start_time(
            session_folder_path=pipeline_state["session_copy"], acquisition_format=acquisition_format
        ):
            scalars["session_start_time"] = SUPPLIED_SESSION_START_TIME.isoformat()

        metadata = build_metadata_dict(devices, channel_rows, scalars, channels)
        path = tmp_path / "nwb_metadata.yaml"
        dump_yaml(metadata, path)
        return str(path)

    def test_exports_stubbed_session(self, pipeline_state, acquisition_format, metadata_yaml_path, tmp_path):
        session_folder_path = str(pipeline_state["session_copy"])
        guppy_folder_path = str(pipeline_state["output_directory"])
        nwbfile_path = tmp_path / "exported.nwb"
        expected = EXPECTED_TOPOLOGY[acquisition_format]

        # The format the export picks must be the one the pipeline processed the session as.
        assert resolve_acquisition_format(session_folder_path) == acquisition_format

        written_path = export_session_to_nwb(
            session_folder_path=session_folder_path,
            acquisition_format=acquisition_format,
            guppy_folder_path=guppy_folder_path,
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=str(nwbfile_path),
        )

        assert nwbfile_path.exists()
        assert written_path == str(nwbfile_path)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            # One acquisition series per role, each stacking that role's store from every site.
            assert set(nwbfile.acquisition) == {
                "FiberPhotometryResponseSeriesSignal",
                "FiberPhotometryResponseSeriesControl",
            }
            assert "guppy" in nwbfile.processing, "Expected a 'guppy' processing module"
            assert nwbfile.subject.subject_id == "63_207"
            assert nwbfile.session_description == "RI30 photometry session"

            # Each GuPPy recording site must resolve to the two fiber_photometry_table rows written
            # for it -- the control row (index 0) and the signal row (index 1), in the order
            # build_metadata_dict emitted them.
            recording_sites = nwbfile.processing["guppy"]["recording_sites"].to_dataframe()
            assert list(recording_sites["recording_site"]) == expected["recording_sites"]
            fiber_photometry_rows = recording_sites["fiber_photometry_table_region"][0]
            assert list(fiber_photometry_rows.index) == [0, 1]
            assert list(fiber_photometry_rows["location"]) == expected["recording_sites"] * 2
            assert list(fiber_photometry_rows["excitation_wavelength_in_nm"]) == [405.0, 465.0]

            behavioral_events = nwbfile.events["BehavioralEvents"].to_dataframe()
            assert set(behavioral_events["event_type"]) == expected["event_types"]

    def test_acquisition_timestamps_are_on_the_clock_guppy_analyzed(
        self, pipeline_state, acquisition_format, metadata_yaml_path, tmp_path
    ):
        """The raw series must land on GuPPy's own clock, not a unit-scaled copy of it.

        GuPPy records that clock's origin as ``recordingStart`` (0 for TDT, the first raw timestamp
        otherwise). A series read in the wrong timestamp unit lands orders of magnitude away from it
        while every topology assertion still passes -- the failure that held Neurophotometrics back
        (issue #411), and the one this guards against for all four formats.
        """
        nwbfile_path = tmp_path / "exported.nwb"
        export_session_to_nwb(
            session_folder_path=str(pipeline_state["session_copy"]),
            acquisition_format=acquisition_format,
            guppy_folder_path=str(pipeline_state["output_directory"]),
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=str(nwbfile_path),
        )

        recording_site = EXPECTED_TOPOLOGY[acquisition_format]["recording_sites"][0]
        guppy_origin = read_hdf5(
            f"timeCorrection_{recording_site}", str(pipeline_state["output_directory"]), "recordingStart"
        )[0]

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            series = io.read().acquisition["FiberPhotometryResponseSeriesSignal"]
            # A regularly sampled series carries starting_time + rate rather than a timestamps array.
            first_timestamp = float(series.starting_time if series.timestamps is None else series.timestamps[0])

        # Loose on purpose: this catches a wrong unit (a factor of 1000), not sample-level drift
        # between the reference channel GuPPy timed from and the one stacked into this series.
        assert first_timestamp == pytest.approx(guppy_origin, abs=1.0)

    def test_session_start_time_comes_from_the_acquisition_or_the_form(
        self, pipeline_state, acquisition_format, metadata_yaml_path, tmp_path
    ):
        nwbfile_path = tmp_path / "exported.nwb"
        export_session_to_nwb(
            session_folder_path=str(pipeline_state["session_copy"]),
            acquisition_format=acquisition_format,
            guppy_folder_path=str(pipeline_state["output_directory"]),
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=str(nwbfile_path),
        )

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            session_start_time = io.read().session_start_time

        if acquisition_format == "tdt":
            # Read from the tank: Photo_63_207-181030-103332 was recorded on 2018-10-30.
            assert (session_start_time.year, session_start_time.month, session_start_time.day) == (2018, 10, 30)
        else:
            assert session_start_time == SUPPLIED_SESSION_START_TIME

    def test_missing_session_start_time_names_the_step_that_supplies_it(
        self, pipeline_state, acquisition_format, tmp_path
    ):
        # Formats that read their own start time have nothing for the form to omit.
        assert acquisition_supplies_session_start_time(
            session_folder_path=str(pipeline_state["session_copy"]), acquisition_format=acquisition_format
        ) == (acquisition_format == "tdt")
        if acquisition_format == "tdt":
            return

        with pytest.raises(ValueError) as excinfo:
            export_session_to_nwb(
                session_folder_path=str(pipeline_state["session_copy"]),
                acquisition_format=acquisition_format,
                guppy_folder_path=str(pipeline_state["output_directory"]),
                metadata_yaml_path=None,
                nwbfile_path=str(tmp_path / "exported.nwb"),
            )
        assert "No session start time is available" in str(excinfo.value)
        assert "Step 6 (Input Metadata)" in str(excinfo.value)


class TestUnsupportedSource:
    """Formats GuPPy processes that the export refuses, each named before the converter is built."""

    def test_nwb_sourced_session_is_refused(self, step5_output_nwb):
        # GuPPy can process a session read from an NWB file, but the converter has no interface for
        # re-exporting one, so the export must say so rather than fail deep inside neuroconv.
        with pytest.raises(ValueError) as excinfo:
            resolve_acquisition_format(str(step5_output_nwb["session_copy"]))
        assert "does not support the 'nwb' acquisition format" in str(excinfo.value)
