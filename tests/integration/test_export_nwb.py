"""End-to-end test of NWB export over GuPPy's native stubbed data, one acquisition format at a time.

Uses the session-scoped ``step5_output_*`` fixtures (tests/integration/conftest.py), which copy a
stubbed session to a temp dir and run the real pipeline to produce a populated
``<session>_output_<run>`` dir. Everything the export does is format-independent except which
interfaces read the raw folder, so each format runs the same assertions against its own topology.
"""

import shutil
from pathlib import Path

import h5py
import pytest
from pynwb import NWBHDF5IO

from guppy.extractors.nwb_recording_extractor import _find_nwb_file
from guppy.orchestration.export_nwb import export_session_to_nwb
from guppy.orchestration.metadata import METADATA_FILENAME
from guppy.testing.api import step7
from guppy.utils._hdf5_io import read_hdf5
from guppy.utils.acquisition_format import (
    acquisition_supplies_session_start_time,
    resolve_acquisition_format,
    resolve_session_source,
)
from guppy.utils.nwb_io import open_nwbfile_io
from guppy.utils.utils import parse_run_name
from guppy_test_data import STUBBED_TESTING_DATA

from .integration_helpers import SUPPLIED_SESSION_START_TIME, write_metadata_yaml

MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE = (
    STUBBED_TESTING_DATA
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"
    / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.nwb"
)

# The topology each stubbed session was labeled with in REPRESENTATIVE_SESSIONS
# (tests/integration/integration_helpers.py): one recording site, one behavioral event store.
# ``raw_event_tables`` are the per-event tables the acquisition's own events interface writes,
# named after the event in CamelCase; ``event_types`` are the rows GuPPy analyzed, which the
# converter collects into the single ``GuppyEvents`` table.
EXPECTED_TOPOLOGY = {
    "tdt": {
        "recording_sites": ["dms"],
        "event_types": {"port_entries_dms"},
        "raw_event_tables": {"PortEntriesDms"},
    },
    "csv": {"recording_sites": ["region"], "event_types": {"ttl"}, "raw_event_tables": {"Ttl"}},
    "npm": {"recording_sites": ["region1"], "event_types": {"ttl_region1"}, "raw_event_tables": {"TtlRegion1"}},
    "doric": {"recording_sites": ["region"], "event_types": {"ttl"}, "raw_event_tables": {"Ttl"}},
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
        return write_metadata_yaml(
            session_folder_path=str(pipeline_state["session_copy"]),
            output_directory=str(pipeline_state["output_directory"]),
            acquisition_format=acquisition_format,
            path=tmp_path / "nwb_metadata.yaml",
        )

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

            # Both halves of the events story must reach the file: the raw stream the acquisition
            # recorded, in its own per-event table, and the onsets GuPPy analyzed, collected into
            # one chronological table keyed by event name.
            assert set(nwbfile.events) == expected["raw_event_tables"] | {"GuppyEvents"}
            analyzed_events = nwbfile.events["GuppyEvents"].to_dataframe()
            assert set(analyzed_events["event_type"]) == expected["event_types"]

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


class TestExportNwbSourcedSession:
    """A session GuPPy processed out of an NWB file is exported by adding its outputs to that file.

    No metadata form feeds this route: the source already carries the devices, the fiber photometry
    chain, and the session start time, and GuPPy's store ids were derived from its own response
    series, so the recording sites registry links into the table already there.
    """

    @pytest.fixture
    def source_nwbfile_path(self, step5_output_nwb) -> str:
        return _find_nwb_file(str(step5_output_nwb["session_copy"]))

    @pytest.fixture
    def exported_nwbfile_path(self, step5_output_nwb, source_nwbfile_path, tmp_path) -> Path:
        nwbfile_path = tmp_path / "exported.nwb"
        export_session_to_nwb(
            session_folder_path=str(step5_output_nwb["session_copy"]),
            acquisition_format="nwb",
            guppy_folder_path=str(step5_output_nwb["output_directory"]),
            metadata_yaml_path=None,
            nwbfile_path=str(nwbfile_path),
            nwb_source=source_nwbfile_path,
        )
        return nwbfile_path

    def test_session_source_resolves_to_the_file_it_came_from(self, step5_output_nwb, source_nwbfile_path):
        assert resolve_session_source(
            session_folder_path=str(step5_output_nwb["session_copy"]), inputParameters={}
        ) == ("nwb", source_nwbfile_path)

    def test_the_source_is_carried_across_un_duplicated(self, exported_nwbfile_path, source_nwbfile_path):
        with NWBHDF5IO(source_nwbfile_path, "r") as io:
            source = io.read()
            expected_acquisition = set(source.acquisition)
            expected_session_start_time = source.session_start_time

        with NWBHDF5IO(str(exported_nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert set(nwbfile.acquisition) == expected_acquisition
            assert nwbfile.session_start_time == expected_session_start_time

    def test_the_guppy_outputs_are_added(self, exported_nwbfile_path):
        with NWBHDF5IO(str(exported_nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert "guppy" in nwbfile.processing
            # GuPPy's own analyzed onsets, written by the interface rather than by a converter.
            assert "GuppyEvents" in nwbfile.events
            analyzed_events = nwbfile.events["GuppyEvents"].to_dataframe()
            assert set(analyzed_events["event_type"]) == {"ttl"}

    def test_each_recording_site_links_into_the_files_own_fiber_photometry_table(self, exported_nwbfile_path):
        # The stores GuPPy listed name the source's response series columns, so the registry can
        # reference the FiberPhotometryTable rows those columns were recorded on.
        with NWBHDF5IO(str(exported_nwbfile_path), "r") as io:
            recording_sites = io.read().processing["guppy"]["recording_sites"].to_dataframe()
            assert list(recording_sites["recording_site"]) == ["region"]
            assert list(recording_sites["fiber_photometry_table_region"][0].index) == [0, 1]


class TestExportLegacyNwbSource:
    """A source written with ndx-fiber-photometry 0.1.0 exports, and keeps exporting.

    That file predates NWB 2.9, so pynwb reads its ``Device.model`` strings back as ``DeviceModel``
    containers it never adds to the file, and writing a link to a container with no parent fails.
    Exporting twice in one process covers the second hazard: the first export imports ndx-guppy,
    registering ndx-fiber-photometry 0.2.x, after which reading a 0.1.0 file against the
    process-wide type map builds it under the wrong spec.

    Driven off the v0.2 mock's GuPPy outputs, whose store ids name the same response series columns
    the v0.1.0 mock holds.
    """

    @pytest.fixture(scope="class")
    def exported_nwbfile_paths(self, step5_output_nwb, tmp_path_factory) -> list[Path]:
        output_directory = tmp_path_factory.mktemp("legacy_nwb_source")
        nwbfile_paths = []
        for export_index in range(2):
            nwbfile_path = output_directory / f"exported_{export_index}.nwb"
            export_session_to_nwb(
                session_folder_path=str(step5_output_nwb["session_copy"]),
                acquisition_format="nwb",
                guppy_folder_path=str(step5_output_nwb["output_directory"]),
                metadata_yaml_path=None,
                nwbfile_path=str(nwbfile_path),
                nwb_source=str(MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE),
            )
            nwbfile_paths.append(nwbfile_path)
        return nwbfile_paths

    @pytest.fixture(params=[0, 1])
    def exported_nwbfile_path(self, exported_nwbfile_paths, request) -> Path:
        return exported_nwbfile_paths[request.param]

    def test_every_device_model_is_written(self, exported_nwbfile_path):
        with open_nwbfile_io(path=str(exported_nwbfile_path)) as io:
            nwbfile = io.read()
            models = {device.name: device.model for device in nwbfile.devices.values()}

            assert sorted(name for name, model in models.items() if model is not None) == [
                "dichroic_mirror",
                "excitation_source_control",
                "excitation_source_signal",
                "optical_fiber",
                "optical_filter",
            ]
            assert all(model.parent is nwbfile for model in models.values() if model is not None)

    def test_devices_sharing_a_model_string_keep_their_own_metadata(self, exported_nwbfile_path):
        # Both lasers are recorded as "laser model" but describe different excitation wavelengths,
        # so collapsing them onto one DeviceModel would put one laser's description on the other.
        with open_nwbfile_io(path=str(exported_nwbfile_path)) as io:
            nwbfile = io.read()

            assert nwbfile.devices["excitation_source_control"].model.description == (
                "405 nm isosbestic excitation source."
            )
            assert nwbfile.devices["excitation_source_signal"].model.description == ("470 nm signal excitation source.")

    def test_the_guppy_outputs_are_added(self, exported_nwbfile_path):
        with open_nwbfile_io(path=str(exported_nwbfile_path)) as io:
            nwbfile = io.read()
            assert "guppy" in nwbfile.processing

    def test_the_output_keeps_the_sources_extension_versions(self, exported_nwbfile_path):
        # The point of the whole write path: a file claiming 0.2.x over a table with 0.1.0's columns
        # is one nothing can read back, so the output stays on the version its contents match.
        with h5py.File(str(exported_nwbfile_path), "r") as file:
            cached_versions = {name: list(group) for name, group in file["/specifications"].items()}

        assert cached_versions["ndx-fiber-photometry"] == ["0.1.0"]
        assert "ndx-guppy" in cached_versions


class TestExportMixedFormatSession:
    """GuPPy's custom-event import writes single-column ``timestamps`` CSVs into the session folder,
    which the converter reads alongside whatever the rig recorded."""

    @pytest.fixture
    def mixed_session(self, step5_output_tdt, tmp_path) -> Path:
        """A copy of the TDT session with an imported-event CSV dropped in beside the tank."""
        session_copy = tmp_path / "mixed_session"
        shutil.copytree(step5_output_tdt["session_copy"], session_copy)
        (session_copy / "imported_licks.csv").write_text("timestamps\n1.5\n2.5\n3.5\n")
        return session_copy

    @pytest.fixture
    def output_directory(self, mixed_session, step5_output_tdt) -> Path:
        return mixed_session / Path(step5_output_tdt["output_directory"]).name

    def test_the_traces_still_resolve_to_the_acquisition_format(self, mixed_session):
        assert resolve_acquisition_format(str(mixed_session)) == "tdt"

    def test_the_session_exports(self, mixed_session, output_directory, tmp_path):
        nwbfile_path = tmp_path / "exported.nwb"
        metadata_yaml_path = write_metadata_yaml(
            session_folder_path=str(mixed_session),
            output_directory=str(output_directory),
            acquisition_format="tdt",
            path=tmp_path / "nwb_metadata.yaml",
        )

        export_session_to_nwb(
            session_folder_path=str(mixed_session),
            acquisition_format="tdt",
            guppy_folder_path=str(output_directory),
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=str(nwbfile_path),
        )

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert set(nwbfile.events) == EXPECTED_TOPOLOGY["tdt"]["raw_event_tables"] | {"GuppyEvents"}


class TestStep7EndToEnd:
    """The orchestration layer driven the way the GUI drives it: batching, the upfront
    prerequisite checks, and the per-session metadata overlay picked up from disk."""

    @pytest.fixture
    def session_with_metadata(self, step5_output_tdt) -> Path:
        """A pipeline-processed session whose output directory holds the Step 6 metadata YAML."""
        output_directory = Path(step5_output_tdt["output_directory"])
        write_metadata_yaml(
            session_folder_path=str(step5_output_tdt["session_copy"]),
            output_directory=str(output_directory),
            acquisition_format="tdt",
            path=output_directory / METADATA_FILENAME,
        )
        return Path(step5_output_tdt["session_copy"])

    def test_step7_writes_one_nwb_file_per_selected_run(self, session_with_metadata, step5_output_tdt):
        output_directory = Path(step5_output_tdt["output_directory"])

        step7(
            base_dir=str(session_with_metadata.parent),
            selected_folders=[str(session_with_metadata)],
            selected_runs={str(session_with_metadata): [parse_run_name(str(output_directory))]},
        )

        # Named after the full output directory so exports from several runs stay distinct.
        nwbfile_path = output_directory / f"{output_directory.name}.nwb"
        assert nwbfile_path.exists()

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            # The overlay the orchestrator found on disk reached the file.
            assert nwbfile.subject.subject_id == "63_207"
            assert nwbfile.session_description == "RI30 photometry session"
            assert "guppy" in nwbfile.processing

    def test_step7_refuses_a_combined_run_before_writing_anything(self, session_with_metadata, step5_output_tdt):
        output_directory = Path(step5_output_tdt["output_directory"])
        # step5_output_tdt is session-scoped, so a sibling test may already have exported into it.
        nwbfile_path = output_directory / f"{output_directory.name}.nwb"
        nwbfile_path.unlink(missing_ok=True)

        with pytest.raises(ValueError, match="does not support combine_data=True"):
            step7(
                base_dir=str(session_with_metadata.parent),
                selected_folders=[str(session_with_metadata)],
                combine_data=True,
                selected_runs={str(session_with_metadata): [parse_run_name(str(output_directory))]},
            )

        assert not nwbfile_path.exists()
