"""End-to-end test of NWB export over GuPPy's native stubbed TDT data.

Uses the session-scoped ``step5_output_tdt`` fixture (tests/integration/conftest.py),
which copies ``stubbed_testing_data/tdt/Photo_63_207-181030-103332`` to a temp dir
and runs the real pipeline to produce a populated ``<session>_output_<run>`` dir.
"""

from pathlib import Path

import pytest
from pynwb import NWBHDF5IO

from guppy.orchestration.export_nwb import export_session_to_nwb
from guppy.utils.nwb_metadata import (
    build_metadata_dict,
    derive_channels,
    dump_yaml,
    load_yaml,
    parse_metadata_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_METADATA = PROJECT_ROOT / "data" / "fiber_photometry_metadata_example.yaml"


class TestExportSessionToNwb:
    @pytest.fixture
    def metadata_yaml_path(self, step5_output_tdt, tmp_path) -> str:
        """Build a session metadata YAML the way the form does: device library + per-channel annotations."""
        channels = derive_channels(step5_output_tdt["output_directory"])  # 2 dms channels (control, signal)
        # Reuse the example's hardware/biology library (recombined into merged device entries).
        devices, _channel_rows, _scalars = parse_metadata_dict(load_yaml(EXAMPLE_METADATA), channels)
        common = {
            "emission_wavelength_in_nm": 525.0,
            "optical_fiber": "optical_fiber",
            "photodetector": "photodetector",
            "indicator": "dms_green_fluorophore",
            "dichroic_mirror": "dichroic_mirror",
            "emission_filter": "emission_filter",
        }
        channel_rows = [
            {  # control / isosbestic
                "excitation_wavelength_in_nm": 405.0,
                "excitation_source": "excitation_source_isosbestic_control",
                "excitation_filter": "isosbestic_excitation_filter",
                **common,
            },
            {  # signal / calcium
                "excitation_wavelength_in_nm": 465.0,
                "excitation_source": "excitation_source_calcium_signal",
                "excitation_filter": "excitation_filter",
                **common,
            },
        ]
        scalars = {
            "session_description": "RI30 photometry session",
            "identifier": "Photo_63_207_run1",
            "lab": "Lerner Lab",
            "institution": "Northwestern University",
            "subject_id": "63_207",
            "sex": "M",
            "species": "Mus musculus",
        }
        metadata = build_metadata_dict(devices, channel_rows, scalars, channels)
        path = tmp_path / "nwb_metadata.yaml"
        dump_yaml(metadata, path)
        return str(path)

    def test_exports_stubbed_tdt_session(self, step5_output_tdt, metadata_yaml_path, tmp_path):
        session_folder_path = str(step5_output_tdt["session_copy"])
        guppy_folder_path = str(step5_output_tdt["output_directory"])
        nwbfile_path = tmp_path / "exported.nwb"

        written_path = export_session_to_nwb(
            session_folder_path=session_folder_path,
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

            # The GuPPy 'dms' recording site must resolve to the two fiber_photometry_table rows
            # written for it -- the control row (index 0) and the signal row (index 1), in the order
            # build_metadata_dict emitted them.
            recording_sites = nwbfile.processing["guppy"]["recording_sites"].to_dataframe()
            assert list(recording_sites["recording_site"]) == ["dms"]
            fiber_photometry_rows = recording_sites["fiber_photometry_table_region"][0]
            assert list(fiber_photometry_rows.index) == [0, 1]
            assert list(fiber_photometry_rows["location"]) == ["dms", "dms"]
            assert list(fiber_photometry_rows["excitation_wavelength_in_nm"]) == [405.0, 465.0]

            behavioral_events = nwbfile.events["BehavioralEvents"].to_dataframe()
            assert set(behavioral_events["event_type"]) == {"port_entries_dms"}
