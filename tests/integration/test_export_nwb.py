"""End-to-end test of NWB export over GuPPy's native stubbed TDT data.

Uses the session-scoped ``step5_output_tdt`` fixture (tests/integration/conftest.py),
which copies ``stubbed_testing_data/tdt/Photo_63_207-181030-103332`` to a temp dir
and runs the real pipeline to produce a populated ``<session>_output_<run>`` dir.
"""

import os
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO

from guppy.orchestration.export_nwb import (
    MERGED_METADATA_FILENAME,
    export_session_to_nwb,
)
from guppy.utils.nwb_metadata import dump_yaml, load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_METADATA = PROJECT_ROOT / "data" / "fiber_photometry_metadata_example.yaml"


class TestExportSessionToNwb:
    @pytest.fixture
    def metadata_yaml_path(self, tmp_path) -> str:
        """One self-contained metadata overlay: hardware/biology example + session + subject."""
        metadata = load_yaml(EXAMPLE_METADATA)
        metadata["NWBFile"] = {
            "session_description": "RI30 photometry session",
            "identifier": "Photo_63_207_run1",
            "lab": "Lerner Lab",
            "institution": "Northwestern University",
        }
        metadata["Subject"] = {"subject_id": "63_207", "sex": "M", "species": "Mus musculus"}
        path = tmp_path / "nwb_metadata.yaml"
        dump_yaml(metadata, path)
        return str(path)

    def test_exports_stubbed_tdt_session(self, step5_output_tdt, metadata_yaml_path, tmp_path):
        tdt_folder_path = str(step5_output_tdt["session_copy"])
        guppy_folder_path = str(step5_output_tdt["output_directory"])
        nwbfile_path = tmp_path / "exported.nwb"

        merged_metadata_path = export_session_to_nwb(
            tdt_folder_path=tdt_folder_path,
            guppy_folder_path=guppy_folder_path,
            metadata_yaml_path=metadata_yaml_path,
            nwbfile_path=str(nwbfile_path),
            stub_test=True,
        )

        assert nwbfile_path.exists()
        assert os.path.basename(merged_metadata_path) == MERGED_METADATA_FILENAME
        assert os.path.exists(merged_metadata_path)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            response_series = [
                name
                for name, obj in nwbfile.acquisition.items()
                if obj.neurodata_type == "FiberPhotometryResponseSeries"
            ]
            assert response_series, "Expected at least one FiberPhotometryResponseSeries in acquisition"
            assert "guppy" in nwbfile.processing, "Expected a 'guppy' processing module"
            assert nwbfile.subject.subject_id == "63_207"
            assert nwbfile.session_description == "RI30 photometry session"
