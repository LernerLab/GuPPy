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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_TEMPLATE = PROJECT_ROOT / "src" / "guppy" / "resources" / "fiber_photometry_metadata_template.yaml"


class TestExportSessionToNwb:
    @pytest.fixture
    def project_yaml_path(self, tmp_path) -> str:
        """The shared hardware/biology template plus project-level NWBFile/Subject fields."""
        project_metadata = load_yaml(BUNDLED_TEMPLATE)
        project_metadata["NWBFile"] = {"lab": "Lerner Lab", "institution": "Northwestern University"}
        project_metadata["Subject"] = {"species": "Mus musculus"}
        path = tmp_path / "nwb_project_metadata.yaml"
        dump_yaml(project_metadata, path)
        return str(path)

    @pytest.fixture
    def session_yaml_path(self, tmp_path) -> str:
        """The per-session description/subject fields."""
        session_metadata = {
            "NWBFile": {"session_description": "RI30 photometry session", "identifier": "Photo_63_207_run1"},
            "Subject": {"subject_id": "63_207", "sex": "M"},
        }
        path = tmp_path / "nwb_session_metadata.yaml"
        dump_yaml(session_metadata, path)
        return str(path)

    def test_exports_stubbed_tdt_session(self, step5_output_tdt, project_yaml_path, session_yaml_path, tmp_path):
        tdt_folder_path = str(step5_output_tdt["session_copy"])
        guppy_folder_path = str(step5_output_tdt["output_directory"])
        nwbfile_path = tmp_path / "exported.nwb"

        merged_metadata_path = export_session_to_nwb(
            tdt_folder_path=tdt_folder_path,
            guppy_folder_path=guppy_folder_path,
            project_yaml_path=project_yaml_path,
            session_yaml_path=session_yaml_path,
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
