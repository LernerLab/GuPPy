"""Unit tests for the upfront NWB-export prerequisite check.

`orchestrate_export_nwb_page` reads each selected session's ``GuPPyParamtersUsed.json`` and
aborts the whole batch before writing anything if any session was processed with the
``concatenate`` artifact-removal method, which re-times kept samples and breaks alignment to
the acquisition clock.
"""

import json
import os

import pytest

from guppy.orchestration.export_nwb import (
    _validate_artifact_removal_methods,
    orchestrate_export_nwb_page,
)


class TestValidateArtifactRemovalMethods:
    @pytest.fixture
    def session_path(self, tmp_path):
        """A session folder containing one ``<basename>_output_run1`` output directory."""
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        return session

    def _write_parameters(self, session_path, parameters):
        output_dir = session_path / f"{os.path.basename(session_path)}_output_run1"
        with open(output_dir / "GuPPyParamtersUsed.json", "w") as parameters_file:
            json.dump(parameters, parameters_file)

    def test_concatenate_with_remove_artifacts_aborts(self, session_path):
        self._write_parameters(session_path, {"removeArtifacts": True, "artifactsRemovalMethod": "concatenate"})
        with pytest.raises(ValueError) as excinfo:
            _validate_artifact_removal_methods([(str(session_path), "run1")])
        message = str(excinfo.value)
        assert "Photo_session (run1)" in message
        assert "concatenate" in message
        assert "replace with NaN" in message
        assert "https://github.com/LernerLab/GuPPy/issues/new" in message

    def test_replace_with_nan_does_not_abort(self, session_path):
        self._write_parameters(session_path, {"removeArtifacts": True, "artifactsRemovalMethod": "replace with NaN"})
        _validate_artifact_removal_methods([(str(session_path), "run1")])

    def test_concatenate_without_remove_artifacts_does_not_abort(self, session_path):
        # removeArtifacts is False, so the method is irrelevant and must not trigger the abort.
        self._write_parameters(session_path, {"removeArtifacts": False, "artifactsRemovalMethod": "concatenate"})
        _validate_artifact_removal_methods([(str(session_path), "run1")])

    def test_orchestrate_aborts_before_any_export(self, session_path):
        # End-to-end through the public entry point: the offending config must raise the
        # ValueError before the export loop touches neuroconv.
        self._write_parameters(session_path, {"removeArtifacts": True, "artifactsRemovalMethod": "concatenate"})
        input_parameters = {"selectedOutputs": {str(session_path): ["run1"]}}
        with pytest.raises(ValueError, match="does not support the 'concatenate'"):
            orchestrate_export_nwb_page(input_parameters)
