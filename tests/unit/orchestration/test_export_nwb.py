"""Unit tests for the upfront NWB-export prerequisite check.

`orchestrate_export_nwb` reads each selected session's ``GuPPyParamtersUsed.json`` and
aborts the whole batch before writing anything if any session was processed with the
``concatenate`` artifact-removal method, which re-times kept samples and breaks alignment to
the acquisition clock.
"""

import json
import os

import pytest

from guppy.orchestration import export_nwb as export_nwb_module
from guppy.orchestration.export_nwb import (
    _validate_artifact_removal_methods,
    orchestrate_export_nwb,
    run_export_nwb_step,
)
from guppy.utils.progress import StepProgress, _current_step


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
        input_parameters = {"selected_runs": {str(session_path): ["run1"]}}
        with pytest.raises(ValueError, match="does not support the 'concatenate'"):
            orchestrate_export_nwb(input_parameters)


@pytest.fixture
def bound_step():
    """Bind a StepProgress for the duration of one test, as ``home.py`` does per step run."""
    step = StepProgress()
    token = _current_step.set(step)
    yield step
    _current_step.reset(token)


class TestOrchestrateExportNwb:
    @pytest.fixture
    def two_sessions(self):
        return {"selected_runs": {"/data/Photo_A": ["run1"], "/data/Photo_B": ["run1"]}}

    @pytest.fixture
    def exported(self, monkeypatch):
        """Record the NWB paths the export loop would write, without invoking neuroconv."""
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)
        paths = []
        monkeypatch.setattr(
            export_nwb_module,
            "export_session_to_nwb",
            lambda **kwargs: paths.append(kwargs["nwbfile_path"]),
        )
        return paths

    def test_exports_each_session_and_advances_progress(self, two_sessions, exported, bound_step):
        orchestrate_export_nwb(two_sessions)

        assert exported == [
            "/data/Photo_A/Photo_A_output_run1/Photo_A_output_run1.nwb",
            "/data/Photo_B/Photo_B_output_run1/Photo_B_output_run1.nwb",
        ]
        assert bound_step.total == 2
        assert bound_step.value == 2
        assert bound_step.error_message is None

    def test_one_failed_session_is_reported_and_skipped(self, two_sessions, bound_step, monkeypatch):
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)
        exported = []

        def export(**kwargs):
            if "Photo_B" in kwargs["nwbfile_path"]:
                raise RuntimeError("converter blew up")
            exported.append(kwargs["nwbfile_path"])

        monkeypatch.setattr(export_nwb_module, "export_session_to_nwb", export)

        # One failure must not abort the batch.
        orchestrate_export_nwb(two_sessions)

        assert exported == ["/data/Photo_A/Photo_A_output_run1/Photo_A_output_run1.nwb"]
        # Progress still advances past the failed session, so the bar reaches its total.
        assert bound_step.value == 2
        assert bound_step.error_message == "NWB export failed for 1 of 2 session(s): Photo_B (run1): converter blew up"

    def test_runs_unbound_with_no_progress_channel(self, two_sessions, exported):
        # The headless path (guppy.testing.api) binds no StepProgress; emitting must be a no-op
        # rather than requiring the caller to pass a sink.
        orchestrate_export_nwb(two_sessions)

        assert exported == [
            "/data/Photo_A/Photo_A_output_run1/Photo_A_output_run1.nwb",
            "/data/Photo_B/Photo_B_output_run1/Photo_B_output_run1.nwb",
        ]


class TestRunExportNwbStep:
    def test_validation_failure_is_reported_through_the_progress_channel(self, tmp_path, bound_step):
        """The upfront concatenate check runs inside the worker thread, so its ValueError must
        reach the progress error channel — that is how the GUI poller surfaces it."""
        session = tmp_path / "Photo_session"
        output_dir = session / "Photo_session_output_run1"
        output_dir.mkdir(parents=True)
        with open(output_dir / "GuPPyParamtersUsed.json", "w") as parameters_file:
            json.dump({"removeArtifacts": True, "artifactsRemovalMethod": "concatenate"}, parameters_file)

        with pytest.raises(ValueError, match="does not support the 'concatenate'"):
            run_export_nwb_step({"selected_runs": {str(session): ["run1"]}})

        assert "does not support the 'concatenate'" in bound_step.error_message
