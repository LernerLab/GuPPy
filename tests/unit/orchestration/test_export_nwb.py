"""Unit tests for the upfront NWB-export prerequisite checks and the export batch loop.

`orchestrate_export_nwb` reads each selected run's recorded artifact provenance and aborts the
whole batch before writing anything if any run had its artifacts removed by the ``concatenate``
method, which re-times kept samples and breaks alignment to the acquisition clock. It aborts the
same way for combined runs, which have no single session the collapsed outputs belong to. Each
surviving session is then routed by its resolved source: raw acquisition files through the
converter, an NWB file through the standalone interface.
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
            _validate_artifact_removal_methods(pairs=[(str(session_path), "run1")])
        message = str(excinfo.value)
        assert "Photo_session (run1)" in message
        assert "concatenate" in message
        assert "replace with NaN" in message
        assert "https://github.com/LernerLab/GuPPy/issues/new" in message

    def test_replace_with_nan_does_not_abort(self, session_path):
        self._write_parameters(session_path, {"removeArtifacts": True, "artifactsRemovalMethod": "replace with NaN"})
        _validate_artifact_removal_methods(pairs=[(str(session_path), "run1")])

    def test_concatenate_without_remove_artifacts_does_not_abort(self, session_path):
        # removeArtifacts is False, so the method is irrelevant and must not trigger the abort.
        self._write_parameters(session_path, {"removeArtifacts": False, "artifactsRemovalMethod": "concatenate"})
        _validate_artifact_removal_methods(pairs=[(str(session_path), "run1")])

    def test_snapshot_without_artifact_keys_does_not_abort(self, session_path):
        # The shape a Step-3-only run leaves behind once artifact removal became its own step:
        # the snapshot records no removal at all, which must read as "artifacts not removed".
        self._write_parameters(session_path, {"combine_data": False})
        _validate_artifact_removal_methods(pairs=[(str(session_path), "run1")])

    def test_orchestrate_aborts_before_any_export(self, session_path):
        # End-to-end through the public entry point: the offending config must raise the
        # ValueError before the export loop touches neuroconv.
        self._write_parameters(session_path, {"removeArtifacts": True, "artifactsRemovalMethod": "concatenate"})
        input_parameters = {"selected_runs": {str(session_path): ["run1"]}, "combine_data": False}
        with pytest.raises(ValueError, match="does not support the 'concatenate'"):
            orchestrate_export_nwb(input_parameters)


class TestValidateDataNotCombined:
    """Combining collapses a run group into one output directory, so the per-session export
    has no session its outputs belong to. Refused upfront rather than half-exported."""

    def test_combined_run_aborts_before_any_export(self, tmp_path):
        # Refused before the artifact-provenance check reads anything, so a bare path is enough.
        input_parameters = {"selected_runs": {str(tmp_path / "Photo_A"): ["run1"]}, "combine_data": True}

        with pytest.raises(ValueError) as excinfo:
            orchestrate_export_nwb(input_parameters)

        assert "does not support combine_data=True" in str(excinfo.value)


@pytest.fixture
def bound_step():
    """Bind a StepProgress for the duration of one test, as ``home.py`` does per step run."""
    step = StepProgress()
    token = _current_step.set(step)
    yield step
    _current_step.reset(token)


def _make_session(base_dir, name, *, acquisition_files):
    """Create a session folder with one ``run1`` output directory and the given raw files."""
    session = base_dir / name
    (session / f"{name}_output_run1").mkdir(parents=True)
    for filename, contents in acquisition_files.items():
        (session / filename).write_text(contents)
    return session


# Header shapes the format detector classifies by; see tests/unit/extractors/test_detect_acquisition_formats.py.
_DORIC_CSV = "Time,AIn-1,AIn-2\n--,--,--\n0.0,1.0,2.0\n"
_NPM_CSV = "FrameCounter,Timestamp,LedState,Region0G,Region1G\n0,0.0,1,10.0,20.0\n"


class TestOrchestrateExportNwb:
    """Driven against real session folders on disk: only ``export_session_to_nwb`` -- the slow
    neuroconv boundary -- is stubbed, so the prerequisite checks and source resolution run for real."""

    @pytest.fixture
    def two_sessions(self, tmp_path):
        """Two Doric sessions, both exportable."""
        _make_session(tmp_path, "Photo_A", acquisition_files={"signal.csv": _DORIC_CSV})
        _make_session(tmp_path, "Photo_B", acquisition_files={"signal.csv": _DORIC_CSV})
        return {
            "selected_runs": {str(tmp_path / "Photo_A"): ["run1"], str(tmp_path / "Photo_B"): ["run1"]},
            "combine_data": False,
        }

    @pytest.fixture
    def exported(self, monkeypatch):
        """Record the calls the export loop would make, without invoking neuroconv."""
        calls = []
        monkeypatch.setattr(export_nwb_module, "export_session_to_nwb", lambda **kwargs: calls.append(kwargs))
        return calls

    def test_passes_each_session_its_resolved_acquisition_format(self, two_sessions, exported):
        # The converter reads the raw folder through the format detected in it, so a session must
        # never be exported as whatever the previous PR hardcoded.
        orchestrate_export_nwb(two_sessions)

        assert [call["acquisition_format"] for call in exported] == ["doric", "doric"]
        assert [call["nwb_source"] for call in exported] == [None, None]

    def test_passes_an_nwb_sourced_session_the_file_it_came_from(self, tmp_path, exported):
        session = _make_session(tmp_path, "Photo_nwb", acquisition_files={"session.nwb": ""})
        input_parameters = {"selected_runs": {str(session): ["run1"]}, "combine_data": False}

        orchestrate_export_nwb(input_parameters)

        assert [call["nwb_source"] for call in exported] == [str(session / "session.nwb")]
        assert [call["acquisition_format"] for call in exported] == ["nwb"]

    def test_unsupported_format_fails_only_that_session(self, tmp_path, bound_step, exported):
        # Photo_B holds traces from two acquisition systems, which the export refuses. The
        # refusal must take down that session only, not the batch.
        _make_session(tmp_path, "Photo_A", acquisition_files={"signal.csv": _DORIC_CSV})
        _make_session(tmp_path, "Photo_B", acquisition_files={"doric.csv": _DORIC_CSV, "npm.csv": _NPM_CSV})
        input_parameters = {
            "selected_runs": {str(tmp_path / "Photo_A"): ["run1"], str(tmp_path / "Photo_B"): ["run1"]},
            "combine_data": False,
        }

        orchestrate_export_nwb(input_parameters)

        assert [call["acquisition_format"] for call in exported] == ["doric"]
        assert bound_step.value == 2
        assert "NWB export failed for 1 of 2 session(s): Photo_B (run1):" in bound_step.error_message
        assert "more than one acquisition system" in bound_step.error_message

    def test_exports_each_session_and_advances_progress(self, tmp_path, two_sessions, exported, bound_step):
        orchestrate_export_nwb(two_sessions)

        assert [call["nwbfile_path"] for call in exported] == [
            str(tmp_path / "Photo_A" / "Photo_A_output_run1" / "Photo_A_output_run1.nwb"),
            str(tmp_path / "Photo_B" / "Photo_B_output_run1" / "Photo_B_output_run1.nwb"),
        ]
        assert bound_step.total == 2
        assert bound_step.value == 2
        assert bound_step.error_message is None

    def test_one_failed_session_is_reported_and_skipped(self, tmp_path, two_sessions, bound_step, monkeypatch):
        exported = []

        def export(**kwargs):
            if "Photo_B" in kwargs["nwbfile_path"]:
                raise RuntimeError("converter blew up")
            exported.append(kwargs["nwbfile_path"])

        monkeypatch.setattr(export_nwb_module, "export_session_to_nwb", export)

        # One failure must not abort the batch.
        orchestrate_export_nwb(two_sessions)

        assert exported == [str(tmp_path / "Photo_A" / "Photo_A_output_run1" / "Photo_A_output_run1.nwb")]
        # Progress still advances past the failed session, so the bar reaches its total.
        assert bound_step.value == 2
        assert bound_step.error_message == "NWB export failed for 1 of 2 session(s): Photo_B (run1): converter blew up"

    def test_runs_unbound_with_no_progress_channel(self, tmp_path, two_sessions, exported):
        # The headless path (guppy.testing.api) binds no StepProgress; emitting must be a no-op
        # rather than requiring the caller to pass a sink.
        orchestrate_export_nwb(two_sessions)

        assert [call["nwbfile_path"] for call in exported] == [
            str(tmp_path / "Photo_A" / "Photo_A_output_run1" / "Photo_A_output_run1.nwb"),
            str(tmp_path / "Photo_B" / "Photo_B_output_run1" / "Photo_B_output_run1.nwb"),
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
            run_export_nwb_step({"selected_runs": {str(session): ["run1"]}, "combine_data": False})

        assert "does not support the 'concatenate'" in bound_step.error_message
