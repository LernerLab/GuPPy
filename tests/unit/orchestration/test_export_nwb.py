"""Unit tests for the upfront NWB-export prerequisite check.

`orchestrate_export_nwb_page` reads each selected session's ``GuPPyParamtersUsed.json`` and
aborts the whole batch before writing anything if any session was processed with the
``concatenate`` artifact-removal method, which re-times kept samples and breaks alignment to
the acquisition clock.
"""

import json
import os

import panel as pn
import pytest

from guppy.orchestration import export_nwb as export_nwb_module
from guppy.orchestration.export_nwb import (
    _prune_absent_commanded_voltage,
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


def _metadata_with_commanded_voltage(stream_names):
    """Build a metadata dict with one CommandedVoltageSeries per stream and a table row referencing each."""
    return {
        "Ophys": {
            "FiberPhotometry": {
                "CommandedVoltageSeries": [{"name": f"cvs_{stream}", "stream_name": stream} for stream in stream_names],
                "FiberPhotometryTable": {
                    "rows": [
                        {"name": index, "commanded_voltage_series": f"cvs_{stream}"}
                        for index, stream in enumerate(stream_names)
                    ]
                },
            }
        }
    }


class TestPruneAbsentCommandedVoltage:
    def test_drops_absent_keeps_present_and_clears_dangling_row_refs(self):
        metadata = _metadata_with_commanded_voltage(["Fi1d", "Fi1r"])
        _prune_absent_commanded_voltage(metadata, available_streams={"Fi1d"})
        fiber_photometry = metadata["Ophys"]["FiberPhotometry"]
        assert fiber_photometry["CommandedVoltageSeries"] == [{"name": "cvs_Fi1d", "stream_name": "Fi1d"}]
        rows = fiber_photometry["FiberPhotometryTable"]["rows"]
        assert rows[0]["commanded_voltage_series"] == "cvs_Fi1d"
        assert "commanded_voltage_series" not in rows[1]

    def test_all_absent_removes_the_series_key_entirely(self):
        metadata = _metadata_with_commanded_voltage(["Fi1d", "Fi1r"])
        _prune_absent_commanded_voltage(metadata, available_streams=set())
        fiber_photometry = metadata["Ophys"]["FiberPhotometry"]
        assert "CommandedVoltageSeries" not in fiber_photometry
        for row in fiber_photometry["FiberPhotometryTable"]["rows"]:
            assert "commanded_voltage_series" not in row

    def test_no_fiber_photometry_section_is_a_noop(self):
        metadata = {}
        _prune_absent_commanded_voltage(metadata, available_streams={"Fi1d"})
        assert metadata == {}

    def test_empty_commanded_voltage_list_is_a_noop(self):
        metadata = {"Ophys": {"FiberPhotometry": {"CommandedVoltageSeries": []}}}
        _prune_absent_commanded_voltage(metadata, available_streams={"Fi1d"})
        assert metadata == {"Ophys": {"FiberPhotometry": {"CommandedVoltageSeries": []}}}


class _FakeProgressBar:
    def __init__(self):
        self.max = None
        self.value = None


class _FakeNotifications:
    def __init__(self):
        self.successes = []
        self.errors = []

    def success(self, message):
        self.successes.append(message)

    def error(self, message, duration=None):
        self.errors.append({"message": message, "duration": duration})


class TestOrchestrateExportNwbPage:
    @pytest.fixture
    def two_sessions(self):
        return {"selectedOutputs": {"/data/Photo_A": ["run1"], "/data/Photo_B": ["run1"]}}

    @pytest.fixture
    def notifications(self, monkeypatch):
        # pn.state.notifications is a read-only property; replace it at the class level so the
        # success/error branches in the export loop have a recording sink to write to.
        fake = _FakeNotifications()
        monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: fake))
        return fake

    def test_exports_each_session_and_advances_progress(self, two_sessions, notifications, monkeypatch):
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)
        exported = []
        monkeypatch.setattr(
            export_nwb_module,
            "export_session_to_nwb",
            lambda **kwargs: exported.append(kwargs["nwbfile_path"]),
        )
        progress_bar = _FakeProgressBar()

        orchestrate_export_nwb_page(two_sessions, progress_bar=progress_bar)

        assert len(exported) == 2
        assert progress_bar.max == 2
        assert progress_bar.value == 2
        assert len(notifications.successes) == 2
        assert notifications.errors == []

    def test_one_failed_session_is_reported_and_skipped(self, two_sessions, notifications, monkeypatch):
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)

        def export(**kwargs):
            if "Photo_B" in kwargs["nwbfile_path"]:
                raise RuntimeError("converter blew up")

        monkeypatch.setattr(export_nwb_module, "export_session_to_nwb", export)
        progress_bar = _FakeProgressBar()

        # One failure must not abort the batch.
        orchestrate_export_nwb_page(two_sessions, progress_bar=progress_bar)

        assert len(notifications.successes) == 1
        assert len(notifications.errors) == 1
        assert "converter blew up" in notifications.errors[0]["message"]
        assert notifications.errors[0]["duration"] == 0
        assert progress_bar.value == 2  # progress still advances past the failed session

    def test_runs_without_progress_bar_or_notifications(self, two_sessions, monkeypatch):
        # No progress bar and no notification area configured: the loop must skip both cleanly.
        monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: None))
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)
        exported = []
        monkeypatch.setattr(
            export_nwb_module,
            "export_session_to_nwb",
            lambda **kwargs: exported.append(kwargs["nwbfile_path"]),
        )

        orchestrate_export_nwb_page(two_sessions, progress_bar=None)

        assert len(exported) == 2

    def test_failure_without_notifications_still_continues(self, two_sessions, monkeypatch):
        monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: None))
        monkeypatch.setattr(export_nwb_module, "_validate_artifact_removal_methods", lambda pairs: None)

        def export(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(export_nwb_module, "export_session_to_nwb", export)

        # No notification sink, but the batch must not raise.
        orchestrate_export_nwb_page(two_sessions, progress_bar=None)
