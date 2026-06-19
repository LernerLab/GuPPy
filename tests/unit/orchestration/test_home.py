import sys

import panel as pn
import pytest

from guppy.orchestration import home as home_module
from guppy.orchestration.home import (
    build_homepage,
    preprocess,
    psthComputation,
    readRawData,
)


@pytest.fixture
def homepage(panel_extension):
    """Build a fresh homepage template for each test."""
    return build_homepage()


def test_returns_bootstrap_template(homepage):
    assert isinstance(homepage, pn.template.BootstrapTemplate)


def test_hooks_omits_removed_save_parameters_hook(homepage):
    # The manual "Save Input Parameters" button (and its onclickProcess hook) was
    # removed; each consuming step now writes the snapshot automatically.
    assert "onclickProcess" not in homepage._hooks


def test_hooks_contains_onclick_visualization(homepage):
    assert "onclickVisualization" in homepage._hooks
    assert callable(homepage._hooks["onclickVisualization"])


def test_hooks_contains_get_input_parameters(homepage):
    assert "getInputParameters" in homepage._hooks
    assert callable(homepage._hooks["getInputParameters"])


def test_widgets_contains_files_1(homepage):
    assert "files_1" in homepage._widgets
    assert hasattr(homepage._widgets["files_1"], "value")


def test_get_input_parameters_returns_dict(homepage, tmp_path):
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    result = homepage._hooks["getInputParameters"]()
    assert isinstance(result, dict)


def test_onclick_surfaces_no_folder_selected_as_panel_notification(homepage, monkeypatch):
    """When no folder is selected, getInputParameters raises and the click
    handler must surface the error as a persistent Panel notification (duration=0)
    rather than letting it propagate, and must not invoke the underlying worker."""
    homepage._widgets["files_1"].value = []

    visualize_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.home.visualizeResults",
        lambda params: visualize_calls.append(params),
    )

    captured_notifications = []

    def fake_error(message, *, duration):
        captured_notifications.append({"message": message, "duration": duration})

    monkeypatch.setattr(pn.state.notifications, "error", fake_error)

    # Must not raise — the exception is caught and forwarded to Panel
    homepage._hooks["onclickVisualization"]()

    assert len(captured_notifications) == 1
    assert "No folder is selected for analysis" in captured_notifications[0]["message"]
    assert captured_notifications[0]["duration"] == 0
    assert visualize_calls == []


def test_onclick_visualization_surfaces_value_error_as_panel_notification(homepage, tmp_path, monkeypatch):
    """When visualizeResults raises ValueError the error must be surfaced as a
    persistent Panel error notification (duration=0) rather than propagated to
    the caller."""
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    error_text = "Metric 'z_score' not found in step-5 outputs"

    def _raise(params):
        raise ValueError(error_text)

    monkeypatch.setattr(
        "guppy.orchestration.home.visualizeResults",
        _raise,
    )

    captured_notifications = []

    def fake_error(message, *, duration):
        captured_notifications.append({"message": message, "duration": duration})

    monkeypatch.setattr(pn.state.notifications, "error", fake_error)

    # Must not raise — the ValueError is caught and forwarded to Panel
    homepage._hooks["onclickVisualization"]()

    assert len(captured_notifications) == 1
    assert error_text in captured_notifications[0]["message"]
    assert captured_notifications[0]["duration"] == 0


class TestSubprocessSteps:
    """The read/preprocess/psth steps shell out to a module entry point with the params as JSON."""

    @pytest.mark.parametrize(
        "function, module_suffix",
        [
            (readRawData, "guppy.orchestration.read_raw_data"),
            (preprocess, "guppy.orchestration.preprocess"),
            (psthComputation, "guppy.orchestration.psth"),
        ],
    )
    def test_invokes_subprocess_with_module_and_json_params(self, function, module_suffix, monkeypatch):
        calls = []
        monkeypatch.setattr(home_module.subprocess, "call", lambda command: calls.append(command))

        function({"a": 1})

        assert calls == [[sys.executable, "-m", module_suffix, '{"a": 1}']]


class TestMetadataAndExportHooks:
    def test_hooks_contain_metadata_and_export(self, homepage):
        assert callable(homepage._hooks["onclickMetadata"])
        assert callable(homepage._hooks["onclickExportNwb"])

    def test_onclick_metadata_invokes_worker_with_input_parameters(self, homepage, tmp_path, monkeypatch):
        folder = tmp_path / "session1"
        folder.mkdir()
        homepage._widgets["files_1"].value = [str(folder)]

        worker_calls = []
        monkeypatch.setattr(home_module, "orchestrate_metadata_page", lambda params: worker_calls.append(params))

        homepage._hooks["onclickMetadata"]()

        assert len(worker_calls) == 1
        assert isinstance(worker_calls[0], dict)

    def test_onclick_metadata_no_folder_notifies_and_skips_worker(self, homepage, monkeypatch):
        homepage._widgets["files_1"].value = []

        worker_calls = []
        monkeypatch.setattr(home_module, "orchestrate_metadata_page", lambda params: worker_calls.append(params))

        captured = []
        monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: captured.append(message))

        homepage._hooks["onclickMetadata"]()

        assert worker_calls == []
        assert any("No folder is selected for analysis" in message for message in captured)

    def test_onclick_export_invokes_worker_with_progress_bar(self, homepage, tmp_path, monkeypatch):
        folder = tmp_path / "session1"
        folder.mkdir()
        homepage._widgets["files_1"].value = [str(folder)]

        worker_calls = []
        monkeypatch.setattr(
            home_module,
            "orchestrate_export_nwb_page",
            lambda params, progress_bar: worker_calls.append((params, progress_bar)),
        )

        homepage._hooks["onclickExportNwb"]()

        assert len(worker_calls) == 1
        params, progress_bar = worker_calls[0]
        assert isinstance(params, dict)
        assert progress_bar is not None

    def test_onclick_export_no_folder_notifies_and_skips_worker(self, homepage, monkeypatch):
        homepage._widgets["files_1"].value = []

        worker_calls = []
        monkeypatch.setattr(
            home_module,
            "orchestrate_export_nwb_page",
            lambda params, progress_bar: worker_calls.append(params),
        )

        captured = []
        monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: captured.append(message))

        homepage._hooks["onclickExportNwb"]()

        assert worker_calls == []
        assert any("No folder is selected for analysis" in message for message in captured)

    def test_onclick_export_surfaces_value_error_as_notification(self, homepage, tmp_path, monkeypatch):
        folder = tmp_path / "session1"
        folder.mkdir()
        homepage._widgets["files_1"].value = [str(folder)]

        def _raise(params, progress_bar):
            raise ValueError("export prerequisite failed")

        monkeypatch.setattr(home_module, "orchestrate_export_nwb_page", _raise)

        captured = []
        monkeypatch.setattr(
            pn.state.notifications,
            "error",
            lambda message, *, duration: captured.append({"message": message, "duration": duration}),
        )

        homepage._hooks["onclickExportNwb"]()

        assert len(captured) == 1
        assert "export prerequisite failed" in captured[0]["message"]
        assert captured[0]["duration"] == 0
