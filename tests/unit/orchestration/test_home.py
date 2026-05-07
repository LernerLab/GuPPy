import panel as pn
import pytest

from guppy.orchestration.home import build_homepage


@pytest.fixture
def homepage(panel_extension):
    """Build a fresh homepage template for each test."""
    return build_homepage()


def test_returns_bootstrap_template(homepage):
    assert isinstance(homepage, pn.template.BootstrapTemplate)


def test_hooks_contains_onclick_process(homepage):
    assert "onclickProcess" in homepage._hooks
    assert callable(homepage._hooks["onclickProcess"])


def test_hooks_contains_onclick_visualization(homepage):
    assert "onclickVisualization" in homepage._hooks
    assert callable(homepage._hooks["onclickVisualization"])


def test_hooks_contains_get_input_parameters(homepage):
    assert "getInputParameters" in homepage._hooks
    assert callable(homepage._hooks["getInputParameters"])


def test_widgets_contains_files_1(homepage):
    assert "files_1" in homepage._widgets
    assert hasattr(homepage._widgets["files_1"], "value")


def test_onclick_process_calls_save_parameters(homepage, tmp_path, monkeypatch):
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    calls = []
    monkeypatch.setattr(
        "guppy.orchestration.home.save_parameters",
        lambda inputParameters: calls.append(inputParameters),
    )
    homepage._hooks["onclickProcess"]()
    assert len(calls) == 1
    assert isinstance(calls[0], dict)


def test_get_input_parameters_returns_dict(homepage, tmp_path):
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    result = homepage._hooks["getInputParameters"]()
    assert isinstance(result, dict)


def test_onclick_process_surfaces_no_folder_selected_as_panel_notification(homepage, monkeypatch):
    """When no folder is selected, getInputParameters raises and the click
    handler must surface the error as a persistent Panel notification (duration=0)
    rather than letting it propagate, and must not invoke the underlying worker."""
    homepage._widgets["files_1"].value = []

    save_parameters_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.home.save_parameters",
        lambda inputParameters: save_parameters_calls.append(inputParameters),
    )

    captured_notifications = []

    def fake_error(message, *, duration):
        captured_notifications.append({"message": message, "duration": duration})

    monkeypatch.setattr(pn.state.notifications, "error", fake_error)

    # Must not raise — the exception is caught and forwarded to Panel
    homepage._hooks["onclickProcess"]()

    assert len(captured_notifications) == 1
    assert "No folder is selected for analysis" in captured_notifications[0]["message"]
    assert captured_notifications[0]["duration"] == 0
    assert save_parameters_calls == []


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
