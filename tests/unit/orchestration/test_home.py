import panel as pn
import pytest

from guppy.orchestration.home import build_homepage


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

    error_text = "Metric 'z_score' not found in step-4 outputs"

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


# The three pipeline steps that run a worker in a background thread behind a progress
# bar all share the `_run_worker_with_progress` closure. Only PSTH injects `curr_dir`.
STEP_HANDLERS = [
    ("onclickreaddata", "readRawData", False),
    ("onclickpreprocess", "preprocess", False),
    ("onclickpsth", "psthComputation", True),
]


@pytest.fixture
def selected_session(homepage, tmp_path):
    """Select a single session folder so getInputParameters succeeds."""
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]
    return folder


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_runs_worker_with_input_parameters(
    homepage, selected_session, monkeypatch, hook_name, worker_attr, adds_curr_dir
):
    """Each step handler launches its worker (in a background thread that is joined
    before returning) with the collected input parameters. Only PSTH injects
    ``curr_dir``."""
    captured = []
    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", lambda params: captured.append(params))
    # Stub the progress poll to report success (no error) without touching PB_STEPS_FILE.
    monkeypatch.setattr("guppy.orchestration.home.readPBIncrementValues", lambda progress, *, file_path: "")

    homepage._hooks[hook_name]()

    assert len(captured) == 1
    assert isinstance(captured[0], dict)
    assert ("curr_dir" in captured[0]) == adds_curr_dir


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_surfaces_progress_error_as_panel_notification(
    homepage, selected_session, monkeypatch, hook_name, worker_attr, adds_curr_dir
):
    """When the progress poll returns a non-empty error message, the handler surfaces
    it as a persistent Panel error notification (duration=0)."""
    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", lambda params: None)
    error_text = "Step failed in subprocess"
    monkeypatch.setattr("guppy.orchestration.home.readPBIncrementValues", lambda progress, *, file_path: error_text)

    captured_notifications = []

    def fake_error(message, *, duration):
        captured_notifications.append({"message": message, "duration": duration})

    monkeypatch.setattr(pn.state.notifications, "error", fake_error)

    homepage._hooks[hook_name]()

    assert captured_notifications == [{"message": error_text, "duration": 0}]


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_no_folder_selected_skips_worker(homepage, monkeypatch, hook_name, worker_attr, adds_curr_dir):
    """With no folder selected, getInputParameters raises; the handler must surface the
    error and never launch the worker or poll for progress."""
    homepage._widgets["files_1"].value = []

    captured = []
    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", lambda params: captured.append(params))

    def _should_not_run(*args, **kwargs):
        raise AssertionError("progress polling must not run when no folder is selected")

    monkeypatch.setattr("guppy.orchestration.home.readPBIncrementValues", _should_not_run)

    captured_notifications = []
    monkeypatch.setattr(
        pn.state.notifications,
        "error",
        lambda message, *, duration: captured_notifications.append(message),
    )

    homepage._hooks[hook_name]()

    assert captured == []
    assert len(captured_notifications) == 1
    assert "No folder is selected for analysis" in captured_notifications[0]
