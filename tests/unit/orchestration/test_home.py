import threading
import time
from importlib.metadata import version

import panel as pn
import pytest

from guppy.orchestration.home import build_homepage
from guppy.utils import progress


def poll_until_stopped(capture_periodic, timeout=3.0):
    """Drive the poll callback the way the IOLoop would, until it reports the step finished.

    Completion is the worker thread exiting, which happens shortly after the worker body
    returns, so a single poll can land while the thread is still winding down.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        capture_periodic["poll"]()
        if capture_periodic["callback"].stopped:
            return
        time.sleep(0.01)
    raise AssertionError("poll callback never reported the step finished")


@pytest.fixture
def homepage(panel_extension):
    """Build a fresh homepage template for each test."""
    return build_homepage()


def test_returns_bootstrap_template(homepage):
    assert isinstance(homepage, pn.template.BootstrapTemplate)


def test_header_shows_installed_version(homepage):
    version_pane = homepage.header.objects[0]
    assert version_pane.object == f"GuPPy {version('guppy-neuro')}"


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


class TestMetadataHook:
    """Step 6 opens a page synchronously; Step 7's export goes through STEP_HANDLERS below."""

    def test_hooks_contain_metadata_and_export(self, homepage):
        assert callable(homepage._hooks["onclickMetadata"])
        assert callable(homepage._hooks["onclickExportNwb"])

    def test_onclick_metadata_invokes_worker_with_input_parameters(self, homepage, selected_session, monkeypatch):
        worker_calls = []
        monkeypatch.setattr(
            "guppy.orchestration.home.orchestrate_metadata_page", lambda params: worker_calls.append(params)
        )

        homepage._hooks["onclickMetadata"]()

        assert len(worker_calls) == 1
        assert isinstance(worker_calls[0], dict)

    def test_onclick_metadata_no_folder_notifies_and_skips_worker(self, homepage, monkeypatch):
        homepage._widgets["files_1"].value = []

        worker_calls = []
        monkeypatch.setattr(
            "guppy.orchestration.home.orchestrate_metadata_page", lambda params: worker_calls.append(params)
        )

        captured = []
        monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: captured.append(message))

        homepage._hooks["onclickMetadata"]()

        assert worker_calls == []
        assert any("No folder is selected for analysis" in message for message in captured)


# The pipeline steps that run a worker in a background thread behind a progress bar all share
# the `_run_worker_with_progress` closure, so launch, error surfacing, and the no-folder guard
# are asserted once for all of them. Only PSTH injects `curr_dir`.
STEP_HANDLERS = [
    ("onclickreaddata", "run_read_raw_data_step", False),
    ("onclickpreprocess", "run_preprocess_step", False),
    ("onclickRemoveArtifacts", "run_remove_artifacts_step", False),
    ("onclickpsth", "run_psth_step", True),
    ("onclickExportNwb", "run_export_nwb_step", False),
]


@pytest.fixture
def selected_session(homepage, tmp_path):
    """Select a single session folder so getInputParameters succeeds."""
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]
    return folder


class _FakePeriodicCallback:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


@pytest.fixture
def capture_periodic(monkeypatch):
    """Capture the poll callback registered via ``pn.state.add_periodic_callback`` without running it.

    The real callback fires on the server IOLoop, which is not running in unit tests;
    capturing it lets a test drive one poll deterministically.
    """
    captured = {}

    def fake_add_periodic_callback(callback, **kwargs):
        captured["poll"] = callback
        captured["callback"] = _FakePeriodicCallback()
        return captured["callback"]

    monkeypatch.setattr(pn.state, "add_periodic_callback", fake_add_periodic_callback)
    return captured


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_launches_worker_with_input_parameters(
    homepage, selected_session, monkeypatch, capture_periodic, hook_name, worker_attr, adds_curr_dir
):
    """Each step handler launches its worker in a background thread (without blocking the
    IOLoop) with the collected input parameters, and registers a progress poller. Only
    PSTH injects ``curr_dir``."""
    captured = []
    finished = threading.Event()

    def fake_worker(params):
        captured.append(params)
        finished.set()

    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", fake_worker)

    homepage._hooks[hook_name]()

    assert finished.wait(timeout=3), "worker thread did not run"
    assert len(captured) == 1
    assert isinstance(captured[0], dict)
    assert ("curr_dir" in captured[0]) == adds_curr_dir
    # A poller was registered on the IOLoop instead of the handler blocking on it.
    assert "poll" in capture_periodic


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_surfaces_progress_error_as_panel_notification(
    homepage, selected_session, monkeypatch, capture_periodic, hook_name, worker_attr, adds_curr_dir
):
    """When the worker reports failure through the progress channel, the poll callback
    surfaces it as a persistent Panel error notification (duration=0) and stops polling."""
    finished = threading.Event()

    def worker(params):
        # The worker runs inside the context the handler bound, so the module-level
        # channel reaches the StepProgress the poller reads.
        progress.fail("Step failed")
        finished.set()

    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", worker)

    captured_notifications = []
    monkeypatch.setattr(
        pn.state.notifications,
        "error",
        lambda message, *, duration: captured_notifications.append({"message": message, "duration": duration}),
    )

    homepage._hooks[hook_name]()
    assert finished.wait(timeout=3), "worker thread did not run"
    poll_until_stopped(capture_periodic)

    assert captured_notifications == [{"message": "Step failed", "duration": 0}]


@pytest.mark.parametrize("hook_name, worker_attr, adds_curr_dir", STEP_HANDLERS)
def test_step_handler_no_folder_selected_skips_worker(
    homepage, monkeypatch, capture_periodic, hook_name, worker_attr, adds_curr_dir
):
    """With no folder selected, getInputParameters raises; the handler must surface the
    error and never launch the worker or register a poller."""
    homepage._widgets["files_1"].value = []

    def _should_not_run(*args, **kwargs):
        raise AssertionError("worker must not run when no folder is selected")

    monkeypatch.setattr(f"guppy.orchestration.home.{worker_attr}", _should_not_run)

    captured_notifications = []
    monkeypatch.setattr(
        pn.state.notifications,
        "error",
        lambda message, *, duration: captured_notifications.append(message),
    )

    homepage._hooks[hook_name]()

    assert len(captured_notifications) == 1
    assert "No folder is selected for analysis" in captured_notifications[0]
    # No worker launched and no poller registered.
    assert "poll" not in capture_periodic


class TestExportNwbCompletionNotification:
    """Export writes its files and opens no result view, so a completion notification is the
    only sign it finished. It was lost when the step moved onto the background-thread pattern."""

    def test_successful_export_notifies_completion(self, homepage, selected_session, monkeypatch, capture_periodic):
        finished = threading.Event()
        monkeypatch.setattr("guppy.orchestration.home.run_export_nwb_step", lambda params: finished.set())

        captured = []
        monkeypatch.setattr(pn.state.notifications, "success", lambda message: captured.append(message))

        homepage._hooks["onclickExportNwb"]()
        assert finished.wait(timeout=3), "worker thread did not run"
        poll_until_stopped(capture_periodic)

        assert captured == ["Export to NWB complete."]

    def test_failed_export_notifies_the_error_and_not_success(
        self, homepage, selected_session, monkeypatch, capture_periodic
    ):
        finished = threading.Event()

        def worker(params):
            progress.fail("NWB export failed for 1 of 1 session(s): Photo_A (run1): boom")
            finished.set()

        monkeypatch.setattr("guppy.orchestration.home.run_export_nwb_step", worker)

        successes = []
        monkeypatch.setattr(pn.state.notifications, "success", lambda message: successes.append(message))
        monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: None)

        homepage._hooks["onclickExportNwb"]()
        assert finished.wait(timeout=3), "worker thread did not run"
        poll_until_stopped(capture_periodic)

        assert successes == []


def test_poll_reports_progress_without_completing_while_worker_runs(
    homepage, selected_session, monkeypatch, capture_periodic
):
    """A worker that reports every unit of its declared total must still not be treated as
    finished while its thread is alive.

    Step 3 undercounted its own total (artifact removal was not budgeted), so a poll that
    equated "counter reached total" with "step done" fired ``on_success`` partway through
    and opened the result view against half-written output. Completion is the thread
    exiting, so reaching the total early is now only a full-looking bar.
    """
    release = threading.Event()
    reported = threading.Event()

    def worker(params):
        progress.start(2)
        progress.advance(2)
        reported.set()
        release.wait(timeout=3)

    monkeypatch.setattr("guppy.orchestration.home.run_read_raw_data_step", worker)
    notifications = []
    monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: notifications.append(message))

    homepage._hooks["onclickreaddata"]()
    assert reported.wait(timeout=3), "worker thread did not run"
    capture_periodic["poll"]()

    # The bar shows the work as complete, but the step is not.
    read_progress = homepage._widgets["read_progress"]
    assert (read_progress.value, read_progress.max) == (2, 2)
    assert capture_periodic["callback"].stopped is False
    assert notifications == []

    release.set()


def test_second_step_launch_is_refused_while_one_is_running(homepage, selected_session, monkeypatch, capture_periodic):
    """The handlers no longer block the IOLoop, so a run-guard must reject a second launch
    until the first finishes (the poll, which clears the guard, is not driven here)."""
    launches = []
    finished = threading.Event()

    def fake_worker(params):
        launches.append(params)
        finished.set()

    monkeypatch.setattr("guppy.orchestration.home.run_read_raw_data_step", fake_worker)
    notifications = []
    monkeypatch.setattr(pn.state.notifications, "error", lambda message, *, duration: notifications.append(message))

    homepage._hooks["onclickreaddata"]()
    assert finished.wait(timeout=3), "first worker did not run"
    homepage._hooks["onclickreaddata"]()  # second launch while the first is still "running"

    assert len(launches) == 1
    assert any("already running" in message for message in notifications)


def test_preprocess_success_opens_the_result_view(homepage, selected_session, monkeypatch, capture_periodic):
    """On a successful preprocess run the completion path opens the preprocessing view."""

    def worker(params):
        progress.start(3)
        progress.advance(3)

    monkeypatch.setattr("guppy.orchestration.home.run_preprocess_step", worker)
    opened = []
    monkeypatch.setattr(
        "guppy.orchestration.home.open_preprocess_view",
        lambda session_folders, params: opened.append((session_folders, params)),
    )

    homepage._hooks["onclickpreprocess"]()
    poll_until_stopped(capture_periodic)

    assert len(opened) == 1
    session_folders, params = opened[0]
    assert session_folders == params["session_folders"]
    extract_progress = homepage._widgets["extract_progress"]
    assert (extract_progress.value, extract_progress.max) == (3, 3)


def test_psth_success_opens_the_transients_view(homepage, selected_session, monkeypatch, capture_periodic):
    """On a successful PSTH/transients run the completion path opens the transient-peaks view."""

    def worker(params):
        progress.start(3)
        progress.advance(3)

    monkeypatch.setattr("guppy.orchestration.home.run_psth_step", worker)
    opened = []
    monkeypatch.setattr(
        "guppy.orchestration.home.open_transients_view",
        lambda session_folders, params: opened.append((session_folders, params)),
    )

    homepage._hooks["onclickpsth"]()
    poll_until_stopped(capture_periodic)

    assert len(opened) == 1
    session_folders, params = opened[0]
    assert session_folders == params["session_folders"]


def test_success_snaps_the_bar_to_full_when_the_step_undercounted(
    homepage, selected_session, monkeypatch, capture_periodic
):
    """A step that finishes having reported fewer units than it declared must still leave the
    bar full, rather than stranded partway."""

    def worker(params):
        progress.start(10)
        progress.advance(4)

    monkeypatch.setattr("guppy.orchestration.home.run_read_raw_data_step", worker)

    homepage._hooks["onclickreaddata"]()
    poll_until_stopped(capture_periodic)

    read_progress = homepage._widgets["read_progress"]
    assert (read_progress.value, read_progress.max) == (10, 10)


class TestSelectArtifactWindowsHandler:
    """The marking step runs no compute: it opens a page synchronously, with no progress bar."""

    def test_opens_the_page_with_the_collected_input_parameters(self, homepage, selected_session, monkeypatch):
        captured = []
        monkeypatch.setattr(
            "guppy.orchestration.home.orchestrate_select_artifact_windows", lambda params: captured.append(params)
        )

        homepage._hooks["onclickSelectArtifactWindows"]()

        assert len(captured) == 1
        assert captured[0]["session_folders"] == [str(selected_session)]

    def test_surfaces_value_error_as_panel_notification(self, homepage, selected_session, monkeypatch):
        error_text = "No preprocessing outputs found in '/a/output_1'."

        def _raise(params):
            raise ValueError(error_text)

        monkeypatch.setattr("guppy.orchestration.home.orchestrate_select_artifact_windows", _raise)
        captured_notifications = []
        monkeypatch.setattr(
            pn.state.notifications,
            "error",
            lambda message, *, duration: captured_notifications.append({"message": message, "duration": duration}),
        )

        # Must not raise — the ValueError is caught and forwarded to Panel.
        homepage._hooks["onclickSelectArtifactWindows"]()

        assert len(captured_notifications) == 1
        assert error_text in captured_notifications[0]["message"]
        assert captured_notifications[0]["duration"] == 0
