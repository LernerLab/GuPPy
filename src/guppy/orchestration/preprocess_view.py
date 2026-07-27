"""Step-3 (preprocessing) presentation, served by the persistent main app.

The preprocessing compute job runs as a subprocess and writes its outputs to disk. When
it finishes, ``home.py`` calls :func:`open_preprocess_view`, which opens a browser tab on
the main app's own (never-torn-down) server at the ``/preprocess-view`` route. That route
is served by :func:`build_preprocess_view`, which composes the marking / review page from
the on-disk outputs. Because the page lives on the persistent server, nothing is torn down
when the user is done — no "server connection lost" banner and no blocked backend.

The per-tab context (which folders / parameters to show) is passed through a token in the
URL and a small in-process registry, since the compute job and this server share the
main process.
"""

import logging
import webbrowser
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import panel as pn

from ..frontend.artifact_removal import build_preprocess_view_page
from ..utils.utils import get_all_stores_for_combining_data, select_run_folders

logger = logging.getLogger(__name__)

_VIEW_ROUTE = "preprocess-view"

# token -> (session_folders, inputParameters) for views awaiting a browser tab.
_PENDING_VIEWS: dict[str, tuple[list, dict]] = {}


def _resolve_run_folders(session_folders: list, inputParameters: dict) -> list[str]:
    """Return the output (run) folders the preprocessing job wrote, matching the worker.

    Mirrors the folder selection in ``preprocess.extractTsAndSignal`` /
    ``execute_zscore``: per-session run folders normally, or the first folder of each
    combine-group when ``combine_data`` is set.
    """
    selected_runs = inputParameters.get("selected_runs") or {}
    run_folders: list[str] = []
    for session in session_folders:
        run_folders.append(select_run_folders(session, selected_runs.get(session)))
    run_folders = list(np.concatenate(run_folders).flatten())

    if inputParameters["combine_data"] == True:
        return [group[0] for group in get_all_stores_for_combining_data(run_folders)]
    return run_folders


def _read_token() -> str:
    values = pn.state.session_args.get("token")
    if not values:
        return ""
    return values[0].decode() if isinstance(values[0], bytes) else str(values[0])


def _current_href() -> str:
    """The current session's browser URL, used to derive the main server's origin."""
    return pn.state.location.href


def build_preprocess_view() -> pn.template.BootstrapTemplate:
    """Per-session route factory for ``/preprocess-view`` — composes the marking/review page.

    Reads the token from the request query args, looks up the pending-view context, and
    renders the page from the on-disk outputs. An unknown/expired token yields a short
    notice instead.
    """
    template = pn.template.BootstrapTemplate(title="GuPPy — Preprocessing")
    token = _read_token()
    entry = _PENDING_VIEWS.get(token)
    if entry is None:
        template.main.append(pn.pane.Markdown("This view has expired. You can close this tab."))
        return template

    session_folders, inputParameters = entry
    run_folders = _resolve_run_folders(session_folders, inputParameters)
    page = build_preprocess_view_page(
        run_folders, inputParameters["removeArtifacts"], inputParameters["plot_zScore_dff"]
    )
    template.main.append(page)
    # Drop the token when the tab closes so the registry does not grow unbounded.
    pn.state.on_session_destroyed(lambda session_context: _PENDING_VIEWS.pop(token, None))
    return template


def open_preprocess_view(session_folders: list, inputParameters: dict) -> None:
    """Register a pending view and open its tab on the persistent main server.

    Parameters
    ----------
    session_folders : list
        The session directories the preprocessing job ran on (``inputParameters["session_folders"]``).
    inputParameters : dict
        The parameters the job ran with; supplies ``removeArtifacts`` / ``plot_zScore_dff`` /
        ``combine_data`` / ``selected_runs``.
    """
    token = uuid4().hex
    _PENDING_VIEWS[token] = (session_folders, inputParameters)
    parts = urlsplit(_current_href())
    url = f"{parts.scheme}://{parts.netloc}/{_VIEW_ROUTE}?token={token}"
    logger.info(f"Opening preprocessing view at {url}")
    webbrowser.open(url)
