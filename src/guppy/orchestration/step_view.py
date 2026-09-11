"""Shared plumbing for serving a pipeline step's result view on the persistent main app.

Each step that shows results after its compute job (preprocess, transients, …) opens a
browser tab on a dedicated route of the always-running main server. The compute job and
the server share the main process, so the per-tab context is passed through a URL token
and an in-process registry. Nothing is ever torn down, so the browser never sees Bokeh's
"server connection lost" banner.

A step wires this up by constructing a :class:`StepView` with its route, tab title, and a
``build_page(session_folders, inputParameters) -> Viewable`` composer, then exposing the
view's ``route_factory`` (for ``main.py``'s route map) and ``open`` (for the step's
completion hook in ``home.py``).
"""

import logging
import webbrowser
from collections.abc import Callable
from urllib.parse import urlsplit
from uuid import uuid4

import panel as pn

from ..utils.utils import (
    resolve_run_folders,  # noqa: F401  (re-exported for the step views)
)

logger = logging.getLogger(__name__)


def _read_token() -> str:
    values = pn.state.session_args.get("token")
    if not values:
        return ""
    return values[0].decode() if isinstance(values[0], bytes) else str(values[0])


def _current_href() -> str:
    """The current session's browser URL, used to derive the main server's origin."""
    return pn.state.location.href


class StepView:
    """A single step's result-view route + opener on the persistent server."""

    def __init__(self, route: str, title: str, build_page: Callable[[list, dict], "pn.viewable.Viewable"]) -> None:
        self.route = route
        self.title = title
        self.build_page = build_page
        # token -> (session_folders, inputParameters) for views awaiting a browser tab.
        self.pending: dict[str, tuple[list, dict]] = {}

    def route_factory(self) -> pn.template.BootstrapTemplate:
        """Per-session factory for the step's route — composes the page for the tab's token."""
        template = pn.template.BootstrapTemplate(title=self.title)
        token = _read_token()
        entry = self.pending.get(token)
        if entry is None:
            template.main.append(pn.pane.Markdown("This view has expired. You can close this tab."))
            return template
        session_folders, inputParameters = entry
        template.main.append(self.build_page(session_folders, inputParameters))
        # Drop the token when the tab closes so the registry does not grow unbounded.
        pn.state.on_session_destroyed(lambda session_context: self.pending.pop(token, None))
        return template

    def open(self, session_folders: list, inputParameters: dict) -> None:
        """Register a pending view and open its tab on the persistent main server."""
        token = uuid4().hex
        self.pending[token] = (session_folders, inputParameters)
        parts = urlsplit(_current_href())
        url = f"{parts.scheme}://{parts.netloc}/{self.route}?token={token}"
        logger.info("Opening %s at %s", self.route, url)
        webbrowser.open(url)
