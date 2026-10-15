"""Shared plumbing for serving the pages pipeline steps produce.

Each step that shows results after its compute job (preprocess, transients, …) serves them on
a dedicated route of the always-running main server. The compute job and the server share the
main process, so the per-tab context is passed through a URL token and an in-process registry.
Nothing is ever torn down, so the browser never sees Bokeh's "server connection lost" banner.

A step wires this up by constructing a :class:`StepView` with its route, tab title, and a
``build_page(session_folders, inputParameters) -> Viewable`` composer, then exposing the
view's ``route_factory`` (for ``app.py``'s route map) and ``open`` (for the step's
completion hook in ``home.py``).

Pages built before they are shown (Label Stores, the visualization dashboard, …) are served on a
port of their own instead: :func:`serve_pages` serves all the pages one click produced as a
single page, one in-page tab each, and :func:`serve_page` serves a page that is always alone.

None of them opens a browser tab. Each yields a link, which the homepage shows for the user to
open.
"""

import logging
from collections.abc import Callable
from uuid import uuid4

import panel as pn

from ..frontend.frontend_utils import scanPortsAndFind
from ..utils import page_links
from ..utils.utils import (
    resolve_run_folders,  # noqa: F401  (re-exported for the step views)
)

logger = logging.getLogger(__name__)


def _read_token() -> str:
    values = pn.state.session_args.get("token")
    if not values:
        return ""
    return values[0].decode() if isinstance(values[0], bytes) else str(values[0])


class StepView:
    """A single step's result-view route on the persistent server, and the link that opens it."""

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
        """Register a pending view and offer a link to it on the persistent main server."""
        token = uuid4().hex
        self.pending[token] = (session_folders, inputParameters)
        page_links.offer_link(title=self.title, href=f"{self.route}?token={token}")


def _serve(*, template: pn.template.BaseTemplate) -> str:
    port = scanPortsAndFind(start_port=5000, end_port=5200)
    template.show(port=port, open=False)
    return f"http://localhost:{port}/"


def serve_page(*, template: pn.template.BaseTemplate, title: str) -> None:
    """Serve a finished page on a free port and offer a link to it.

    Parameters
    ----------
    template : pn.template.BaseTemplate
        The page to serve.
    title : str
        Name of the page, shown in its link.
    """
    page_links.offer_link(title=title, href=_serve(template=template))


def serve_pages(*, pages: list[page_links.Page], title: str, plural_noun: str) -> page_links.PageLink:
    """Serve the pages one click produced as a single page on a free port.

    A lone page is served as it is. Several are combined into the first page's template, keeping
    its styling, with each page's content in an in-page tab named after it.

    Parameters
    ----------
    pages : list of page_links.Page
        The pages, in the order their tabs appear.
    title : str
        What the pages are, such as ``"Label Stores"``.
    plural_noun : str
        What one page is per, in the plural, such as ``"sessions"``; counts several pages in the link.

    Returns
    -------
    page_links.PageLink
        The link to the served page.
    """
    if len(pages) == 1:
        (page,) = pages
        return page_links.PageLink(title=f"{title} — {page.name}", href=_serve(template=page.template))
    combined_title = f"{title} — {len(pages)} {plural_noun}"
    template = pages[0].template
    # Only the visible tab renders, so a dozen dashboards do not all draw at once.
    tabs = pn.Tabs(*[(page.name, pn.Column(*page.template.main.objects)) for page in pages], dynamic=True)
    template.main.objects = [tabs]
    template.title = combined_title
    return page_links.PageLink(title=combined_title, href=_serve(template=template))
