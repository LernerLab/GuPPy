"""Channel through which pipeline steps offer the pages they produce.

Modelled on :mod:`guppy.utils.progress`: deep pipeline code offers a page by calling
:func:`offer_page` or :func:`offer_template`, with nothing threaded through its signature, and the
homepage decides where the offered pages go. The homepage shows them inside its own browser tab.
Offering when nothing is collecting is a no-op, so the workers can be driven headlessly
(``guppy.testing.api``) without side effects.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import panel as pn


@dataclass(frozen=True)
class Page:
    """A page a step produced: its name and its content."""

    name: str
    content: pn.viewable.Viewable


_current_pages: ContextVar[list[Page] | None] = ContextVar("guppy_page_offers", default=None)


def offer_page(*, name: str, content: pn.viewable.Viewable) -> None:
    """Offer a page to be shown along with the others from the same click.

    Parameters
    ----------
    name : str
        What distinguishes this page from the others, such as its session; names its tab.
    content : pn.viewable.Viewable
        The page.
    """
    pages = _current_pages.get()
    if pages is not None:
        pages.append(Page(name=name, content=content))


def offer_template(*, name: str, template: pn.template.BaseTemplate) -> None:
    """Offer the main area of a page built as a template.

    Parameters
    ----------
    name : str
        What distinguishes this page from the others, such as its session; names its tab.
    template : pn.template.BaseTemplate
        The page, whose ``main`` area is shown.
    """
    offer_page(name=name, content=pn.Column(*template.main.objects, sizing_mode="stretch_width"))


@contextmanager
def collect() -> Iterator[list[Page]]:
    """Collect every page offered inside the ``with`` block into the yielded list."""
    pages: list[Page] = []
    token = _current_pages.set(pages)
    try:
        yield pages
    finally:
        _current_pages.reset(token)
