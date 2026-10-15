"""Channel through which pipeline steps offer the pages they produce.

Modelled on :mod:`guppy.utils.progress`: deep pipeline code offers a page by calling
:func:`offer_link` or :func:`offer_page`, with nothing threaded through its signature, and the
homepage decides where the offered pages go. GuPPy never opens a browser tab itself; the homepage
shows each offered page as a link the user clicks, and serves the finished pages one click
produced together, as one page. Offering when nothing is collecting is a no-op, so the workers can
be driven headlessly (``guppy.testing.api``) without side effects.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import panel as pn


@dataclass(frozen=True)
class PageLink:
    """A page a step produced, and where the browser finds it."""

    title: str
    href: str


@dataclass(frozen=True)
class Page:
    """A finished page a step built, not yet served."""

    name: str
    template: pn.template.BaseTemplate


@dataclass
class Offers:
    """Everything offered while :func:`collect` was active."""

    links: list[PageLink] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)


_current_offers: ContextVar[Offers | None] = ContextVar("guppy_page_offers", default=None)


def offer_link(*, title: str, href: str) -> None:
    """Offer a page that is already served, as a link.

    Parameters
    ----------
    title : str
        Name of the page, shown in the link.
    href : str
        Where the browser finds the page, either absolute or relative to the homepage.
    """
    offers = _current_offers.get()
    if offers is not None:
        offers.links.append(PageLink(title=title, href=href))


def offer_page(*, name: str, template: pn.template.BaseTemplate) -> None:
    """Offer a finished page to be served along with the others from the same click.

    Parameters
    ----------
    name : str
        What distinguishes this page from the others, such as its session; names its tab.
    template : pn.template.BaseTemplate
        The page.
    """
    offers = _current_offers.get()
    if offers is not None:
        offers.pages.append(Page(name=name, template=template))


@contextmanager
def collect() -> Iterator[Offers]:
    """Collect everything offered inside the ``with`` block into the yielded :class:`Offers`."""
    offers = Offers()
    token = _current_offers.set(offers)
    try:
        yield offers
    finally:
        _current_offers.reset(token)
