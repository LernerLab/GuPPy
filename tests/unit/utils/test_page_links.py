import panel as pn
import pytest

from guppy.utils import page_links
from guppy.utils.page_links import Page, PageLink


class TestPageLinks:
    def test_collect_gathers_the_links_offered_inside_it(self):
        with page_links.collect() as offers:
            page_links.offer_link(title="GuPPy — Preprocessing", href="preprocess-view?token=a")
            page_links.offer_link(title="GuPPy — Transient peaks", href="transients-view?token=b")

        assert offers.links == [
            PageLink(title="GuPPy — Preprocessing", href="preprocess-view?token=a"),
            PageLink(title="GuPPy — Transient peaks", href="transients-view?token=b"),
        ]
        assert offers.pages == []

    def test_collect_gathers_the_pages_offered_inside_it(self, panel_extension):
        mouse_01 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")
        mouse_02 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_02")

        with page_links.collect() as offers:
            page_links.offer_page(name="mouse_01", template=mouse_01)
            page_links.offer_page(name="mouse_02", template=mouse_02)

        assert offers.pages == [Page(name="mouse_01", template=mouse_01), Page(name="mouse_02", template=mouse_02)]
        assert offers.links == []

    def test_offering_outside_collect_does_nothing(self, panel_extension):
        page_links.offer_link(title="Page", href="page")
        page_links.offer_page(name="mouse_01", template=pn.template.BootstrapTemplate(title="Page"))

        with page_links.collect() as offers:
            pass

        assert (offers.links, offers.pages) == ([], [])

    def test_collect_stops_gathering_after_a_failure_inside_it(self):
        with pytest.raises(ValueError):
            with page_links.collect() as failed_offers:
                raise ValueError("step failed")

        page_links.offer_link(title="Page", href="page")

        assert failed_offers.links == []
