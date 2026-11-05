import panel as pn
import pytest

from guppy.utils import page_links
from guppy.utils.page_links import Page


class TestPageLinks:
    def test_collect_gathers_the_pages_offered_inside_it(self, panel_extension):
        preprocessing = pn.pane.Markdown("Preprocessing")
        transients = pn.pane.Markdown("Transient peaks")

        with page_links.collect() as pages:
            page_links.offer_page(name="Preprocessing", content=preprocessing)
            page_links.offer_page(name="Transient peaks", content=transients)

        assert pages == [
            Page(name="Preprocessing", content=preprocessing),
            Page(name="Transient peaks", content=transients),
        ]

    def test_offer_template_offers_the_templates_main_area(self, panel_extension):
        heading = pn.pane.Markdown("Label Stores")
        table = pn.pane.Markdown("Stores")
        template = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")
        template.main.extend([heading, table])

        with page_links.collect() as pages:
            page_links.offer_template(name="mouse_01", template=template)

        (page,) = pages
        assert page.name == "mouse_01"
        assert isinstance(page.content, pn.Column)
        assert list(page.content) == [heading, table]

    def test_offering_outside_collect_does_nothing(self, panel_extension):
        page_links.offer_page(name="Preprocessing", content=pn.pane.Markdown("Preprocessing"))
        page_links.offer_template(name="mouse_01", template=pn.template.BootstrapTemplate(title="Page"))

        with page_links.collect() as pages:
            pass

        assert pages == []

    def test_collect_stops_gathering_after_a_failure_inside_it(self, panel_extension):
        with pytest.raises(ValueError):
            with page_links.collect() as failed_pages:
                raise ValueError("step failed")

        page_links.offer_page(name="Preprocessing", content=pn.pane.Markdown("Preprocessing"))

        assert failed_pages == []
