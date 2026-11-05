import panel as pn
import pytest

from guppy.frontend.page_area import PageArea


@pytest.fixture
def form(panel_extension):
    return pn.pane.Markdown("Input parameters form")


@pytest.fixture
def page_area(form):
    return PageArea(home=form)


class TestPageArea:
    def test_first_tab_is_the_form(self, page_area, form):
        assert page_area.tabs._names == ["Input Parameters"]
        assert page_area.tabs[0][0] is form
        assert page_area.tabs.active == 0

    def test_show_appends_and_activates_a_new_tab(self, page_area):
        preprocessing = pn.pane.Markdown("Preprocessing")

        page_area.show(key="preprocess", name="Preprocessing", content=preprocessing)

        assert page_area.tabs._names == ["Input Parameters", "Preprocessing"]
        assert page_area.tabs[1][0] is preprocessing
        assert page_area.tabs.active == 1

    def test_showing_the_same_key_again_replaces_its_tab_in_place(self, page_area):
        page_area.show(key="preprocess", name="Preprocessing", content=pn.pane.Markdown("first run"))
        page_area.show(key="open_label_stores", name="Label Stores — mouse_01", content=pn.pane.Markdown("labels"))
        page_area.tabs.active = 0
        second_run = pn.pane.Markdown("second run")

        page_area.show(key="preprocess", name="Preprocessing (rerun)", content=second_run)

        assert page_area.tabs._names == ["Input Parameters", "Preprocessing (rerun)", "Label Stores — mouse_01"]
        assert page_area.tabs[1][0] is second_run
        assert page_area.tabs.active == 1

    def test_showing_a_key_whose_tab_was_closed_appends_a_new_tab(self, page_area):
        page_area.show(key="preprocess", name="Preprocessing", content=pn.pane.Markdown("first run"))
        page_area.show(key="open_label_stores", name="Label Stores — mouse_01", content=pn.pane.Markdown("labels"))
        # Closing a tab in the browser removes it from the Tabs' objects.
        page_area.tabs.pop(1)
        second_run = pn.pane.Markdown("second run")

        page_area.show(key="preprocess", name="Preprocessing", content=second_run)

        assert page_area.tabs._names == ["Input Parameters", "Label Stores — mouse_01", "Preprocessing"]
        assert page_area.tabs[2][0] is second_run
        assert page_area.tabs.active == 2
