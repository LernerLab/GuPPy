import panel as pn
import pytest

from guppy.frontend.sidebar import Sidebar
from guppy.utils.page_links import PageLink


@pytest.fixture
def sidebar(panel_extension):
    """Build a BootstrapTemplate + Sidebar."""
    template = pn.template.BootstrapTemplate(title="Test")
    return Sidebar(template=template)


class TestSidebar:
    @pytest.mark.parametrize(
        "name",
        [
            "open_label_stores",
            "read_rawData",
            "preprocess",
            "psth_computation",
            "open_visualization",
            "open_metadata",
            "export_nwb",
        ],
    )
    def test_numbered_step_buttons_are_primary(self, sidebar, name):
        button = getattr(sidebar, name)
        assert isinstance(button, pn.widgets.Button), f"{name} is not a Button"
        assert button.button_type == "primary", f"{name} does not have primary type"
        assert button.width == 300, f"{name} does not have width 300"

    @pytest.mark.parametrize("name", ["import_custom_events", "select_artifact_windows", "remove_artifacts"])
    def test_optional_step_buttons_are_default(self, sidebar, name):
        """ "default" is this sidebar's vocabulary for an optional, unnumbered step."""
        button = getattr(sidebar, name)
        assert isinstance(button, pn.widgets.Button), f"{name} is not a Button"
        assert button.button_type == "default", f"{name} does not have default type"
        assert button.width == 300, f"{name} does not have width 300"

    def test_preprocess_step_no_longer_claims_to_remove_artifacts(self, sidebar):
        assert sidebar.preprocess.name == "Preprocess"
        assert sidebar.mark_down_preprocess.object == "**Step 3 : Preprocess**"

    def test_progress_bars_initial_values(self, sidebar):
        for bar_name in (
            "read_progress",
            "extract_progress",
            "psth_progress",
            "remove_artifacts_progress",
            "export_progress",
        ):
            bar = getattr(sidebar, bar_name)
            assert bar.value == 100, f"{bar_name} does not start at 100"
            assert bar.max == 100, f"{bar_name} max is not 100"

    def test_attach_callbacks_triggers_function(self, sidebar):
        calls = []

        def on_label_stores(event=None):
            calls.append(event)

        sidebar.attach_callbacks({"open_label_stores": on_label_stores})
        sidebar.open_label_stores.clicks += 1

        assert len(calls) == 1

    def test_add_to_template_populates_sidebar(self, panel_extension):
        template = pn.template.BootstrapTemplate(title="Test")
        sidebar = Sidebar(template=template)
        sidebar.add_to_template()

        objects = template.sidebar.objects
        assert len(objects) > 0
        assert sidebar.open_label_stores in objects
        assert sidebar.read_progress in objects
        assert sidebar.extract_progress in objects
        assert sidebar.psth_progress in objects
        assert sidebar.select_artifact_windows in objects
        assert sidebar.remove_artifacts in objects
        assert sidebar.remove_artifacts_progress in objects
        assert sidebar.open_metadata in objects
        assert sidebar.export_nwb in objects
        assert sidebar.export_progress in objects

    def test_nwb_export_steps_follow_visualization(self, panel_extension):
        """Steps 6 and 7 close the pipeline: Step 5 -> Input Metadata -> Export to NWB."""
        template = pn.template.BootstrapTemplate(title="Test")
        sidebar = Sidebar(template=template)
        sidebar.add_to_template()

        objects = template.sidebar.objects
        assert (
            objects.index(sidebar.open_visualization)
            < objects.index(sidebar.open_metadata)
            < objects.index(sidebar.export_nwb)
        )

    def test_artifact_steps_sit_between_preprocess_and_psth(self, panel_extension):
        """The optional pair is positioned, not free-floating: Step 3 -> mark -> remove -> Step 4."""
        template = pn.template.BootstrapTemplate(title="Test")
        sidebar = Sidebar(template=template)
        sidebar.add_to_template()

        objects = template.sidebar.objects
        assert (
            objects.index(sidebar.preprocess)
            < objects.index(sidebar.select_artifact_windows)
            < objects.index(sidebar.remove_artifacts)
            < objects.index(sidebar.psth_computation)
        )

    def test_save_input_parameters_button_removed(self, sidebar):
        assert not hasattr(sidebar, "save_button")


class TestShowPageLinks:
    def test_shows_one_new_tab_link_per_page(self, sidebar):
        sidebar.show_page_links(
            button_name="open_label_stores",
            links=[
                PageLink(title="Label Stores — session_1", href="http://localhost:5100/"),
                PageLink(title="Label Stores — session_2", href="http://localhost:5101/"),
            ],
        )

        assert [pane.object for pane in sidebar.page_links["open_label_stores"]] == [
            '<a href="http://localhost:5100/" target="_blank">Open Label Stores — session_1</a> (new tab)',
            '<a href="http://localhost:5101/" target="_blank">Open Label Stores — session_2</a> (new tab)',
        ]

    def test_replaces_the_previous_links(self, sidebar):
        sidebar.show_page_links(button_name="preprocess", links=[PageLink(title="Old", href="old")])

        sidebar.show_page_links(button_name="preprocess", links=[])

        assert sidebar.page_links["preprocess"].objects == []

    def test_escapes_the_title_and_address(self, sidebar):
        sidebar.show_page_links(button_name="label_groups", links=[PageLink(title="A & <B>", href="page?a=1&b=2")])

        (pane,) = sidebar.page_links["label_groups"]
        assert pane.object == '<a href="page?a=1&amp;b=2" target="_blank">Open A &amp; &lt;B&gt;</a> (new tab)'

    def test_each_links_column_follows_its_button_in_the_sidebar(self, sidebar):
        sidebar.add_to_template()
        sidebar_objects = list(sidebar.template.sidebar)

        assert sidebar_objects.index(sidebar.page_links["open_label_stores"]) == (
            sidebar_objects.index(sidebar.open_label_stores) + 1
        )
        assert sidebar_objects.index(sidebar.page_links["preprocess"]) == (
            sidebar_objects.index(sidebar.extract_progress) + 1
        )
