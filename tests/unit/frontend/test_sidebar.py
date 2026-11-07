import panel as pn
import pytest

from guppy.frontend.sidebar import STEP_NAMES, Sidebar
from guppy.frontend.step_status import StepStatus


def sidebar_position(objects, widget):
    """Index of the sidebar entry that is ``widget`` or the row holding it."""
    for index, item in enumerate(objects):
        if item is widget or (isinstance(item, pn.Row) and widget in item):
            return index
    return None


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
        assert sidebar.mark_down_preprocess.object == "**Step 3: Preprocess**"

    def test_every_step_has_a_status(self, sidebar):
        assert list(sidebar.statuses) == list(STEP_NAMES)
        assert all(isinstance(status, StepStatus) for status in sidebar.statuses.values())

    def test_only_background_steps_have_a_progress_bar(self, sidebar):
        with_progress = [name for name, status in sidebar.statuses.items() if status.progress is not None]
        assert with_progress == [
            "read_rawData",
            "preprocess",
            "remove_artifacts",
            "psth_computation",
            "group_analysis",
            "export_nwb",
        ]

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
        assert len(objects) == 3 * len(STEP_NAMES)
        for name in STEP_NAMES:
            # Each step is its label, then its button, then its status.
            index = objects.index(getattr(sidebar, name))
            assert objects[index + 1] is sidebar.statuses[name].layout

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
            sidebar_position(objects, sidebar.preprocess)
            < sidebar_position(objects, sidebar.select_artifact_windows)
            < sidebar_position(objects, sidebar.remove_artifacts)
            < sidebar_position(objects, sidebar.psth_computation)
        )

    def test_save_input_parameters_button_removed(self, sidebar):
        assert not hasattr(sidebar, "save_button")


class TestShowResults:
    @pytest.mark.parametrize("name", ["preprocess", "remove_artifacts", "psth_computation"])
    def test_starts_hidden(self, sidebar, name):
        button = sidebar.results_buttons[name]
        assert button is sidebar.statuses[name].results_button
        assert button.name == "View results →"
        assert button.visible is False

    def test_a_click_calls_what_opens_the_results(self, sidebar):
        calls = []
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("preprocess"))

        sidebar.results_buttons["preprocess"].clicks += 1

        assert sidebar.results_buttons["preprocess"].visible is True
        assert calls == ["preprocess"]

    def test_a_click_calls_the_latest_opener(self, sidebar):
        calls = []
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("first run"))
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("second run"))

        sidebar.results_buttons["preprocess"].clicks += 1

        assert calls == ["second run"]

    def test_hides_again_without_an_opener(self, sidebar):
        sidebar.show_results(button_name="preprocess", open_results=lambda: None)

        sidebar.show_results(button_name="preprocess", open_results=None)

        assert sidebar.results_buttons["preprocess"].visible is False
