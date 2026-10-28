import panel as pn
import pytest

from guppy.frontend.sidebar import Sidebar


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
    # The long-running steps share their row with a Show results button, so they are narrower.
    @pytest.mark.parametrize(
        "name, width",
        [
            ("open_label_stores", 300),
            ("read_rawData", 300),
            ("preprocess", 190),
            ("psth_computation", 190),
            ("open_visualization", 300),
            ("open_metadata", 300),
            ("export_nwb", 300),
        ],
    )
    def test_numbered_step_buttons_are_primary(self, sidebar, name, width):
        button = getattr(sidebar, name)
        assert isinstance(button, pn.widgets.Button), f"{name} is not a Button"
        assert button.button_type == "primary", f"{name} does not have primary type"
        assert button.width == width, f"{name} does not have width {width}"

    @pytest.mark.parametrize(
        "name, width", [("import_custom_events", 300), ("select_artifact_windows", 300), ("remove_artifacts", 190)]
    )
    def test_optional_step_buttons_are_default(self, sidebar, name, width):
        """ "default" is this sidebar's vocabulary for an optional, unnumbered step."""
        button = getattr(sidebar, name)
        assert isinstance(button, pn.widgets.Button), f"{name} is not a Button"
        assert button.button_type == "default", f"{name} does not have default type"
        assert button.width == width, f"{name} does not have width {width}"

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
        assert sidebar_position(objects, sidebar.remove_artifacts) is not None
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
            sidebar_position(objects, sidebar.preprocess)
            < sidebar_position(objects, sidebar.select_artifact_windows)
            < sidebar_position(objects, sidebar.remove_artifacts)
            < sidebar_position(objects, sidebar.psth_computation)
        )

    def test_save_input_parameters_button_removed(self, sidebar):
        assert not hasattr(sidebar, "save_button")


class TestShowResults:
    @pytest.mark.parametrize("name", ["preprocess", "remove_artifacts", "psth_computation"])
    def test_starts_disabled(self, sidebar, name):
        button = sidebar.results_buttons[name]
        assert isinstance(button, pn.widgets.Button)
        assert button.name == "Show results"
        assert button.disabled is True

    def test_a_click_calls_what_opens_the_results(self, sidebar):
        calls = []
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("preprocess"))

        sidebar.results_buttons["preprocess"].clicks += 1

        assert sidebar.results_buttons["preprocess"].disabled is False
        assert calls == ["preprocess"]

    def test_a_click_calls_the_latest_opener(self, sidebar):
        calls = []
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("first run"))
        sidebar.show_results(button_name="preprocess", open_results=lambda: calls.append("second run"))

        sidebar.results_buttons["preprocess"].clicks += 1

        assert calls == ["second run"]

    def test_disables_again_without_an_opener(self, sidebar):
        sidebar.show_results(button_name="preprocess", open_results=lambda: None)

        sidebar.show_results(button_name="preprocess", open_results=None)

        assert sidebar.results_buttons["preprocess"].disabled is True

    @pytest.mark.parametrize("name", ["preprocess", "remove_artifacts", "psth_computation"])
    def test_sits_beside_its_step_button(self, sidebar, name):
        sidebar.add_to_template()

        (row,) = [
            item for item in sidebar.template.sidebar if isinstance(item, pn.Row) and getattr(sidebar, name) in item
        ]
        assert list(row) == [getattr(sidebar, name), sidebar.results_buttons[name]]
