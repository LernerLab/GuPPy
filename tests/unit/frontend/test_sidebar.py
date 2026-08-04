import panel as pn
import pytest

from guppy.frontend.sidebar import Sidebar


@pytest.fixture
def sidebar(panel_extension):
    """Build a BootstrapTemplate + Sidebar."""
    template = pn.template.BootstrapTemplate(title="Test")
    return Sidebar(template=template)


class TestSidebar:
    @pytest.mark.parametrize(
        "name",
        ["open_label_stores", "read_rawData", "preprocess", "psth_computation", "open_visualization"],
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
        for bar_name in ("read_progress", "extract_progress", "psth_progress", "remove_artifacts_progress"):
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
