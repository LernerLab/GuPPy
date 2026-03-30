import panel as pn

from guppy.frontend.sidebar import Sidebar


class TestSidebar:
    def test_all_buttons_exist_with_primary_type(self, sidebar):
        button_names = [
            "save_button",
            "open_storenames",
            "read_rawData",
            "preprocess",
            "psth_computation",
            "open_visualization",
        ]
        for name in button_names:
            button = getattr(sidebar, name)
            assert isinstance(button, pn.widgets.Button), f"{name} is not a Button"
            assert button.button_type == "primary", f"{name} does not have primary type"
            assert button.width == 300, f"{name} does not have width 300"

    def test_progress_bars_initial_values(self, sidebar):
        for bar_name in ("read_progress", "extract_progress", "psth_progress"):
            bar = getattr(sidebar, bar_name)
            assert bar.value == 100, f"{bar_name} does not start at 100"
            assert bar.max == 100, f"{bar_name} max is not 100"

    def test_attach_callbacks_triggers_function(self, sidebar):
        calls = []

        def on_save(event=None):
            calls.append(event)

        sidebar.attach_callbacks({"save_button": on_save})
        sidebar.save_button.clicks += 1

        assert len(calls) == 1

    def test_add_to_template_populates_sidebar(self, panel_extension):
        template = pn.template.BootstrapTemplate(title="Test")
        sidebar = Sidebar(template=template)
        sidebar.add_to_template()

        objects = template.sidebar.objects
        assert len(objects) > 0
        assert sidebar.save_button in objects
        assert sidebar.read_progress in objects
        assert sidebar.extract_progress in objects
        assert sidebar.psth_progress in objects
