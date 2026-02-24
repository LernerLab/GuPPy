import logging

import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


class StorenamesConfig:
    def __init__(
        self,
        show_config_button,
        storename_dropdowns,
        storename_textboxes,
        storenames,
        storenames_cache,
    ):
        self.config_widgets = []
        self._dropdown_help_map = {}
        storename_dropdowns.clear()
        storename_textboxes.clear()

        if len(storenames) == 0:
            return

        self.config_widgets.append(
            pn.pane.Markdown(
                "## Configure Storenames\nSelect appropriate options for each storename and provide names as needed:"
            )
        )

        for i, storename in enumerate(storenames):
            self.setup_storename(i, storename, storename_dropdowns, storename_textboxes, storenames_cache)

        # Add show button
        self.config_widgets.append(pn.Spacer(height=20))
        self.config_widgets.append(show_config_button)
        self.config_widgets.append(
            pn.pane.Markdown(
                "*Click 'Show Selected Configuration' to apply your selections.*",
                styles={"font-size": "12px", "color": "gray"},
            )
        )

    def _on_dropdown_value_change(self, event):
        help_pane = self._dropdown_help_map.get(event.obj)
        if help_pane is None:
            return
        dropdown_value = event.new
        help_pane.object = self._get_help_text(dropdown_value=dropdown_value)

    def _get_help_text(self, dropdown_value):
        if dropdown_value == "control":
            return "*Type appropriate region name*"
        elif dropdown_value == "signal":
            return "*Type appropriate region name*"
        elif dropdown_value == "event TTLs":
            return "*Type event name for the TTLs*"
        else:
            return ""

    def _parse_cached_value(self, cached_value):
        # Split an assembled cache entry back into (type, name) for pre-populating widgets
        if cached_value.startswith("control_"):
            return "control", cached_value[len("control_") :]
        elif cached_value.startswith("signal_"):
            return "signal", cached_value[len("signal_") :]
        elif cached_value:
            return "event TTLs", cached_value
        else:
            return "", ""

    def setup_storename(self, i, storename, storename_dropdowns, storename_textboxes, storenames_cache):
        # Create a row for each storename
        row_widgets = []

        # Label
        label = pn.pane.Markdown(f"**{storename}:**")
        row_widgets.append(label)

        # Pre-populate from cache if available; otherwise start blank
        options = ["", "control", "signal", "event TTLs"]
        if storename in storenames_cache and storenames_cache[storename]:
            default_type, default_name = self._parse_cached_value(storenames_cache[storename][0])
        else:
            default_type, default_name = "", ""

        # Create unique key for widget
        widget_key = (
            f"{storename}_{i}"
            if f"{storename}_{i}" not in storename_dropdowns
            else f"{storename}_{i}_{len(storename_dropdowns)}"
        )

        dropdown = pn.widgets.Select(name="Type", value=default_type, options=options, width=150)
        storename_dropdowns[widget_key] = dropdown
        row_widgets.append(dropdown)

        # Always show textbox and help pane so every row has a uniform layout
        textbox = pn.widgets.TextInput(
            name="Name", value=default_name, placeholder="Enter region/event name", width=200
        )
        storename_textboxes[widget_key] = textbox
        row_widgets.append(textbox)

        initial_help_text = self._get_help_text(default_type)
        help_pane = pn.pane.Markdown(initial_help_text, styles={"color": "gray", "font-size": "12px"})
        self._dropdown_help_map[dropdown] = help_pane
        dropdown.param.watch(self._on_dropdown_value_change, "value")
        row_widgets.append(help_pane)

        # Add the row to config widgets
        self.config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))
