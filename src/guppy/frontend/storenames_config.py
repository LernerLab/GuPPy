import logging

import panel as pn

from .storenames_selector import StorenamesSelector

pn.extension()

logger = logging.getLogger(__name__)


class StorenamesConfig:
    def __init__(
        self,
        storenames_selector: StorenamesSelector,
        storename_dropdowns,
        storename_textboxes,
        storenames,
        storenames_cache,
    ):
        self.config_widgets = []
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
        self.config_widgets.append(storenames_selector.show_config_button)
        self.config_widgets.append(
            pn.pane.Markdown(
                "*Click 'Show Selected Configuration' to apply your selections.*",
                styles={"font-size": "12px", "color": "gray"},
            )
        )

    def setup_storename(self, i, storename, storename_dropdowns, storename_textboxes, storenames_cache):
        # Create a row for each storename
        row_widgets = []

        # Label
        label = pn.pane.Markdown(f"**{storename}:**")
        row_widgets.append(label)

        # Dropdown options
        if storename in storenames_cache:
            options = storenames_cache[storename]
            default_value = options[0] if options else ""
        else:
            options = ["", "control", "signal", "event TTLs"]
            default_value = ""

        # Create unique key for widget
        widget_key = (
            f"{storename}_{i}"
            if f"{storename}_{i}" not in storename_dropdowns
            else f"{storename}_{i}_{len(storename_dropdowns)}"
        )

        dropdown = pn.widgets.Select(name="Type", value=default_value, options=options, width=150)
        storename_dropdowns[widget_key] = dropdown
        row_widgets.append(dropdown)

        # Text input (only show if not cached or if control/signal/event TTLs selected)
        if storename not in storenames_cache or default_value in ["control", "signal", "event TTLs"]:
            textbox = pn.widgets.TextInput(name="Name", value="", placeholder="Enter region/event name", width=200)
            storename_textboxes[widget_key] = textbox
            row_widgets.append(textbox)

            # Add helper text based on selection
            def create_help_function(dropdown_widget, help_pane_container):
                @pn.depends(dropdown_widget.param.value, watch=True)
                def update_help(dropdown_value):
                    if dropdown_value == "control":
                        help_pane_container[0] = pn.pane.Markdown(
                            "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                        )
                    elif dropdown_value == "signal":
                        help_pane_container[0] = pn.pane.Markdown(
                            "*Type appropriate region name*", styles={"color": "gray", "font-size": "12px"}
                        )
                    elif dropdown_value == "event TTLs":
                        help_pane_container[0] = pn.pane.Markdown(
                            "*Type event name for the TTLs*", styles={"color": "gray", "font-size": "12px"}
                        )
                    else:
                        help_pane_container[0] = pn.pane.Markdown("", styles={"color": "gray", "font-size": "12px"})

                return update_help

            help_container = [pn.pane.Markdown("")]
            help_function = create_help_function(dropdown, help_container)
            help_function(dropdown.value)  # Initialize
            row_widgets.append(help_container[0])

        # Add the row to config widgets
        self.config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))
