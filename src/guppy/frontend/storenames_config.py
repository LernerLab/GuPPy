import logging

import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


class StorenamesConfig:
    """Panel widget that renders a configuration row for each discovered storename.

    Each row lets the user classify a storename as ``"control"``, ``"signal"``,
    or ``"event TTLs"`` and enter a region or event name.  Selections are stored
    in-place in the provided ``storename_dropdowns`` and ``storename_textboxes``
    dictionaries so the orchestration layer can read them back later.

    Parameters
    ----------
    show_config_button : pn.widgets.Button
        Button appended at the bottom of the configuration panel; the caller
        is responsible for wiring its click handler.
    storename_dropdowns : dict
        Mutable mapping that will be populated with ``{key: Select}`` entries.
        Cleared on entry.
    storename_textboxes : dict
        Mutable mapping that will be populated with ``{key: TextInput}`` entries.
        Cleared on entry.
    storenames : list of str
        Raw storename strings discovered from the data files.
    storenames_cache : dict
        Previously saved storename assignments, keyed by storename.  Used to
        pre-populate the widgets when the user re-opens the configuration.
    """

    def __init__(
        self,
        show_config_button: pn.widgets.Button,
        storename_dropdowns: dict[str, pn.widgets.Select],
        storename_textboxes: dict[str, pn.widgets.TextInput],
        storenames: list[str],
        storenames_cache: dict[str, list[str]],
    ) -> None:
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

    def _on_dropdown_value_change(self, event: object) -> None:
        help_pane = self._dropdown_help_map.get(event.obj)
        if help_pane is None:
            return
        dropdown_value = event.new
        help_pane.object = self._get_help_text(dropdown_value=dropdown_value)

    def _get_help_text(self, dropdown_value: str) -> str:
        if dropdown_value == "control":
            return "*Type appropriate region name*"
        elif dropdown_value == "signal":
            return "*Type appropriate region name*"
        elif dropdown_value == "event TTLs":
            return "*Type event name for the TTLs*"
        else:
            return ""

    def _parse_cached_value(self, cached_value: str) -> tuple[str, str]:
        # Split an assembled cache entry back into (type, name) for pre-populating widgets
        if cached_value.startswith("control_"):
            return "control", cached_value[len("control_") :]
        elif cached_value.startswith("signal_"):
            return "signal", cached_value[len("signal_") :]
        elif cached_value:
            return "event TTLs", cached_value
        else:
            return "", ""

    def setup_storename(
        self,
        i: int,
        storename: str,
        storename_dropdowns: dict[str, pn.widgets.Select],
        storename_textboxes: dict[str, pn.widgets.TextInput],
        storenames_cache: dict[str, list[str]],
    ) -> None:
        """Build and register a configuration row for a single storename.

        Parameters
        ----------
        i : int
            Zero-based index of this storename in the list; used to create a
            unique widget key.
        storename : str
            Raw storename string (e.g. ``"Dv1A"``).
        storename_dropdowns : dict
            Mutable mapping to which the new ``Select`` widget is added.
        storename_textboxes : dict
            Mutable mapping to which the new ``TextInput`` widget is added.
        storenames_cache : dict
            Previously saved assignments used to pre-populate the widgets.
        """
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

        # Always show textbox and help pane so every row has a uniform layout
        textbox = pn.widgets.TextInput(
            name="Name", value=default_name, placeholder="Enter region/event name", width=200
        )
        storename_textboxes[widget_key] = textbox

        initial_help_text = self._get_help_text(default_type)
        help_pane = pn.pane.Markdown(
            initial_help_text, styles={"color": "gray", "font-size": "12px"}, height=20, margin=(0, 0, 0, 0)
        )
        self._dropdown_help_map[dropdown] = help_pane
        dropdown.param.watch(self._on_dropdown_value_change, "value")
        # Wrap dropdown in a column with a same-height spacer so it bottom-aligns with the textbox
        row_widgets.append(pn.Column(pn.Spacer(height=20), dropdown))
        row_widgets.append(pn.Column(help_pane, textbox))

        # Add the row to config widgets
        self.config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))
