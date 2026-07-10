import logging

import panel as pn

pn.extension()

logger = logging.getLogger(__name__)


class StoreLabelingConfig:
    """Panel widget that renders a configuration row for each discovered store_id.

    Each row lets the user classify a store_id as ``"control"``, ``"signal"``,
    or ``"event TTLs"`` and enter a region or event name.  Selections are stored
    in-place in the provided ``store_id_dropdowns`` and ``store_id_textboxes``
    dictionaries so the orchestration layer can read them back later.

    Parameters
    ----------
    show_config_button : pn.widgets.Button
        Button appended at the bottom of the configuration panel; the caller
        is responsible for wiring its click handler.
    store_id_dropdowns : dict
        Mutable mapping that will be populated with ``{key: Select}`` entries.
        Cleared on entry.
    store_id_textboxes : dict
        Mutable mapping that will be populated with ``{key: TextInput}`` entries.
        Cleared on entry.
    store_ids : list of str
        Raw store_id strings discovered from the data files.
    store_id_to_store_labels : dict
        Previously saved store_id assignments, keyed by store_id.  Used to
        pre-populate the widgets when the user re-opens the configuration.
    """

    def __init__(
        self,
        show_config_button: pn.widgets.Button,
        store_id_dropdowns: dict[str, pn.widgets.Select],
        store_id_textboxes: dict[str, pn.widgets.TextInput],
        store_ids: list[str],
        store_id_to_store_labels: dict[str, list[str]],
    ) -> None:
        self.config_widgets = []
        self._dropdown_help_map = {}
        store_id_dropdowns.clear()
        store_id_textboxes.clear()

        if len(store_ids) == 0:
            return

        self.config_widgets.append(
            pn.pane.Markdown(
                "## Label Stores\nSelect appropriate options for each store and provide a label as needed:"
            )
        )

        for i, store_id in enumerate(store_ids):
            self.setup_store_id(i, store_id, store_id_dropdowns, store_id_textboxes, store_id_to_store_labels)

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

    def setup_store_id(
        self,
        i: int,
        store_id: str,
        store_id_dropdowns: dict[str, pn.widgets.Select],
        store_id_textboxes: dict[str, pn.widgets.TextInput],
        store_id_to_store_labels: dict[str, list[str]],
    ) -> None:
        """Build and register a configuration row for a single store_id.

        Parameters
        ----------
        i : int
            Zero-based index of this store_id in the list; used to create a
            unique widget key.
        store_id : str
            Raw store_id string (e.g. ``"Dv1A"``).
        store_id_dropdowns : dict
            Mutable mapping to which the new ``Select`` widget is added.
        store_id_textboxes : dict
            Mutable mapping to which the new ``TextInput`` widget is added.
        store_id_to_store_labels : dict
            Previously saved assignments used to pre-populate the widgets.
        """
        # Create a row for each store_id
        row_widgets = []

        # Label
        label = pn.pane.Markdown(f"**{store_id}:**")
        row_widgets.append(label)

        # Pre-populate from cache if available; otherwise start blank
        options = ["", "control", "signal", "event TTLs"]
        if store_id in store_id_to_store_labels and store_id_to_store_labels[store_id]:
            default_type, default_name = self._parse_cached_value(store_id_to_store_labels[store_id][0])
        else:
            default_type, default_name = "", ""

        # Create unique key for widget
        widget_key = (
            f"{store_id}_{i}"
            if f"{store_id}_{i}" not in store_id_dropdowns
            else f"{store_id}_{i}_{len(store_id_dropdowns)}"
        )

        dropdown = pn.widgets.Select(name="Type", value=default_type, options=options, width=150)
        store_id_dropdowns[widget_key] = dropdown

        # Always show textbox and help pane so every row has a uniform layout
        textbox = pn.widgets.TextInput(
            name="Name", value=default_name, placeholder="Enter region/event name", width=200
        )
        store_id_textboxes[widget_key] = textbox

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
