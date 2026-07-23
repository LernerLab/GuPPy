import logging

import panel as pn

pn.extension()

logger = logging.getLogger(__name__)

# Sentinel option shown in a control's "control for" dropdown before a signal is chosen.
NO_SIGNAL_OPTION = "— select signal —"


class StoreLabelingConfig:
    """Panel widget that renders a configuration row for each discovered store_id.

    Each row lets the user classify a store_id as ``"control"``, ``"signal"``,
    or ``"event TTLs"``.  ``signal`` and ``event TTLs`` rows provide a free-text
    name box; ``control`` rows instead pick which signal store they are the
    control for (via a dropdown), so the pair name is entered exactly once — on
    the signal row — and a control can never be mismatched to the wrong recording site.

    Selections are stored in-place in the provided ``store_id_dropdowns``,
    ``store_id_textboxes`` and ``store_id_control_refs`` dictionaries so the
    orchestration layer can read them back later.

    Parameters
    ----------
    show_config_button : pn.widgets.Button
        Button appended at the bottom of the configuration panel; the caller
        is responsible for wiring its click handler.
    store_id_dropdowns : dict
        Mutable mapping that will be populated with ``{key: Select}`` (Type).
        Cleared on entry.
    store_id_textboxes : dict
        Mutable mapping that will be populated with ``{key: TextInput}`` (Name,
        used for signal/event rows). Cleared on entry.
    store_id_control_refs : dict
        Mutable mapping populated with ``{key: Select}`` (the "control for"
        signal picker, used for control rows). Cleared on entry.
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
        store_id_control_refs: dict[str, pn.widgets.Select],
        store_ids: list[str],
        store_id_to_store_labels: dict[str, list[str]],
    ) -> None:
        self.config_widgets = []
        self._dropdown_help_map = {}
        self._dropdown_to_key = {}
        # Per control-row widget_key → the signal recording site it was assigned to in a
        # previous session (from cache), so we can re-select it once options exist.
        self._control_desired_recording_site = {}
        # Ordered widget keys, and per-key the store_id it belongs to.
        self._widget_keys = []
        self._widget_key_to_store_id = {}
        store_id_dropdowns.clear()
        store_id_textboxes.clear()
        store_id_control_refs.clear()
        self.store_id_dropdowns = store_id_dropdowns
        self.store_id_textboxes = store_id_textboxes
        self.store_id_control_refs = store_id_control_refs

        if len(store_ids) == 0:
            return

        self.config_widgets.append(
            pn.pane.Markdown(
                "## Label Stores\nSelect appropriate options for each store and provide a label as needed:"
            )
        )

        for i, store_id in enumerate(store_ids):
            self.setup_store_id(
                i, store_id, store_id_dropdowns, store_id_textboxes, store_id_control_refs, store_id_to_store_labels
            )

        # Populate the "control for" dropdowns from the initial signal assignments.
        self._refresh_control_options()

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
        if help_pane is not None:
            help_pane.object = self._get_help_text(dropdown_value=event.new)
        self._apply_row_visibility(event.obj)
        # A row switching to/from "signal" changes the available control targets.
        self._refresh_control_options()

    def _get_help_text(self, dropdown_value: str) -> str:
        if dropdown_value == "control":
            return "*Select the signal this control belongs to*"
        elif dropdown_value == "signal":
            return "*Type appropriate recording-site name*"
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

    def _apply_row_visibility(self, dropdown: pn.widgets.Select) -> None:
        """Show the Name box for signal/event rows and the signal picker for control rows."""
        widget_key = self._dropdown_to_key[dropdown]
        dropdown_value = dropdown.value
        self.store_id_textboxes[widget_key].visible = dropdown_value in ("signal", "event TTLs")
        self.store_id_control_refs[widget_key].visible = dropdown_value == "control"

    def _signal_options(self) -> dict[str, str]:
        """Map ``"<store_id> – <name>"`` → widget_key for every row currently marked signal."""
        options = {}
        for widget_key in self._widget_keys:
            if self.store_id_dropdowns[widget_key].value != "signal":
                continue
            store_id = self._widget_key_to_store_id[widget_key]
            name = self.store_id_textboxes[widget_key].value or "(unnamed)"
            options[f"{store_id} – {name}"] = widget_key
        return options

    def _refresh_control_options(self) -> None:
        """Refresh every control row's signal-picker options from the current signal rows."""
        signal_options = self._signal_options()
        options = {NO_SIGNAL_OPTION: "", **signal_options}
        valid_values = set(options.values())
        for widget_key in self._widget_keys:
            control_ref = self.store_id_control_refs[widget_key]
            previous_value = control_ref.value
            control_ref.options = options
            if previous_value in valid_values and previous_value != "":
                control_ref.value = previous_value
            else:
                # Re-select a cached assignment once its signal recording site becomes available.
                desired_recording_site = self._control_desired_recording_site.get(widget_key)
                restored = ""
                if desired_recording_site is not None:
                    for label, key in signal_options.items():
                        if self.store_id_textboxes[key].value == desired_recording_site:
                            restored = key
                            break
                control_ref.value = restored

    def setup_store_id(
        self,
        i: int,
        store_id: str,
        store_id_dropdowns: dict[str, pn.widgets.Select],
        store_id_textboxes: dict[str, pn.widgets.TextInput],
        store_id_control_refs: dict[str, pn.widgets.Select],
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
            Mutable mapping to which the new ``Select`` (Type) widget is added.
        store_id_textboxes : dict
            Mutable mapping to which the new ``TextInput`` (Name) widget is added.
        store_id_control_refs : dict
            Mutable mapping to which the new ``Select`` (control-for) widget is added.
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
        self._widget_keys.append(widget_key)
        self._widget_key_to_store_id[widget_key] = store_id

        dropdown = pn.widgets.Select(name="Type", value=default_type, options=options, width=150)
        store_id_dropdowns[widget_key] = dropdown
        self._dropdown_to_key[dropdown] = widget_key

        # Name box (used for signal / event TTLs rows)
        textbox = pn.widgets.TextInput(
            name="Name", value=default_name, placeholder="Enter recording-site/event name", width=200
        )
        store_id_textboxes[widget_key] = textbox
        textbox.param.watch(lambda event: self._refresh_control_options(), "value")

        # Signal picker (used for control rows) — options filled by _refresh_control_options
        control_ref = pn.widgets.Select(name="Control for", value="", options={NO_SIGNAL_OPTION: ""}, width=250)
        store_id_control_refs[widget_key] = control_ref
        if default_type == "control":
            self._control_desired_recording_site[widget_key] = default_name

        initial_help_text = self._get_help_text(default_type)
        help_pane = pn.pane.Markdown(
            initial_help_text, styles={"color": "gray", "font-size": "12px"}, height=20, margin=(0, 0, 0, 0)
        )
        self._dropdown_help_map[dropdown] = help_pane
        dropdown.param.watch(self._on_dropdown_value_change, "value")

        # Show the Name box for signal/event rows, the signal picker for control rows.
        textbox.visible = default_type in ("signal", "event TTLs")
        control_ref.visible = default_type == "control"

        # Wrap dropdown in a column with a same-height spacer so it bottom-aligns with the inputs
        row_widgets.append(pn.Column(pn.Spacer(height=20), dropdown))
        row_widgets.append(pn.Column(help_pane, textbox, control_ref))

        # Add the row to config widgets
        self.config_widgets.append(pn.Row(*row_widgets, margin=(5, 0)))
