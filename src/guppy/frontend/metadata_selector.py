"""Panel widget for editing one session's NWB metadata (channel-centric, schema-driven).

GuPPy fixes the fiber-photometry channels (rows) from ``storesList.csv``; the user
annotates each channel with wavelengths and device links (dropdowns), and defines a
device library once. Field labels, required markers, and help text come from the
installed ndx specs via :func:`guppy.utils.nwb_metadata.field_specs`. Structured
widgets feed a YAML ``CodeEditor`` that is the authoritative serialized artifact.
"""

import logging

import panel as pn

from ..utils.nwb_metadata import (
    CATEGORIES,
    CHANNEL_LINKS,
    CHANNEL_REQUIRED_LINKS,
    Channel,
    FieldSpec,
    build_metadata_dict,
    dumps_yaml,
    field_specs,
    loads_yaml,
    parse_metadata_dict,
)

logger = logging.getLogger(__name__)

_SEX_OPTIONS = ["", "M", "F", "U", "O"]
# Scalar NWBFile/Subject fields that are genuinely required (identifier auto-generates if blank).
_REQUIRED_SCALARS = {"session_description", "subject_id", "sex", "species"}
# Help text shown as the little "?" tooltip on each session/subject field.
_SCALAR_DOCS = {
    "session_description": "A description of the experimental session.",
    "identifier": "A unique identifier for the session. Auto-generated if left blank.",
    "lab": "Name of the lab that produced this data.",
    "institution": "Institution where the experiment was performed.",
    "subject_id": "Unique identifier for the subject (lab convention).",
    "sex": "Sex of the subject: M, F, U (unknown) or O (other).",
    "species": "Latin binomial species name, e.g. 'Mus musculus'.",
    "genotype": "Genetic strain. If absent, assume wild type (WT).",
    "strain": "The strain of the subject, e.g. 'C57BL/6J'.",
}


def _required_marker(required: bool) -> str:
    return " *" if required else ""


class MetadataSelector:
    """Editor for one session's complete NWB metadata overlay."""

    def __init__(self, session_label: str, channels: list[Channel], initial_metadata: dict) -> None:
        self.session_label = session_label
        self.channels = channels
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="success", sizing_mode="stretch_width")

        devices, channel_rows, scalars = parse_metadata_dict(initial_metadata, channels)

        # Tracking structures populated as widgets are built.
        self.device_entry_records: dict[str, list[dict]] = {key: [] for key in CATEGORIES}
        self.device_containers: dict[str, pn.Column] = {
            key: pn.Column(sizing_mode="stretch_width") for key in CATEGORIES
        }
        self.channel_records: list[dict] = []
        # Link dropdowns whose options are device names; channel ones are fixed, device ones rebuild.
        # Each entry is (select, target_category, required); required links carry no empty option.
        self.channel_link_selects: list[tuple[pn.widgets.Select, str, bool]] = []
        self.device_link_selects: list[tuple[pn.widgets.Select, str, bool]] = []

        scalar_card = self._build_scalar_section(scalars)
        library_section = self._build_device_library(devices)
        channel_card = self._build_channel_table(channel_rows)

        self.code_editor = pn.widgets.CodeEditor(
            value="", theme="tomorrow", language="yaml", height=400, sizing_mode="stretch_width"
        )
        self.load_existing = pn.widgets.FileInput(accept=".yaml,.yml", sizing_mode="stretch_width")
        self.load_existing.param.watch(self._on_file_upload, "value")
        self.build_config = pn.widgets.Button(
            name="Show / Refresh YAML from form above", button_type="primary", sizing_mode="stretch_width"
        )
        self.save = pn.widgets.Button(name="Save Metadata", button_type="success", sizing_mode="stretch_width")
        self.path = pn.widgets.TextInput(name="Saved to", disabled=True, sizing_mode="stretch_width")

        self.widget = pn.Column(
            pn.pane.Markdown(f"## NWB Metadata — {session_label}"),
            pn.pane.Markdown("Load metadata from any existing `nwb_metadata.yaml` on disk to pre-populate this form:"),
            self.load_existing,
            scalar_card,
            pn.pane.Markdown("### Device library (define each device once, then link it from the channels below)"),
            library_section,
            channel_card,
            pn.layout.Divider(),
            self.build_config,
            pn.pane.Markdown("Edit the YAML below directly if the form can't express something, then Save."),
            self.code_editor,
            self.alert,
            self.save,
            self.path,
            sizing_mode="stretch_width",
            max_width=1100,
        )

        self.refresh_link_options()
        self.set_yaml(build_metadata_dict(self.get_devices(), self.get_channel_rows(), self.get_scalars(), channels))

    # ------------------------------------------------------------------------------------------------------------------
    # Session & Subject
    # ------------------------------------------------------------------------------------------------------------------
    def _build_scalar_section(self, scalars: dict) -> pn.Card:
        text = lambda name, **kw: pn.widgets.TextInput(  # noqa: E731
            name=f"{name}{_required_marker(name in _REQUIRED_SCALARS)}",
            value=scalars.get(name, ""),
            description=_SCALAR_DOCS.get(name),
            sizing_mode="stretch_width",
            **kw,
        )
        self.session_description = pn.widgets.TextAreaInput(
            name="session_description *",
            value=scalars.get("session_description", ""),
            description=_SCALAR_DOCS["session_description"],
            sizing_mode="stretch_width",
            height=70,
        )
        self.identifier = text("identifier", placeholder="leave blank to auto-generate")
        self.lab = text("lab")
        self.institution = text("institution")

        # Experimenter repeater.
        self.experimenter_box = pn.Column(sizing_mode="stretch_width")
        self.experimenter_inputs: list[pn.widgets.TextInput] = []
        add_experimenter = pn.widgets.Button(name="➕ Add experimenter", button_type="default", width=180)
        add_experimenter.on_click(lambda event: self._add_experimenter(""))
        for name in scalars.get("experimenter") or [""]:
            self._add_experimenter(name)

        self.subject_id = text("subject_id")
        self.sex = pn.widgets.Select(
            name="sex *",
            value=scalars.get("sex", "") or "",
            options=_SEX_OPTIONS,
            description=_SCALAR_DOCS["sex"],
            width=120,
        )
        self.species = text("species", placeholder="e.g. Mus musculus")
        self.genotype = text("genotype")
        self.strain = text("strain")

        # Age XOR date of birth.
        self.age = pn.widgets.TextInput(name="age (ISO 8601, e.g. P90D)", value=scalars.get("age", ""), width=260)
        self.date_of_birth = pn.widgets.TextInput(
            name="date_of_birth (ISO 8601)", value=scalars.get("date_of_birth", ""), width=260
        )
        self.age_or_dob = pn.widgets.RadioBoxGroup(
            name="Specify subject by",
            options=["Age", "Date of birth"],
            value="Date of birth" if scalars.get("date_of_birth") else "Age",
            inline=True,
        )
        self.age_or_dob.param.watch(lambda event: self._sync_age_dob(), "value")
        self._sync_age_dob()

        return pn.Card(
            self.session_description,
            pn.Row(self.identifier, self.lab, self.institution, sizing_mode="stretch_width"),
            pn.pane.Markdown("**Experimenter**"),
            self.experimenter_box,
            add_experimenter,
            pn.layout.Divider(),
            pn.pane.Markdown("**Subject**"),
            pn.Row(self.subject_id, self.sex, self.species, sizing_mode="stretch_width"),
            pn.Row(self.genotype, self.strain, sizing_mode="stretch_width"),
            self.age_or_dob,
            pn.Row(self.age, self.date_of_birth),
            title="Session & Subject",
            collapsed=False,
            sizing_mode="stretch_width",
        )

    def _add_experimenter(self, value: str) -> None:
        row = pn.Row(sizing_mode="stretch_width")
        text_input = pn.widgets.TextInput(value=value, placeholder="Last, First", sizing_mode="stretch_width")
        remove = pn.widgets.Button(name="✕", button_type="default", width=40)
        remove.on_click(lambda event: self._remove_experimenter(row, text_input))
        row[:] = [text_input, remove]
        self.experimenter_inputs.append(text_input)
        self.experimenter_box.append(row)

    def _remove_experimenter(self, row: pn.Row, text_input: pn.widgets.TextInput) -> None:
        if text_input in self.experimenter_inputs:
            self.experimenter_inputs.remove(text_input)
        if row in self.experimenter_box:
            self.experimenter_box.remove(row)

    def _sync_age_dob(self) -> None:
        by_age = self.age_or_dob.value == "Age"
        self.age.visible = by_age
        self.date_of_birth.visible = not by_age

    # ------------------------------------------------------------------------------------------------------------------
    # Device library
    # ------------------------------------------------------------------------------------------------------------------
    def _build_device_library(self, devices: dict[str, list[dict]]) -> pn.Column:
        cards = []
        for key, category in CATEGORIES.items():
            add_button = pn.widgets.Button(name=f"➕ Add {category.singular}", width=240)
            add_button.on_click(lambda event, k=key: (self._add_device(k, {}), self.refresh_link_options()))
            for entry in devices.get(key, []):
                self._add_device(key, entry)
            cards.append(
                pn.Card(
                    self.device_containers[key],
                    add_button,
                    title=category.label,
                    collapsed=True,
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(*cards, sizing_mode="stretch_width")

    def _add_device(self, category_key: str, entry: dict) -> None:
        fields: list[tuple[FieldSpec, object]] = []  # (spec, widget or list-row)
        rows = []
        insertion_header_added = False
        name_widget = None
        for spec in field_specs(category_key):
            if spec.target == "insertion" and not insertion_header_added:
                rows.append(pn.pane.Markdown("**Fiber insertion**"))
                insertion_header_added = True
            widget = self._make_field_widget(spec, entry)
            fields.append((spec, widget))
            rows.append(widget)
            if spec.name == "name":
                name_widget = widget

        remove = pn.widgets.Button(name="🗑 Remove", button_type="default", width=110)
        # Each instance is its own collapsible card, titled by its name so a category with many is navigable.
        card = pn.Card(
            *rows,
            remove,
            title=(name_widget.value or "(unnamed)"),
            collapsed=True,
            sizing_mode="stretch_width",
        )
        if name_widget is not None:

            def _on_name_change(event: object, card: pn.Card = card) -> None:
                # Retitle the card and propagate the new name into every link dropdown that points here.
                card.title = event.new or "(unnamed)"
                self.refresh_link_options()

            name_widget.param.watch(_on_name_change, "value")
        record = {"container": card, "fields": fields}
        remove.on_click(lambda event: self._remove_device(category_key, record))
        self.device_entry_records[category_key].append(record)
        self.device_containers[category_key].append(card)

    def _remove_device(self, category_key: str, record: dict) -> None:
        if record in self.device_entry_records[category_key]:
            self.device_entry_records[category_key].remove(record)
        if record["container"] in self.device_containers[category_key]:
            self.device_containers[category_key].remove(record["container"])
        # Drop the removed device from every link dropdown that pointed at it.
        self.refresh_link_options()

    def _make_field_widget(self, spec: FieldSpec, entry: dict) -> pn.viewable.Viewable:
        label = f"{spec.name}{_required_marker(spec.required)}"
        value = entry.get(spec.name)
        if spec.link_target:
            select = pn.widgets.Select(
                name=label, options=[""], value=(value or ""), description=spec.doc, sizing_mode="stretch_width"
            )
            self.device_link_selects.append((select, spec.link_target, spec.required))
            return select
        if spec.options:
            # Enumerated value -> dropdown. Required enums carry no empty option (default to the first
            # choice); optional ones keep a blank. Any unknown current value is preserved as a choice.
            options = list(spec.options)
            current = "" if value is None else str(value)
            if spec.required:
                choices = options if current in options else [*options, current] if current else options
                chosen = current if current in options else (current or options[0])
            else:
                choices = ["", *options] if current in ("", *options) else ["", *options, current]
                chosen = current
            return pn.widgets.Select(
                name=label, options=choices, value=chosen, description=spec.doc, sizing_mode="stretch_width"
            )
        if spec.is_list:
            # Spec shape [2] is a numeric (min, max) pair: enforce float entry with FloatInput.
            low = pn.widgets.FloatInput(name=f"{label} [min]", value=_as_float(value, 0), width=160)
            high = pn.widgets.FloatInput(name=f"{label} [max]", value=_as_float(value, 1), width=160)
            row = pn.Row(low, high)
            row._field_pair = (low, high)  # noqa: SLF001
            return row
        if spec.dtype == "float":
            # Numeric attribute (e.g. numerical_aperture): a FloatInput rejects non-numeric text up front.
            return pn.widgets.FloatInput(
                name=label, value=_as_float(value), description=spec.doc, sizing_mode="stretch_width"
            )
        return pn.widgets.TextInput(
            name=label,
            value=("" if value is None else str(value)),
            description=spec.doc,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Channel table (fixed rows)
    # ------------------------------------------------------------------------------------------------------------------
    def _build_channel_table(self, channel_rows: list[dict]) -> pn.Card:
        header = pn.pane.Markdown(
            "Rows are fixed by `storesList.csv`. Fill the excitation/emission wavelengths and link each channel "
            "to the devices defined above."
        )
        rows = [header]
        for index, channel in enumerate(self.channels):
            saved = channel_rows[index] if index < len(channel_rows) else {}
            fields: dict[str, object] = {}
            excitation = pn.widgets.FloatInput(
                name="excitation_wavelength_in_nm *",
                value=_as_float(saved.get("excitation_wavelength_in_nm")),
                width=180,
            )
            emission = pn.widgets.FloatInput(
                name="emission_wavelength_in_nm *", value=_as_float(saved.get("emission_wavelength_in_nm")), width=180
            )
            fields["excitation_wavelength_in_nm"] = excitation
            fields["emission_wavelength_in_nm"] = emission
            link_widgets = []
            for link_name, target in CHANNEL_LINKS.items():
                marker = _required_marker(link_name in CHANNEL_REQUIRED_LINKS)
                select = pn.widgets.Select(
                    name=f"{link_name}{marker}", options=[""], value=(saved.get(link_name) or ""), width=180
                )
                self.channel_link_selects.append((select, target, link_name in CHANNEL_REQUIRED_LINKS))
                fields[link_name] = select
                link_widgets.append(select)
            self.channel_records.append({"fields": fields})
            rows.append(
                pn.Column(
                    pn.pane.Markdown(f"**{channel.region} — {channel.role}**  (`{channel.store_name}`)"),
                    pn.Row(excitation, emission, *link_widgets[:2]),
                    pn.Row(*link_widgets[2:]),
                    pn.layout.Divider(),
                    sizing_mode="stretch_width",
                )
            )
        return pn.Card(*rows, title="Fiber-photometry channels", collapsed=False, sizing_mode="stretch_width")

    # ------------------------------------------------------------------------------------------------------------------
    # Read form state
    # ------------------------------------------------------------------------------------------------------------------
    def refresh_link_options(self) -> None:
        """Repopulate every link dropdown from the currently-defined device names.

        Required links carry no empty option (defaulting to the first available
        device) so a channel/instance can never be silently left unlinked; if no
        device of that category exists yet, the only option is empty and the
        Build/Save validation reports it as missing.
        """
        names_by_category = {
            key: [name for name in (self._entry_name(record) for record in records) if name]
            for key, records in self.device_entry_records.items()
        }
        for select, target_category, required in (*self.channel_link_selects, *self.device_link_selects):
            names = names_by_category.get(target_category, [])
            options = list(names) if (required and names) else ["", *names]
            current = select.value
            select.options = options
            select.value = current if current in options else options[0]

    @staticmethod
    def _entry_name(record: dict) -> str:
        for spec, widget in record["fields"]:
            if spec.name == "name":
                return (widget.value or "").strip()
        return ""

    def get_devices(self) -> dict[str, list[dict]]:
        """Return the current device library as {category: [entry dict]} (named entries only)."""
        devices: dict[str, list[dict]] = {}
        for key, records in self.device_entry_records.items():
            entries = []
            for record in records:
                entry = {}
                for spec, widget in record["fields"]:
                    value = _read_widget(spec, widget)
                    if value is not None:
                        entry[spec.name] = value
                if entry.get("name"):
                    entries.append(entry)
            devices[key] = entries
        return devices

    def get_channel_rows(self) -> list[dict]:
        """Return the per-channel annotations (wavelengths + device links), aligned to the channels."""
        rows = []
        for record in self.channel_records:
            row = {}
            for name, widget in record["fields"].items():
                value = widget.value
                if name in CHANNEL_LINKS:
                    if value:
                        row[name] = value
                elif value is not None:  # FloatInput wavelength
                    row[name] = float(value)
            rows.append(row)
        return rows

    def get_scalars(self) -> dict:
        """Return the NWBFile/Subject scalar values (experimenter list; age xor date_of_birth)."""
        experimenter = [text_input.value.strip() for text_input in self.experimenter_inputs if text_input.value.strip()]
        scalars = {
            "session_description": self.session_description.value,
            "identifier": self.identifier.value,
            "lab": self.lab.value,
            "institution": self.institution.value,
            "experimenter": experimenter,
            "subject_id": self.subject_id.value,
            "sex": self.sex.value,
            "species": self.species.value,
            "genotype": self.genotype.value,
            "strain": self.strain.value,
            "fiber_photometry_table_name": "",
            "fiber_photometry_table_description": "",
        }
        if self.age_or_dob.value == "Age":
            scalars["age"] = self.age.value
            scalars["date_of_birth"] = ""
        else:
            scalars["age"] = ""
            scalars["date_of_birth"] = self.date_of_birth.value
        return scalars

    def set_from_metadata(self, metadata: dict) -> None:
        """Rebuild the whole form from a saved metadata dict."""
        devices, channel_rows, scalars = parse_metadata_dict(metadata, self.channels)
        # Scalars.
        self.session_description.value = scalars.get("session_description", "")
        self.identifier.value = scalars.get("identifier", "")
        self.lab.value = scalars.get("lab", "")
        self.institution.value = scalars.get("institution", "")
        self.subject_id.value = scalars.get("subject_id", "")
        self.sex.value = scalars.get("sex", "") or ""
        self.species.value = scalars.get("species", "")
        self.genotype.value = scalars.get("genotype", "")
        self.strain.value = scalars.get("strain", "")
        self.age.value = scalars.get("age", "")
        self.date_of_birth.value = scalars.get("date_of_birth", "")
        self.age_or_dob.value = "Date of birth" if scalars.get("date_of_birth") else "Age"
        self.experimenter_box[:] = []
        self.experimenter_inputs = []
        for name in scalars.get("experimenter") or [""]:
            self._add_experimenter(name)
        # Devices: clear and rebuild (device link-selects are recreated by _add_device).
        self.device_link_selects = []
        for key in CATEGORIES:
            self.device_entry_records[key] = []
            self.device_containers[key][:] = []
            for entry in devices.get(key, []):
                self._add_device(key, entry)
        # Channel wavelengths (non-link) can be set directly; refresh the link dropdowns so the
        # just-added device names become valid options before assigning the saved link selections.
        for index, record in enumerate(self.channel_records):
            saved = channel_rows[index] if index < len(channel_rows) else {}
            for name, widget in record["fields"].items():
                if name not in CHANNEL_LINKS:
                    widget.value = _as_float(saved.get(name))
        self.refresh_link_options()
        for index, record in enumerate(self.channel_records):
            saved = channel_rows[index] if index < len(channel_rows) else {}
            for name, widget in record["fields"].items():
                if name in CHANNEL_LINKS and (saved.get(name) or "") in widget.options:
                    widget.value = saved.get(name) or ""

    # ------------------------------------------------------------------------------------------------------------------
    # YAML editor + callbacks
    # ------------------------------------------------------------------------------------------------------------------
    def get_yaml(self) -> dict:
        """Parse and return the metadata dict from the YAML code editor."""
        return loads_yaml(self.code_editor.value)

    def set_yaml(self, metadata: dict) -> None:
        """Serialize ``metadata`` into the YAML code editor."""
        self.code_editor.value = dumps_yaml(metadata)

    def set_alert_message(self, message: str) -> None:
        """Set the alert text, colored red for errors (messages containing 'Alert') and green otherwise."""
        self.alert.object = message
        self.alert.alert_type = "danger" if "Alert" in message else "success"

    def set_path(self, value: str) -> None:
        """Set the displayed save path."""
        self.path.value = value

    def _on_file_upload(self, event: object) -> None:
        """Load an arbitrary metadata YAML the user picked from disk into the form."""
        if not self.load_existing.value:
            return
        try:
            metadata = loads_yaml(self.load_existing.value.decode("utf-8"))
        except Exception as exception:
            self.set_alert_message(f"####Alert !! \n Could not read uploaded YAML: {exception}")
            return
        self.set_from_metadata(metadata)
        self.set_yaml(metadata)
        self.set_alert_message(f"#### Loaded metadata from {self.load_existing.filename}")

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on this selector's buttons."""
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            getattr(self, button_name).on_click(onclick_fn)


def _as_float(value: object, index: int | None = None) -> float | None:
    """Coerce a stored value (or one element of a 2-list) to a float for a FloatInput (None if blank)."""
    if index is not None:
        if isinstance(value, (list, tuple)) and index < len(value):
            value = value[index]
        else:
            return None
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _read_widget(spec: FieldSpec, widget: object) -> object:
    """Read a device field widget back into a Python value (or None to omit)."""
    if spec.link_target:
        return widget.value or None
    if spec.is_list:
        low, high = widget._field_pair  # noqa: SLF001
        if low.value is not None and high.value is not None:
            return [float(low.value), float(high.value)]
        return None
    raw = widget.value
    if spec.dtype == "float":
        # FloatInput already yields a float or None.
        return raw
    if not str(raw).strip():
        return None
    return raw
