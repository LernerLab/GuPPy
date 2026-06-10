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
# Scalar NWBFile/Subject fields the spec marks as required (Subject needs id/sex/species).
_REQUIRED_SCALARS = {"session_description", "identifier", "subject_id", "sex", "species"}


def _required_marker(required: bool) -> str:
    return " *" if required else ""


class MetadataSelector:
    """Editor for one session's complete NWB metadata overlay."""

    def __init__(self, session_label: str, channels: list[Channel], initial_metadata: dict) -> None:
        self.session_label = session_label
        self.channels = channels
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="primary", sizing_mode="stretch_width")

        devices, channel_rows, scalars = parse_metadata_dict(initial_metadata, channels)

        # Tracking structures populated as widgets are built.
        self.device_entry_records: dict[str, list[dict]] = {key: [] for key in CATEGORIES}
        self.device_containers: dict[str, pn.Column] = {
            key: pn.Column(sizing_mode="stretch_width") for key in CATEGORIES
        }
        self.channel_records: list[dict] = []
        # Link dropdowns whose options are device names; channel ones are fixed, device ones rebuild.
        self.channel_link_selects: list[tuple[pn.widgets.Select, str]] = []
        self.device_link_selects: list[tuple[pn.widgets.Select, str]] = []

        scalar_card = self._build_scalar_section(scalars)
        library_section = self._build_device_library(devices)
        channel_card = self._build_channel_table(channel_rows)

        self.code_editor = pn.widgets.CodeEditor(
            value="", theme="tomorrow", language="yaml", height=400, sizing_mode="stretch_width"
        )
        self.load_existing = pn.widgets.Button(name="Load Existing Metadata YAML", sizing_mode="stretch_width")
        self.build_config = pn.widgets.Button(
            name="Show / Refresh YAML from form above", button_type="primary", sizing_mode="stretch_width"
        )
        self.save = pn.widgets.Button(name="Save Metadata", button_type="success", sizing_mode="stretch_width")
        self.path = pn.widgets.TextInput(name="Saved to", disabled=True, sizing_mode="stretch_width")

        self.widget = pn.Column(
            pn.pane.Markdown(f"## NWB Metadata — {session_label}"),
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

        self._refresh_link_options()
        self.set_yaml(build_metadata_dict(self.get_devices(), self.get_channel_rows(), self.get_scalars(), channels))

    # ------------------------------------------------------------------------------------------------------------------
    # Session & Subject
    # ------------------------------------------------------------------------------------------------------------------
    def _build_scalar_section(self, scalars: dict) -> pn.Card:
        text = lambda name, **kw: pn.widgets.TextInput(  # noqa: E731
            name=f"{name}{_required_marker(name in _REQUIRED_SCALARS)}",
            value=scalars.get(name, ""),
            sizing_mode="stretch_width",
            **kw,
        )
        self.session_description = pn.widgets.TextAreaInput(
            name="session_description *",
            value=scalars.get("session_description", ""),
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
        self.sex = pn.widgets.Select(name="sex *", value=scalars.get("sex", "") or "", options=_SEX_OPTIONS, width=120)
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
            add_button = pn.widgets.Button(name=f"➕ Add {category.label.rstrip('s').lower()}", width=220)
            add_button.on_click(lambda event, k=key: (self._add_device(k, {}), self._refresh_link_options()))
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
        fields: list[tuple[object, object]] = []  # (spec, widget or list-row)
        rows = []
        for spec in field_specs(category_key):
            widget = self._make_field_widget(spec, entry)
            fields.append((spec, widget))
            rows.append(widget)
        remove = pn.widgets.Button(name="🗑 Remove", button_type="default", width=110)
        container = pn.Column(
            *rows,
            remove,
            pn.layout.Divider(),
            sizing_mode="stretch_width",
            styles={"border": "1px solid #ddd", "padding": "6px", "margin-bottom": "6px"},
        )
        record = {"container": container, "fields": fields}
        remove.on_click(lambda event: self._remove_device(category_key, record))
        self.device_entry_records[category_key].append(record)
        self.device_containers[category_key].append(container)

    def _remove_device(self, category_key: str, record: dict) -> None:
        if record in self.device_entry_records[category_key]:
            self.device_entry_records[category_key].remove(record)
        if record["container"] in self.device_containers[category_key]:
            self.device_containers[category_key].remove(record["container"])

    def _make_field_widget(self, spec: FieldSpec, entry: dict) -> pn.viewable.Viewable:
        label = f"{spec.name}{_required_marker(spec.required)}"
        value = entry.get(spec.name)
        if spec.link_target:
            select = pn.widgets.Select(name=label, options=[""], value=(value or ""), sizing_mode="stretch_width")
            self.device_link_selects.append((select, spec.link_target))
            return select
        if spec.is_list:
            low = pn.widgets.TextInput(name=f"{label} [min]", value=_as_text(value, 0), width=160)
            high = pn.widgets.TextInput(name=f"{label} [max]", value=_as_text(value, 1), width=160)
            row = pn.Row(low, high)
            row._field_pair = (low, high)  # noqa: SLF001
            return row
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
            excitation = pn.widgets.TextInput(
                name="excitation_wavelength_in_nm *",
                value=_as_text(saved.get("excitation_wavelength_in_nm")),
                width=180,
            )
            emission = pn.widgets.TextInput(
                name="emission_wavelength_in_nm *", value=_as_text(saved.get("emission_wavelength_in_nm")), width=180
            )
            fields["excitation_wavelength_in_nm"] = excitation
            fields["emission_wavelength_in_nm"] = emission
            link_widgets = []
            for link_name, target in CHANNEL_LINKS.items():
                marker = _required_marker(link_name in CHANNEL_REQUIRED_LINKS)
                select = pn.widgets.Select(
                    name=f"{link_name}{marker}", options=[""], value=(saved.get(link_name) or ""), width=180
                )
                self.channel_link_selects.append((select, target))
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
    def _refresh_link_options(self) -> None:
        names_by_category = {
            key: [""] + [name for name in (self._entry_name(record) for record in records) if name]
            for key, records in self.device_entry_records.items()
        }
        for select, target_category in (*self.channel_link_selects, *self.device_link_selects):
            options = names_by_category.get(target_category, [""])
            current = select.value
            select.options = options
            select.value = current if current in options else ""

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
                elif str(value).strip():
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
        # Channel rows: set widget values.
        for index, record in enumerate(self.channel_records):
            saved = channel_rows[index] if index < len(channel_rows) else {}
            for name, widget in record["fields"].items():
                widget.value = _as_text(saved.get(name)) if name not in CHANNEL_LINKS else (saved.get(name) or "")
        self._refresh_link_options()

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
        """Set the text shown in the alert pane."""
        self.alert.object = message

    def set_path(self, value: str) -> None:
        """Set the displayed save path."""
        self.path.value = value

    def attach_callbacks(self, button_name_to_onclick_fn: dict[str, object]) -> None:
        """Register click-handler callbacks on this selector's buttons."""
        for button_name, onclick_fn in button_name_to_onclick_fn.items():
            getattr(self, button_name).on_click(onclick_fn)


def _as_text(value: object, index: int | None = None) -> str:
    """Render a stored value (or one element of a 2-list) as editable text."""
    if value is None:
        return ""
    if index is not None:
        if isinstance(value, (list, tuple)) and index < len(value):
            return str(value[index])
        return ""
    return str(value)


def _read_widget(spec: FieldSpec, widget: object) -> object:
    """Read a device field widget back into a Python value (or None to omit)."""
    if spec.link_target:
        return widget.value or None
    if spec.is_list:
        low, high = widget._field_pair  # noqa: SLF001
        if low.value.strip() and high.value.strip():
            return [float(low.value), float(high.value)]
        return None
    raw = widget.value
    if not str(raw).strip():
        return None
    return float(raw) if spec.dtype == "float" else raw
