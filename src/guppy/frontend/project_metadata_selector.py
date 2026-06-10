"""Panel widget for editing project-level NWB metadata (the reusable hardware/biology overlay).

Mirrors :class:`~guppy.frontend.storenames_selector.StorenamesSelector`: structured
widgets (one editable Tabulator per fiber-photometry component, plus scalar
NWBFile/Subject fields) feed a YAML ``CodeEditor`` that is the authoritative
serialized artifact. The orchestration layer wires the ``load_existing`` /
``build_config`` / ``save`` buttons via :meth:`attach_callbacks`.
"""

import logging

import pandas as pd
import panel as pn

from ..utils.nwb_metadata import (
    COMPONENT_SCHEMAS,
    build_project_metadata_dict,
    dumps_yaml,
    loads_yaml,
    parse_project_metadata_dict,
)

logger = logging.getLogger(__name__)

_WIDTH = 1000


class ProjectMetadataSelector:
    """Editor for the shared project-level metadata overlay (hardware, biology, lab).

    Parameters
    ----------
    initial_metadata : dict
        Project metadata used to pre-populate the tables and scalar fields (and
        the YAML editor). Typically bootstrapped from an existing project YAML,
        a home-directory cache, or the bundled template.
    """

    def __init__(self, initial_metadata: dict) -> None:
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="primary", height=70, width=_WIDTH)

        component_dataframes, scalars = parse_project_metadata_dict(initial_metadata)

        # One editable Tabulator per component, grouped into a collapsible Card per category.
        self.tabulators: dict[str, pn.widgets.Tabulator] = {}
        cards: list[pn.Card] = []
        current_category: str | None = None
        current_members: list[pn.Column] = []

        def _flush_category() -> None:
            if current_category is not None and current_members:
                cards.append(pn.Card(*current_members, title=current_category, collapsed=True, width=_WIDTH))

        for key, schema in COMPONENT_SCHEMAS.items():
            if schema.category != current_category:
                _flush_category()
                current_category = schema.category
                current_members = []
            current_members.append(self._build_component_block(key, component_dataframes[key]))
        _flush_category()
        self.component_cards = cards

        # Scalar project-level widgets.
        self.lab = pn.widgets.TextInput(name="Lab", value=scalars.get("lab", ""), width=320)
        self.institution = pn.widgets.TextInput(name="Institution", value=scalars.get("institution", ""), width=320)
        self.experimenter = pn.widgets.LiteralInput(
            name="Experimenter (list)", value=scalars.get("experimenter", "") or [], type=list, width=320
        )
        self.species = pn.widgets.TextInput(name="Subject species", value=scalars.get("species", ""), width=320)
        self.genotype = pn.widgets.TextInput(name="Subject genotype", value=scalars.get("genotype", ""), width=320)
        self.strain = pn.widgets.TextInput(name="Subject strain", value=scalars.get("strain", ""), width=320)
        self.fiber_photometry_table_name = pn.widgets.TextInput(
            name="Fiber photometry table name",
            value=scalars.get("fiber_photometry_table_name", "") or "fiber_photometry_table",
            width=320,
        )
        self.fiber_photometry_table_description = pn.widgets.TextAreaInput(
            name="Fiber photometry table description",
            value=scalars.get("fiber_photometry_table_description", ""),
            width=660,
            height=80,
        )
        scalar_card = pn.Card(
            pn.Row(self.lab, self.institution, self.experimenter),
            pn.Row(self.species, self.genotype, self.strain),
            pn.Row(self.fiber_photometry_table_name),
            self.fiber_photometry_table_description,
            title="Session defaults & Subject (project-level)",
            collapsed=False,
            width=_WIDTH,
        )

        self.code_editor = pn.widgets.CodeEditor(value="", theme="tomorrow", language="yaml", height=400, width=_WIDTH)

        self.load_existing = pn.widgets.Button(name="Load Existing Project YAML", width=_WIDTH)
        self.build_config = pn.widgets.Button(
            name="Show / Refresh YAML from form above", button_type="primary", width=_WIDTH
        )
        self.save = pn.widgets.Button(name="Save Project Metadata", button_type="success", width=_WIDTH)
        self.path = pn.widgets.TextInput(name="Saved to", disabled=True, width=_WIDTH)

        self.widget = pn.Column(
            pn.pane.Markdown("## Project Metadata (shared across sessions)", width=_WIDTH),
            self.load_existing,
            scalar_card,
            pn.pane.Markdown("### Fiber photometry hardware & biology", width=_WIDTH),
            *self.component_cards,
            pn.layout.Divider(),
            self.build_config,
            pn.pane.Markdown(
                "Edit the YAML below directly if the form cannot express something, then Save.", width=_WIDTH
            ),
            self.code_editor,
            self.alert,
            self.save,
            self.path,
        )

        # Seed the editor with the bootstrapped metadata so it is never empty.
        self.set_yaml(build_project_metadata_dict(self.get_component_dataframes(), self.get_scalars()))

    # ------------------------------------------------------------------------------------------------------------------
    # Component table construction
    # ------------------------------------------------------------------------------------------------------------------
    def _build_component_block(self, key: str, dataframe: pd.DataFrame) -> pn.Column:
        """Build the labelled Tabulator + add/remove-row controls for one component."""
        tabulator = pn.widgets.Tabulator(
            dataframe,
            show_index=False,
            selectable="checkbox",
            layout="fit_data_table",
            width=_WIDTH - 40,
            height=200,
        )
        self.tabulators[key] = tabulator

        add_button = pn.widgets.Button(name="➕ Add row", button_type="default", width=120)
        remove_button = pn.widgets.Button(name="🗑 Remove selected", button_type="default", width=160)

        def add_row(event: object, _key: str = key) -> None:
            table = self.tabulators[_key]
            blank = {column: "" for column in table.value.columns}
            table.value = pd.concat([table.value, pd.DataFrame([blank])], ignore_index=True)

        def remove_rows(event: object, _key: str = key) -> None:
            table = self.tabulators[_key]
            selection = list(table.selection)
            if not selection:
                return
            table.value = table.value.drop(index=selection).reset_index(drop=True)
            table.selection = []

        add_button.on_click(add_row)
        remove_button.on_click(remove_rows)

        return pn.Column(
            pn.pane.Markdown(f"**{key}**", width=_WIDTH - 40),
            tabulator,
            pn.Row(add_button, remove_button),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Getters / setters (mirror StorenamesSelector)
    # ------------------------------------------------------------------------------------------------------------------
    def get_component_dataframes(self) -> dict[str, pd.DataFrame]:
        """Return the current edited DataFrame for every component."""
        return {key: tabulator.value for key, tabulator in self.tabulators.items()}

    def get_scalars(self) -> dict:
        """Return the current scalar project-level field values."""
        return {
            "lab": self.lab.value,
            "institution": self.institution.value,
            "experimenter": self.experimenter.value,
            "species": self.species.value,
            "genotype": self.genotype.value,
            "strain": self.strain.value,
            "fiber_photometry_table_name": self.fiber_photometry_table_name.value,
            "fiber_photometry_table_description": self.fiber_photometry_table_description.value,
        }

    def set_from_metadata(self, metadata: dict) -> None:
        """Repopulate every Tabulator and scalar widget from a metadata dict."""
        component_dataframes, scalars = parse_project_metadata_dict(metadata)
        for key in COMPONENT_SCHEMAS:
            self.tabulators[key].value = component_dataframes[key]
        self.lab.value = scalars.get("lab", "")
        self.institution.value = scalars.get("institution", "")
        self.experimenter.value = scalars.get("experimenter", "") or []
        self.species.value = scalars.get("species", "")
        self.genotype.value = scalars.get("genotype", "")
        self.strain.value = scalars.get("strain", "")
        self.fiber_photometry_table_name.value = (
            scalars.get("fiber_photometry_table_name", "") or "fiber_photometry_table"
        )
        self.fiber_photometry_table_description.value = scalars.get("fiber_photometry_table_description", "")

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
