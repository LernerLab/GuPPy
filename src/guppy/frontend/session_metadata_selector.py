"""Panel widget for editing per-session NWB metadata (session description, subject).

Companion to :class:`~guppy.frontend.project_metadata_selector.ProjectMetadataSelector`:
the project window holds the shared hardware/biology overlay, while one of these
per-session windows holds the fields that differ between sessions. Same pattern --
scalar widgets feed a YAML ``CodeEditor`` that is the authoritative artifact.
"""

import logging

import panel as pn

from ..utils.nwb_metadata import (
    build_session_metadata_dict,
    dumps_yaml,
    loads_yaml,
    parse_session_metadata_dict,
)

logger = logging.getLogger(__name__)

_WIDTH = 600
_SEX_OPTIONS = ["", "M", "F", "U", "O"]


class SessionMetadataSelector:
    """Editor for the per-session metadata overlay of a single session/run.

    Parameters
    ----------
    session_name : str
        Basename of the session, shown in the heading.
    initial_metadata : dict
        Per-session metadata used to pre-populate the fields and the YAML editor.
    """

    def __init__(self, session_name: str, initial_metadata: dict) -> None:
        self.session_name = session_name
        self.alert = pn.pane.Alert("#### No alerts !!", alert_type="primary", height=70, width=_WIDTH)

        scalars = parse_session_metadata_dict(initial_metadata)

        self.session_description = pn.widgets.TextAreaInput(
            name="Session description", value=scalars.get("session_description", ""), width=_WIDTH, height=80
        )
        self.identifier = pn.widgets.TextInput(
            name="Identifier",
            value=scalars.get("identifier", ""),
            placeholder="leave blank to auto-generate",
            width=_WIDTH,
        )
        self.subject_id = pn.widgets.TextInput(name="Subject ID", value=scalars.get("subject_id", ""), width=_WIDTH)
        self.sex = pn.widgets.Select(name="Sex", value=scalars.get("sex", "") or "", options=_SEX_OPTIONS, width=120)
        self.age = pn.widgets.TextInput(name="Age (ISO 8601, e.g. P90D)", value=scalars.get("age", ""), width=240)
        self.date_of_birth = pn.widgets.TextInput(
            name="Date of birth (ISO 8601)", value=scalars.get("date_of_birth", ""), width=240
        )

        self.code_editor = pn.widgets.CodeEditor(value="", theme="tomorrow", language="yaml", height=250, width=_WIDTH)

        self.load_existing = pn.widgets.Button(name="Load Existing Session YAML", width=_WIDTH)
        self.build_config = pn.widgets.Button(
            name="Show / Refresh YAML from form above", button_type="primary", width=_WIDTH
        )
        self.save = pn.widgets.Button(name="Save Session Metadata", button_type="success", width=_WIDTH)
        self.path = pn.widgets.TextInput(name="Saved to", disabled=True, width=_WIDTH)

        self.widget = pn.Column(
            pn.pane.Markdown(f"## Session Metadata — {session_name}", width=_WIDTH),
            self.load_existing,
            self.session_description,
            self.identifier,
            pn.pane.Markdown("### Subject", width=_WIDTH),
            self.subject_id,
            pn.Row(self.sex, self.age, self.date_of_birth),
            pn.layout.Divider(),
            self.build_config,
            self.code_editor,
            self.alert,
            self.save,
            self.path,
        )

        self.set_yaml(build_session_metadata_dict(self.get_scalars()))

    # ------------------------------------------------------------------------------------------------------------------
    # Getters / setters (mirror StorenamesSelector)
    # ------------------------------------------------------------------------------------------------------------------
    def get_scalars(self) -> dict:
        """Return the current per-session field values."""
        return {
            "session_description": self.session_description.value,
            "identifier": self.identifier.value,
            "subject_id": self.subject_id.value,
            "sex": self.sex.value,
            "age": self.age.value,
            "date_of_birth": self.date_of_birth.value,
        }

    def set_from_metadata(self, metadata: dict) -> None:
        """Repopulate the fields from a metadata dict."""
        scalars = parse_session_metadata_dict(metadata)
        self.session_description.value = scalars.get("session_description", "")
        self.identifier.value = scalars.get("identifier", "")
        self.subject_id.value = scalars.get("subject_id", "")
        self.sex.value = scalars.get("sex", "") or ""
        self.age.value = scalars.get("age", "")
        self.date_of_birth.value = scalars.get("date_of_birth", "")

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
