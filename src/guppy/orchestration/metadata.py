"""Step 6 orchestration: the Input Metadata pages.

Opens one window per selected session, each editing that session's single
self-contained ``nwb_metadata.yaml`` (saved in the session's output directory).
Modeled on the Storenames GUI: structured widgets feed a YAML ``CodeEditor`` that
is the authoritative artifact, and Save writes the editor's YAML to disk. The
saved YAML is a reusable, hand-editable source of truth for the Step 7 export.
"""

import logging
import os

import panel as pn

from guppy.frontend.frontend_utils import scanPortsAndFind

from ..frontend.metadata_selector import MetadataSelector
from ..utils.nwb_metadata import build_metadata_dict, dump_yaml, load_yaml
from ..utils.utils import output_dir_for_run

logger = logging.getLogger(__name__)

# One self-contained metadata file per session, saved in its output directory.
METADATA_FILENAME = "nwb_metadata.yaml"


def _selected_session_runs(inputParameters: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten ``selectedOutputs`` into ``(session_path, run_name)`` pairs."""
    selected_outputs: dict[str, list[str]] = inputParameters["selectedOutputs"]
    pairs: list[tuple[str, str]] = []
    for session_path, run_names in selected_outputs.items():
        for run_name in run_names:
            pairs.append((session_path, run_name))
    return pairs


def build_metadata_template(
    session_label: str, metadata: dict, metadata_yaml_path: str
) -> pn.template.BootstrapTemplate:
    """Build one session's metadata page (without serving it)."""
    selector = MetadataSelector(session_label=session_label, initial_metadata=metadata)
    template = pn.template.BootstrapTemplate(title=f"Metadata GUI - {session_label}")

    def load_existing(event: object = None) -> None:
        if not os.path.exists(metadata_yaml_path):
            selector.set_alert_message("####Alert !! \n No existing metadata YAML found for this session.")
            return
        loaded = load_yaml(metadata_yaml_path)
        selector.set_from_metadata(loaded)
        selector.set_yaml(loaded)
        selector.set_alert_message(f"#### Loaded from {metadata_yaml_path}")

    def build_config(event: object = None) -> None:
        try:
            built = build_metadata_dict(selector.get_component_dataframes(), selector.get_scalars())
        except ValueError as exception:
            selector.set_alert_message(f"####Alert !! \n {exception}")
            return
        selector.set_yaml(built)
        selector.set_alert_message("#### No alerts !!")

    def save(event: object = None) -> None:
        try:
            to_save = selector.get_yaml()
        except Exception as exception:
            selector.set_alert_message(f"####Alert !! \n Invalid YAML: {exception}")
            return
        os.makedirs(os.path.dirname(metadata_yaml_path), exist_ok=True)
        dump_yaml(to_save, metadata_yaml_path)
        selector.set_path(metadata_yaml_path)
        selector.set_alert_message("#### No alerts !!")

    selector.attach_callbacks({"load_existing": load_existing, "build_config": build_config, "save": save})
    template.main.append(selector.widget)
    return template


def orchestrate_metadata_page(inputParameters: dict[str, object]) -> None:
    """Open one metadata window per selected session.

    Each window edits that session's ``nwb_metadata.yaml`` (in its output
    directory), bootstrapped from the saved file when present and otherwise empty.
    Each window is served on its own port in a new browser tab, mirroring the
    Storenames GUI. Skipped in headless mode (``GUPPY_BASE_DIR`` set).
    """
    headless = bool(os.environ.get("GUPPY_BASE_DIR"))

    for session_path, run_name in _selected_session_runs(inputParameters):
        guppy_folder_path = output_dir_for_run(session_path, run_name)
        metadata_yaml_path = os.path.join(guppy_folder_path, METADATA_FILENAME)
        initial_metadata = load_yaml(metadata_yaml_path) if os.path.exists(metadata_yaml_path) else {}
        session_label = f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})"
        template = build_metadata_template(session_label, initial_metadata, metadata_yaml_path)
        if not headless:
            template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
