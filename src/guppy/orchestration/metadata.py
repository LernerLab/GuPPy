"""Step 6 orchestration: the Input Project Metadata pages.

Opens one project window (shared hardware/biology overlay) plus one window per
selected session (per-session description/subject), each modeled on the
Storenames GUI: structured widgets feed a YAML ``CodeEditor`` that is the
authoritative artifact, and Save writes the editor's YAML to disk. The project
YAML doubles as a reusable, hand-editable source of truth for later runs.
"""

import importlib.resources
import logging
import os
from pathlib import Path

import panel as pn

from guppy.frontend.frontend_utils import scanPortsAndFind

from ..frontend.project_metadata_selector import ProjectMetadataSelector
from ..frontend.session_metadata_selector import SessionMetadataSelector
from ..utils.nwb_metadata import (
    build_project_metadata_dict,
    build_session_metadata_dict,
    dump_yaml,
    load_yaml,
    loads_yaml,
    split_template_into_project_and_session,
)
from ..utils.utils import output_dir_for_run

logger = logging.getLogger(__name__)

HOME_CACHE_PATH = os.path.join(str(Path.home()), ".guppy_nwb_project_metadata.yaml")

# Filenames for the on-disk metadata overlays, shared with the Step 7 export.
PROJECT_METADATA_FILENAME = "nwb_project_metadata.yaml"
SESSION_METADATA_FILENAME = "nwb_session_metadata.yaml"


def _selected_session_runs(inputParameters: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten ``selectedOutputs`` into ``(session_path, run_name)`` pairs."""
    selected_outputs: dict[str, list[str]] = inputParameters["selectedOutputs"]
    pairs: list[tuple[str, str]] = []
    for session_path, run_names in selected_outputs.items():
        for run_name in run_names:
            pairs.append((session_path, run_name))
    return pairs


def _load_bundled_template() -> dict:
    """Load the packaged starter fiber-photometry metadata template."""
    text = importlib.resources.files("guppy").joinpath("resources/fiber_photometry_metadata_template.yaml").read_text()
    return loads_yaml(text)


def _bootstrap_project_metadata(project_yaml_path: str) -> dict:
    """Pick the best available source for the project window's initial metadata."""
    if os.path.exists(project_yaml_path):
        return load_yaml(project_yaml_path)
    if os.path.exists(HOME_CACHE_PATH):
        return load_yaml(HOME_CACHE_PATH)
    return _load_bundled_template()


def build_project_metadata_template(project_metadata: dict, project_yaml_path: str) -> pn.template.BootstrapTemplate:
    """Build the project-metadata page (without serving it)."""
    selector = ProjectMetadataSelector(initial_metadata=project_metadata)
    template = pn.template.BootstrapTemplate(title="Project Metadata GUI")

    def load_existing(event: object = None) -> None:
        source = project_yaml_path if os.path.exists(project_yaml_path) else HOME_CACHE_PATH
        if not os.path.exists(source):
            selector.set_alert_message("####Alert !! \n No existing project metadata YAML found.")
            return
        metadata = load_yaml(source)
        selector.set_from_metadata(metadata)
        selector.set_yaml(metadata)
        selector.set_alert_message(f"#### Loaded from {source}")

    def build_config(event: object = None) -> None:
        try:
            metadata = build_project_metadata_dict(selector.get_component_dataframes(), selector.get_scalars())
        except ValueError as exception:
            selector.set_alert_message(f"####Alert !! \n {exception}")
            return
        selector.set_yaml(metadata)
        selector.set_alert_message("#### No alerts !!")

    def save(event: object = None) -> None:
        try:
            metadata = selector.get_yaml()
        except Exception as exception:
            selector.set_alert_message(f"####Alert !! \n Invalid YAML: {exception}")
            return
        dump_yaml(metadata, project_yaml_path)
        dump_yaml(metadata, HOME_CACHE_PATH)
        selector.set_path(project_yaml_path)
        selector.set_alert_message("#### No alerts !!")

    selector.attach_callbacks({"load_existing": load_existing, "build_config": build_config, "save": save})
    template.main.append(selector.widget)
    return template


def build_session_metadata_template(
    session_label: str, session_metadata: dict, session_yaml_path: str
) -> pn.template.BootstrapTemplate:
    """Build a per-session metadata page (without serving it)."""
    selector = SessionMetadataSelector(session_name=session_label, initial_metadata=session_metadata)
    template = pn.template.BootstrapTemplate(title=f"Session Metadata GUI - {session_label}")

    def load_existing(event: object = None) -> None:
        if not os.path.exists(session_yaml_path):
            selector.set_alert_message("####Alert !! \n No existing session metadata YAML found.")
            return
        metadata = load_yaml(session_yaml_path)
        selector.set_from_metadata(metadata)
        selector.set_yaml(metadata)
        selector.set_alert_message(f"#### Loaded from {session_yaml_path}")

    def build_config(event: object = None) -> None:
        selector.set_yaml(build_session_metadata_dict(selector.get_scalars()))
        selector.set_alert_message("#### No alerts !!")

    def save(event: object = None) -> None:
        try:
            metadata = selector.get_yaml()
        except Exception as exception:
            selector.set_alert_message(f"####Alert !! \n Invalid YAML: {exception}")
            return
        os.makedirs(os.path.dirname(session_yaml_path), exist_ok=True)
        dump_yaml(metadata, session_yaml_path)
        selector.set_path(session_yaml_path)
        selector.set_alert_message("#### No alerts !!")

    selector.attach_callbacks({"load_existing": load_existing, "build_config": build_config, "save": save})
    template.main.append(selector.widget)
    return template


def orchestrate_project_metadata_page(inputParameters: dict[str, object]) -> None:
    """Open the project window and one window per selected session.

    Each window is served on its own port in a new browser tab, mirroring the
    Storenames GUI. Skipped in headless mode (``GUPPY_BASE_DIR`` set).
    """
    headless = bool(os.environ.get("GUPPY_BASE_DIR"))
    project_yaml_path = os.path.join(inputParameters["abspath"], PROJECT_METADATA_FILENAME)

    full_metadata = _bootstrap_project_metadata(project_yaml_path)
    project_metadata, session_defaults = split_template_into_project_and_session(full_metadata)

    project_template = build_project_metadata_template(project_metadata, project_yaml_path)
    if not headless:
        project_template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))

    for session_path, run_name in _selected_session_runs(inputParameters):
        guppy_folder_path = output_dir_for_run(session_path, run_name)
        session_yaml_path = os.path.join(guppy_folder_path, SESSION_METADATA_FILENAME)
        initial_metadata = load_yaml(session_yaml_path) if os.path.exists(session_yaml_path) else session_defaults
        session_label = f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})"
        session_template = build_session_metadata_template(session_label, initial_metadata, session_yaml_path)
        if not headless:
            session_template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
