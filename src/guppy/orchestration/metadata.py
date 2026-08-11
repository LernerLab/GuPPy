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

from ..frontend import nwb_form_style as style
from ..frontend.metadata_selector import MetadataSelector
from ..utils.acquisition_format import (
    acquisition_supplies_session_start_time,
    resolve_acquisition_format,
)
from ..utils.nwb_metadata import (
    Channel,
    build_metadata_dict,
    derive_channels,
    dump_yaml,
    load_yaml,
    validate_metadata_dict,
)
from ..utils.utils import run_folder_for_run
from ..utils.validation import validate_data_not_combined

logger = logging.getLogger(__name__)

# One self-contained metadata file per session, saved in its output directory.
METADATA_FILENAME = "nwb_metadata.yaml"


def _selected_session_runs(inputParameters: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten ``selected_runs`` into ``(session_path, run_name)`` pairs."""
    selected_runs: dict[str, list[str]] = inputParameters["selected_runs"]
    pairs: list[tuple[str, str]] = []
    for session_path, run_names in selected_runs.items():
        for run_name in run_names:
            pairs.append((session_path, run_name))
    return pairs


def build_metadata_template(
    session_label: str,
    channels: list[Channel],
    metadata: dict,
    metadata_yaml_path: str,
    require_session_start_time: bool = False,
) -> pn.template.BootstrapTemplate:
    """Build one session's metadata page (without serving it).

    ``require_session_start_time`` marks the session start time as a required field, for the
    acquisition formats whose raw files do not record one.
    """
    selector = MetadataSelector(
        session_label=session_label,
        channels=channels,
        initial_metadata=metadata,
        require_session_start_time=require_session_start_time,
    )
    template = pn.template.BootstrapTemplate(
        title=f"Metadata GUI - {session_label}",
        header_background=style.HEADER_BG,
        header_color="#FFFFFF",
        main_max_width="1120px",
    )
    template.config.raw_css = [style.DOCUMENT_CSS]

    def _format_errors(errors: list[str]) -> str:
        return "####Alert !! \n Missing required metadata for NWB export:\n" + "\n".join(
            f"- {error}" for error in errors
        )

    def build_config(event: object = None) -> None:
        selector.refresh_link_options()
        try:
            built = build_metadata_dict(
                selector.get_devices(), selector.get_channel_rows(), selector.get_scalars(), channels
            )
        except ValueError as exception:
            selector.set_alert_message(f"####Alert !! \n {exception}")
            return
        selector.set_yaml(built)
        errors = validate_metadata_dict(built, channels, require_session_start_time=require_session_start_time)
        selector.set_alert_message(_format_errors(errors) if errors else "#### No alerts !!")

    def save(event: object = None) -> None:
        try:
            to_save = selector.get_yaml()
        except Exception as exception:
            selector.set_alert_message(f"####Alert !! \n Invalid YAML: {exception}")
            return
        errors = validate_metadata_dict(to_save, channels, require_session_start_time=require_session_start_time)
        if errors:
            selector.set_alert_message(_format_errors(errors))
            return
        os.makedirs(os.path.dirname(metadata_yaml_path), exist_ok=True)
        dump_yaml(to_save, metadata_yaml_path)
        selector.set_path(metadata_yaml_path)
        selector.set_alert_message("#### No alerts !!")

    selector.attach_callbacks({"build_config": build_config, "save": save})
    template.main.append(selector.widget)
    return template


def _requires_session_start_time(session_path: str) -> bool:
    """Report whether the user must supply this session's start time in the form.

    A session whose format cannot be resolved is treated as needing one: Step 7 reports the format
    problem itself, and asking for the start time here costs nothing if it turns out to be redundant.
    """
    try:
        acquisition_format = resolve_acquisition_format(session_path)
    except ValueError as exception:
        logger.warning(f"Could not resolve the acquisition format of '{session_path}': {exception}")
        return True
    return not acquisition_supplies_session_start_time(
        session_folder_path=session_path, acquisition_format=acquisition_format
    )


def orchestrate_metadata_page(inputParameters: dict[str, object]) -> None:
    """Open one metadata window per selected session.

    Each window edits that session's ``nwb_metadata.yaml`` (in its output
    directory), bootstrapped from the saved file when present and otherwise empty.
    Each window is served on its own port in a new browser tab, mirroring the
    Storenames GUI. Skipped in headless mode (``GUPPY_BASE_DIR`` set).

    Raises
    ------
    ValueError
        If the pipeline was run with ``combine_data`` enabled.
    """
    validate_data_not_combined(combine_data=inputParameters["combine_data"])
    headless = bool(os.environ.get("GUPPY_BASE_DIR"))

    for session_path, run_name in _selected_session_runs(inputParameters):
        guppy_folder_path = run_folder_for_run(session_path, run_name)
        metadata_yaml_path = os.path.join(guppy_folder_path, METADATA_FILENAME)
        initial_metadata = load_yaml(metadata_yaml_path) if os.path.exists(metadata_yaml_path) else {}
        channels = derive_channels(guppy_folder_path)
        session_label = f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})"
        template = build_metadata_template(
            session_label,
            channels,
            initial_metadata,
            metadata_yaml_path,
            require_session_start_time=_requires_session_start_time(session_path),
        )
        if not headless:
            template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
