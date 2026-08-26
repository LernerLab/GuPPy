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
    resolve_session_source,
)
from ..utils.nwb_metadata import (
    Channel,
    build_metadata_dict,
    derive_channels,
    dump_yaml,
    load_yaml,
    validate_metadata_dict,
)
from ..utils.utils import run_folder_for_run, selected_session_runs
from ..utils.validation import validate_data_not_combined

logger = logging.getLogger(__name__)

# One self-contained metadata file per session, saved in its output directory.
METADATA_FILENAME = "nwb_metadata.yaml"


def build_metadata_template(
    *,
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
                devices=selector.get_devices(),
                channel_rows=selector.get_channel_rows(),
                scalars=selector.get_scalars(),
                channels=channels,
            )
        except ValueError as exception:
            selector.set_alert_message(f"####Alert !! \n {exception}", is_error=True)
            return
        selector.set_yaml(built)
        errors = validate_metadata_dict(
            metadata=built, channels=channels, require_session_start_time=require_session_start_time
        )
        selector.set_alert_message(_format_errors(errors) if errors else "#### No alerts !!", is_error=bool(errors))

    def save(event: object = None) -> None:
        try:
            to_save = selector.get_yaml()
        except Exception as exception:
            selector.set_alert_message(f"####Alert !! \n Invalid YAML: {exception}", is_error=True)
            return
        errors = validate_metadata_dict(
            metadata=to_save, channels=channels, require_session_start_time=require_session_start_time
        )
        if errors:
            selector.set_alert_message(_format_errors(errors), is_error=True)
            return
        os.makedirs(os.path.dirname(metadata_yaml_path), exist_ok=True)
        dump_yaml(metadata=to_save, path=metadata_yaml_path)
        selector.set_path(metadata_yaml_path)
        selector.set_alert_message("#### No alerts !!", is_error=False)

    selector.attach_callbacks({"build_config": build_config, "save": save})
    template.main.append(selector.widget)
    return template


def build_metadata_templates(*, inputParameters: dict[str, object]) -> list[pn.template.BootstrapTemplate]:
    """Build one metadata form template per selected session that needs one.

    Each template edits that session's ``nwb_metadata.yaml`` (in its output
    directory), bootstrapped from the saved file when present and otherwise empty.

    Sessions GuPPy processed out of an NWB file are skipped: the export adds their outputs to the
    file they came from, which already carries everything this form collects.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``combine_data`` and ``selected_runs``.

    Returns
    -------
    list of pn.template.BootstrapTemplate
        One fully configured template per session needing a metadata form.

    Raises
    ------
    ValueError
        If the pipeline was run with ``combine_data`` enabled, or if a selected session's source
        cannot be resolved.
    """
    validate_data_not_combined(combine_data=inputParameters["combine_data"])

    pairs = selected_session_runs(inputParameters=inputParameters)
    # Resolved once per session and reused below: a session already in NWB needs no form at all, and
    # for the rest the format decides whether the user must supply the session start time.
    source_by_session = {
        session_path: resolve_session_source(session_folder_path=session_path, inputParameters=inputParameters)
        for session_path, _run_name in pairs
    }
    pairs_needing_metadata = [
        (session_path, run_name) for session_path, run_name in pairs if source_by_session[session_path][1] is None
    ]
    if pairs and not pairs_needing_metadata:
        message = (
            "No metadata form is needed: every selected session was processed out of an NWB file, "
            "which already carries the devices, the fiber photometry chain, and the session start "
            "time. Run Export to NWB to add the GuPPy outputs to it."
        )
        logger.info(message)
        if pn.state.notifications is not None:
            pn.state.notifications.info(message, duration=0)
        return []

    templates = []
    for session_path, run_name in pairs_needing_metadata:
        guppy_folder_path = run_folder_for_run(session_path, run_name)
        metadata_yaml_path = os.path.join(guppy_folder_path, METADATA_FILENAME)
        initial_metadata = load_yaml(metadata_yaml_path) if os.path.exists(metadata_yaml_path) else {}
        channels = derive_channels(output_dir=guppy_folder_path)
        session_label = f"{os.path.basename(session_path.rstrip(os.sep))} ({run_name})"
        acquisition_format, _nwb_source = source_by_session[session_path]
        template = build_metadata_template(
            session_label=session_label,
            channels=channels,
            metadata=initial_metadata,
            metadata_yaml_path=metadata_yaml_path,
            require_session_start_time=not acquisition_supplies_session_start_time(
                session_folder_path=session_path, acquisition_format=acquisition_format
            ),
        )
        templates.append(template)
    return templates


def orchestrate_metadata_page(inputParameters: dict[str, object]) -> None:
    """Open one metadata window per selected session that needs one.

    Each window is served on its own port in a new browser tab, mirroring the
    Label Stores GUI.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; see :func:`build_metadata_templates`.

    Raises
    ------
    ValueError
        If the pipeline was run with ``combine_data`` enabled, or if a selected session's source
        cannot be resolved.
    """
    for template in build_metadata_templates(inputParameters=inputParameters):
        template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
