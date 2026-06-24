import logging
import os

import panel as pn

from ..frontend.custom_events_config import CustomEventsConfig
from ..frontend.frontend_utils import scanPortsAndFind
from ..utils.custom_events import (
    is_sorted,
    parse_pasted_timestamps,
    write_custom_event_csv,
)

logger = logging.getLogger(__name__)


def build_custom_events_template(folder_path: str) -> pn.template.BootstrapTemplate:
    """Build and return the Import Custom Events GUI Panel template without serving it.

    Parameters
    ----------
    folder_path : str
        Absolute path to the session directory the events are written into.

    Returns
    -------
    pn.template.BootstrapTemplate
        Fully configured Panel template ready to be served.
    """
    template = pn.template.BootstrapTemplate(title="Import Custom Events - {}".format(os.path.basename(folder_path)))
    config = CustomEventsConfig()

    def save_button(event: object = None) -> None:
        written = []
        warnings = []
        for raw_name, text in config.get_rows():
            name = raw_name.strip()
            # Fully-empty rows are skipped so the step can be completed with nothing to import.
            if not name and not text.strip():
                continue
            if not name:
                config.set_alert_message("#### Alert !! \n Every event with timestamps needs a name.")
                return
            try:
                timestamps = parse_pasted_timestamps(text)
            except ValueError as exc:
                config.set_alert_message(f"#### Alert !! \n {exc}")
                return
            if not timestamps:
                config.set_alert_message(f"#### Alert !! \n Event '{name}' has a name but no timestamps.")
                return
            if not is_sorted(timestamps):
                warnings.append(f"'{name}' timestamps are not in increasing order; saved in the order pasted.")
            try:
                csv_path = write_custom_event_csv(
                    name=name, timestamps=timestamps, folder_path=folder_path, overwrite=config.overwrite.value
                )
            except FileExistsError as exc:
                config.set_alert_message(f"#### Alert !! \n {exc}")
                return
            written.append(os.path.basename(csv_path))

        if not written:
            config.set_alert_message("#### No events to save — continue to the Storenames GUI.")
            return
        message = "#### Saved: " + ", ".join(written)
        if warnings:
            message += "\n\n**Warning(s):** " + " ".join(warnings)
        config.set_alert_message(message)

    config.attach_callbacks({"add_event": config.add_event_row, "save": save_button})
    template.main.append(config.widget)

    # Expose hooks/widgets so tests can drive the page headlessly without a browser.
    template._hooks = {"save_button": save_button}
    template._widgets = {"config": config}

    return template


def build_custom_events_page(inputParameters: dict[str, object], folder_path: str) -> None:
    """Write custom event CSVs for one session, headlessly or via the Panel GUI.

    In headless mode (``custom_events_map`` key present in ``inputParameters``)
    the events for this session are written directly. Otherwise a Panel GUI is
    launched in a browser so the user can paste events interactively.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; may contain ``custom_events_map`` for
        headless operation, keyed by session-folder name.
    folder_path : str
        Absolute path to the session directory.
    """
    custom_events_map = inputParameters.get("custom_events_map")
    if isinstance(custom_events_map, dict):
        events_for_session = custom_events_map.get(os.path.basename(folder_path), {})
        for name, timestamps in events_for_session.items():
            csv_path = write_custom_event_csv(
                name=name, timestamps=list(timestamps), folder_path=folder_path, overwrite=True
            )
            logger.info(f"Custom event saved at {csv_path}")
        return

    template = build_custom_events_template(folder_path)
    number = scanPortsAndFind(start_port=5000, end_port=5200)
    template.show(port=number)


def orchestrate_custom_events_page(inputParameters: dict[str, object]) -> None:
    """Run the optional Import Custom Events step for every selected session folder.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``folderNames`` and ``abspath``,
        and optionally ``custom_events_map`` for headless operation.
    """
    folderNames = inputParameters["folderNames"]
    logger.info(folderNames)

    for i in folderNames:
        folder_path = os.path.join(inputParameters["abspath"], i)
        build_custom_events_page(inputParameters, folder_path)
    logger.info("#" * 400)
