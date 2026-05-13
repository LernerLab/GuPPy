import json
import logging
import os
import shutil
from pathlib import Path

import holoviews as hv  # noqa: F401
import numpy as np
import panel as pn

from guppy.extractors import (
    CsvRecordingExtractor,
    DandiNwbRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    NwbRecordingExtractor,
    TdtRecordingExtractor,
    detect_acquisition_formats,
)
from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.frontend.npm_gui_prompts import (
    get_multi_event_responses,
    get_timestamp_configuration,
)
from guppy.frontend.storenames_instructions import (
    StorenamesInstructions,
    StorenamesInstructionsNPM,
)
from guppy.frontend.storenames_selector import StorenamesSelector
from guppy.utils.utils import (
    discover_output_dirs,
    output_dir_for_run,
    validate_run_name,
)

pn.extension()

logger = logging.getLogger(__name__)


def show_dir(filepath: str, run_name: str | None = None) -> str:
    """Return the path of an output directory without creating it.

    Parameters
    ----------
    filepath : str
        Path to the session folder.
    run_name : str or None, optional
        Explicit run-name suffix.  When ``None`` (the default) the legacy
        next-available-integer behaviour is used.

    Returns
    -------
    str
        Path of the form ``<filepath>/<basename>_output_<run_name>``.  When
        ``run_name`` is ``None`` the suffix is the lowest integer for which
        the directory does not yet exist.
    """
    if run_name is not None:
        validate_run_name(run_name)
        return output_dir_for_run(filepath, run_name)

    i = 1
    while True:
        op = output_dir_for_run(filepath, str(i))
        if not os.path.exists(op):
            break
        i += 1
    return op


def make_dir(filepath: str, run_name: str | None = None, run_name_policy: str = "create") -> str:
    """Create and return an output directory.

    Parameters
    ----------
    filepath : str
        Path to the session folder.
    run_name : str or None, optional
        Explicit run-name suffix.  When ``None`` (the default) the next-available
        integer is used (legacy behaviour).
    run_name_policy : {"create", "overwrite"}, optional
        With ``run_name`` set, controls collision behaviour.  ``"create"``
        raises if the target directory already exists; ``"overwrite"``
        ``rmtree``s an existing directory before recreating it.  Ignored when
        ``run_name`` is ``None`` (auto-increment never collides).

    Returns
    -------
    str
        Path of the newly created directory.
    """
    if run_name is None:
        i = 1
        while True:
            op = output_dir_for_run(filepath, str(i))
            if not os.path.exists(op):
                os.mkdir(op)
                return op
            i += 1

    validate_run_name(run_name)
    if run_name_policy not in ("create", "overwrite"):
        raise ValueError(f"run_name_policy must be 'create' or 'overwrite'; got {run_name_policy!r}.")
    op = output_dir_for_run(filepath, run_name)
    if os.path.exists(op):
        if run_name_policy == "create":
            raise ValueError(
                f"Output directory already exists: {op!r}. "
                "Choose a different runName or set runNamePolicy='overwrite' to replace it."
            )
        shutil.rmtree(op)
        logger.info(f"Cleared output directory for overwrite: {op}")
    os.mkdir(op)
    return op


def _fetchValues(
    text: object,
    storenames: list,
    storename_dropdowns: dict,
    storename_textboxes: dict,
    storenames_config: dict,
    isosbestic_control: bool = False,
) -> str:
    if not storename_dropdowns or not len(storenames) > 0:
        return "####Alert !! \n No storenames selected."

    comboBoxValues, textBoxValues = [], []
    dropdown_keys = list(storename_dropdowns.keys())

    # Get dropdown values
    for key in dropdown_keys:
        comboBoxValues.append(storename_dropdowns[key].value)

    # Get textbox values (all storenames always have a textbox)
    for key in dropdown_keys:
        textbox_value = storename_textboxes[key].value or ""
        textBoxValues.append(textbox_value)

        # Validation: Check for whitespace
        if len(textbox_value.split()) > 1:
            return "####Alert !! \n Whitespace is not allowed in the text box entry."

        # Validation: Check for empty required fields
        dropdown_value = storename_dropdowns[key].value
        if not textbox_value and dropdown_value in ["control", "signal", "event TTLs"]:
            return "####Alert !! \n One of the text box entry is empty."

    if len(comboBoxValues) != len(textBoxValues):
        return "####Alert !! \n Number of entries in combo box and text box should be same."

    names_for_storenames = []
    signal_regions = []
    control_regions = []
    for i in range(len(comboBoxValues)):
        if comboBoxValues[i] == "control" or comboBoxValues[i] == "signal":
            if "_" in textBoxValues[i]:
                return "####Alert !! \n Please do not use underscore in region name."
            names_for_storenames.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
            if comboBoxValues[i] == "signal":
                signal_regions.append(textBoxValues[i])
            else:
                control_regions.append(textBoxValues[i])
        elif comboBoxValues[i] == "event TTLs":
            names_for_storenames.append(textBoxValues[i])
        else:
            names_for_storenames.append(comboBoxValues[i])

    # Validation: reject duplicate names_for_storenames entries
    seen = set()
    duplicates = []
    for name in names_for_storenames:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        return (
            "####Alert !! \n Duplicate name(s) in names_for_storenames: {}. "
            "Each name (e.g. 'signal_DMS', 'lever_press') must be unique.".format(", ".join(duplicates))
        )

    # Validation: when isosbestic control is enabled, every signal_<R> must have a
    # matching control_<R> and vice versa. Skipped when isosbestic_control is False
    # (signal-only configurations are valid in that case).
    signal_without_control = sorted(set(signal_regions) - set(control_regions))
    control_without_signal = sorted(set(control_regions) - set(signal_regions))
    if isosbestic_control and (signal_without_control or control_without_signal):
        parts = []
        if signal_without_control:
            parts.append("signal region(s) without a matching control: {}".format(", ".join(signal_without_control)))
        if control_without_signal:
            parts.append("control region(s) without a matching signal: {}".format(", ".join(control_without_signal)))
        return (
            "####Alert !! \n Mismatched signal/control region pairs — {}. "
            "Every 'signal_<region>' must have a matching 'control_<region>'.".format("; ".join(parts))
        )

    storenames_config["storenames"] = text.value
    storenames_config["names_for_storenames"] = names_for_storenames
    return "#### No alerts !!"


def _save(storenames_config: dict, select_location: str) -> str:
    arr1, arr2 = np.asarray(storenames_config["storenames"]), np.asarray(storenames_config["names_for_storenames"])

    empty_indices = np.where(arr2 == "")[0].tolist()
    if empty_indices:
        detail = (
            f"Empty string in the list names_for_storenames at index {empty_indices[0]} "
            f"(storename {arr1[empty_indices[0]]!r})."
            if len(empty_indices) == 1
            else (
                f"Empty strings in the list names_for_storenames at {len(empty_indices)} indices: "
                f"{empty_indices} (storenames {[str(arr1[i]) for i in empty_indices]})."
            )
        )
        alert_message = f"#### Alert !! \n {detail} Provide a semantic name for each storename."
        logger.error(detail)
        return alert_message

    if arr1.shape[0] != arr2.shape[0]:
        detail = (
            f"Length of list storenames ({arr1.shape[0]}) and names_for_storenames ({arr2.shape[0]}) "
            "is not equal; each storename must be paired with exactly one semantic name."
        )
        alert_message = f"#### Alert !! \n {detail}"
        logger.error(detail)
        return alert_message

    if not os.path.exists(os.path.join(Path.home(), ".storesList.json")):
        storenames_cache = dict()

        for i in range(arr1.shape[0]):
            if arr1[i] in storenames_cache:
                storenames_cache[arr1[i]].append(arr2[i])
                storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
            else:
                storenames_cache[arr1[i]] = [arr2[i]]

        with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
            json.dump(storenames_cache, f, indent=4)
    else:
        with open(os.path.join(Path.home(), ".storesList.json")) as f:
            storenames_cache = json.load(f)

        for i in range(arr1.shape[0]):
            if arr1[i] in storenames_cache:
                storenames_cache[arr1[i]].append(arr2[i])
                storenames_cache[arr1[i]] = list(set(storenames_cache[arr1[i]]))
            else:
                storenames_cache[arr1[i]] = [arr2[i]]

        with open(os.path.join(Path.home(), ".storesList.json"), "w") as f:
            json.dump(storenames_cache, f, indent=4)

    arr = np.asarray([arr1, arr2])
    logger.info(arr)
    if os.path.exists(select_location):
        # Overwrite mode: clear all derived data from the previous run before saving the new storesList.
        shutil.rmtree(select_location)
        logger.info(f"Cleared output directory for overwrite: {select_location}")
    os.mkdir(select_location)

    np.savetxt(os.path.join(select_location, "storesList.csv"), arr, delimiter=",", fmt="%s")
    logger.info(f"Storeslist file saved at {select_location}")
    logger.info("Storeslist : \n" + str(arr))
    return "#### No alerts !!"


def build_storenames_template(
    events: list[str], flags: list[str], folder_path: str, isosbestic_control: bool = False
) -> pn.template.BootstrapTemplate:
    """Build and return the Storenames GUI Panel template without serving it.

    Parameters
    ----------
    events : list of str
        Storename strings discovered from the data acquisition files.
    flags : list of str
        Feature flags (e.g. ``"data_np_v2"``) that control which instructions widget is shown.
    folder_path : str
        Absolute path to the session directory.

    Returns
    -------
    pn.template.BootstrapTemplate
        Fully configured Panel template ready to be served.
    """
    allnames = events

    template = pn.template.BootstrapTemplate(title="Storenames GUI - {}".format(os.path.basename(folder_path)))

    if "data_np_v2" in flags or "data_np" in flags or "event_np" in flags:
        storenames_instructions = StorenamesInstructionsNPM(folder_path=folder_path)
    else:
        storenames_instructions = StorenamesInstructions(folder_path=folder_path)
    storenames_selector = StorenamesSelector(allnames=allnames)

    storenames = []
    storename_dropdowns = {}
    storename_textboxes = {}

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions
    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event: object) -> None:
        if event.new == "over_write_file":
            options = discover_output_dirs(folder_path)
            storenames_selector.set_select_location_options(options=options)
        else:
            run_name = storenames_selector.get_run_name()
            options = [show_dir(folder_path, run_name=run_name or None)]
            storenames_selector.set_select_location_options(options=options)

    def run_name_input_changed(event: object) -> None:
        if storenames_selector.get_overwrite_mode() != "create_new_file":
            return
        run_name = event.new or None
        try:
            options = [show_dir(folder_path, run_name=run_name)]
        except ValueError as exc:
            storenames_selector.set_alert_message(f"####Alert !! \n {exc}")
            return
        storenames_selector.set_select_location_options(options=options)
        storenames_selector.set_alert_message("#### No alerts !!")

    def fetchValues(event: object) -> None:
        global storenames
        storenames_config = dict()
        alert_message = _fetchValues(
            text=storenames_selector.text,
            storenames=storenames,
            storename_dropdowns=storename_dropdowns,
            storename_textboxes=storename_textboxes,
            storenames_config=storenames_config,
            isosbestic_control=isosbestic_control,
        )
        storenames_selector.set_alert_message(alert_message)
        storenames_selector.set_literal_input_2(storenames_config=storenames_config)

    # on clicking 'Select Storenames' button, following function is executed
    def update_values(event: object) -> None:
        global storenames, vars_list

        arr = storenames_selector.get_take_widgets()
        new_arr = []
        for i in range(len(arr[1])):
            for j in range(arr[1][i]):
                new_arr.append(arr[0][i])
        if len(new_arr) > 0:
            storenames = storenames_selector.get_cross_selector() + new_arr
        else:
            storenames = storenames_selector.get_cross_selector()
        storenames_selector.set_change_widgets(storenames)

        storenames_cache = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                storenames_cache = json.load(f)

        storenames_selector.configure_storenames(
            storename_dropdowns=storename_dropdowns,
            storename_textboxes=storename_textboxes,
            storenames=storenames,
            storenames_cache=storenames_cache,
        )

    # on clicking save button, following function is executed
    def save_button(event: object = None) -> None:
        global storenames
        storenames_config = storenames_selector.get_literal_input_2()
        select_location = storenames_selector.get_select_location()
        alert_message = _save(storenames_config=storenames_config, select_location=select_location)
        storenames_selector.set_alert_message(alert_message)
        storenames_selector.set_path(os.path.join(select_location, "storesList.csv"))

    # ------------------------------------------------------------------------------------------------------------------

    # Connect button callbacks
    button_name_to_onclick_fn = {
        "update_options": update_values,
        "save": save_button,
        "overwrite_button": overwrite_button_actions,
        "show_config_button": fetchValues,
    }
    storenames_selector.attach_callbacks(button_name_to_onclick_fn)
    storenames_selector.attach_run_name_watcher(run_name_input_changed)

    template.main.append(pn.Row(storenames_instructions.widget, storenames_selector.widget))

    return template


def build_storenames_page(
    inputParameters: dict[str, object], events: list[str], flags: list[str], folder_path: str
) -> None:
    """Write storesList.csv for one session, headlessly or via the Panel GUI.

    In headless mode (``storenames_map`` key present in ``inputParameters``)
    the mapping is written directly.  Otherwise a Panel GUI is launched in a
    browser so the user can assign semantic labels interactively.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; may contain ``storenames_map`` for
        headless operation.
    events : list of str
        Storename strings discovered from the acquisition files.
    flags : list of str
        Feature flags (e.g. ``"data_np_v2"``) passed to the GUI template.
    folder_path : str
        Absolute path to the session directory.
    """
    logger.debug("Saving stores list file.")

    # Headless path: if storenames_map provided, write storesList.csv without building the Panel UI
    storenames_map = inputParameters.get("storenames_map")
    if isinstance(storenames_map, dict) and len(storenames_map) > 0:
        run_name = inputParameters.get("runName") or None
        run_name_policy = inputParameters.get("runNamePolicy", "create")
        op = make_dir(folder_path, run_name=run_name, run_name_policy=run_name_policy)
        arr = np.asarray([list(storenames_map.keys()), list(storenames_map.values())], dtype=str)
        np.savetxt(os.path.join(op, "storesList.csv"), arr, delimiter=",", fmt="%s")
        logger.info(f"Storeslist file saved at {op}")
        logger.info("Storeslist : \n" + str(arr))
        return

    template = build_storenames_template(
        events, flags, folder_path, isosbestic_control=bool(inputParameters.get("isosbestic_control"))
    )

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = scanPortsAndFind(start_port=5000, end_port=5200)
    template.show(port=number)


def read_header(
    inputParameters: dict[str, object], num_ch: int, folder_path: str, headless: bool
) -> tuple[list[str], list[str]]:
    """Discover events and feature flags for a single session folder.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; checked for DANDI mode and NPM
        configuration.
    num_ch : int
        Number of photometry channels (used by NPM extractor discovery).
    folder_path : str
        Absolute path to the session directory.
    headless : bool
        When True, suppress interactive NPM GUI prompts.

    Returns
    -------
    events : list of str
        Unique event names discovered across all acquisition formats present.
    flags : list of str
        Feature flags (e.g. ``"data_np_v2"``) from all formats.
    """
    # DANDI mode bypasses local format detection — discover events via streaming
    if inputParameters.get("mode") == "dandi":
        dandi_uri = inputParameters["dandi_uri_map"][folder_path]
        events, flags = DandiNwbRecordingExtractor.discover_events_and_flags(folder_path=dandi_uri)
        return events, flags

    all_formats = detect_acquisition_formats(folder_path)

    # NPM GUI prompts (non-headless only) must run before NPM discovery so that
    # inputParameters is populated with split_events, time_units, etc.
    if "npm" in all_formats and not headless:
        multiple_event_ttls = NpmRecordingExtractor.has_multiple_event_ttls(folder_path=folder_path)
        responses = get_multi_event_responses(multiple_event_ttls)
        inputParameters["npm_split_events"] = responses

        ts_unit_needs, col_names_ts = NpmRecordingExtractor.needs_ts_unit(folder_path=folder_path, num_ch=num_ch)
        ts_units, npm_timestamp_column_names = get_timestamp_configuration(ts_unit_needs, col_names_ts)
        inputParameters["npm_time_units"] = ts_units if ts_units else None
        inputParameters["npm_timestamp_column_names"] = (
            npm_timestamp_column_names if npm_timestamp_column_names else None
        )

    events, flags = [], []
    existing_events = set()

    for format in sorted(all_formats):
        if format == "nwb":
            fmt_events, fmt_flags = NwbRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif format == "tdt":
            fmt_events, fmt_flags = TdtRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif format == "doric":
            fmt_events, fmt_flags = DoricRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif format == "csv":
            fmt_events, fmt_flags = CsvRecordingExtractor.discover_events_and_flags(folder_path=folder_path)
        elif format == "npm":
            fmt_events, fmt_flags = NpmRecordingExtractor.discover_events_and_flags(
                folder_path=folder_path, num_ch=num_ch, inputParameters=inputParameters
            )
        else:
            raise ValueError(f"Format not recognized: '{format}'. Expected one of 'nwb', 'tdt', 'csv', 'doric', 'npm'.")

        for event in fmt_events:
            if event not in existing_events:
                events.append(event)
                existing_events.add(event)

        for flag in fmt_flags:
            flags.append(flag)

    return events, flags


def orchestrate_storenames_page(inputParameters: dict[str, object]) -> None:
    """Run the step-2 storenames configuration for every selected session folder.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``folderNames``, ``abspath``,
        ``isosbestic_control``, and ``noChannels``.
    """
    inputParameters = inputParameters
    folderNames = inputParameters["folderNames"]
    isosbestic_control = inputParameters["isosbestic_control"]
    num_ch = inputParameters["noChannels"]
    headless = bool(os.environ.get("GUPPY_BASE_DIR"))

    logger.info(folderNames)

    try:
        for i in folderNames:
            folder_path = os.path.join(inputParameters["abspath"], i)
            events, flags = read_header(inputParameters, num_ch, folder_path, headless)
            build_storenames_page(inputParameters, events, flags, folder_path)
        logger.info("#" * 400)
    except Exception as e:
        logger.error(str(e))
        raise e
