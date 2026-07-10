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
from guppy.frontend.store_labeling_instructions import (
    StoreLabelingInstructions,
    StoreLabelingInstructionsNPM,
)
from guppy.frontend.store_labeling_selector import StoreLabelingSelector
from guppy.utils.utils import (
    NPM_PARAM_KEYS,
    discover_run_folders,
    is_headless,
    run_folder_for_run,
    validate_run_name,
    write_npm_params,
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
        return run_folder_for_run(filepath, run_name)

    i = 1
    while True:
        run_folder = run_folder_for_run(filepath, str(i))
        if not os.path.exists(run_folder):
            break
        i += 1
    return run_folder


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
            run_folder = run_folder_for_run(filepath, str(i))
            if not os.path.exists(run_folder):
                os.mkdir(run_folder)
                return run_folder
            i += 1

    validate_run_name(run_name)
    if run_name_policy not in ("create", "overwrite"):
        raise ValueError(f"run_name_policy must be 'create' or 'overwrite'; got {run_name_policy!r}.")
    run_folder = run_folder_for_run(filepath, run_name)
    if os.path.exists(run_folder):
        if run_name_policy == "create":
            raise ValueError(
                f"Output directory already exists: {run_folder!r}. "
                "Choose a different run_name or set run_name_policy='overwrite' to replace it."
            )
        shutil.rmtree(run_folder)
        logger.info(f"Cleared output directory for overwrite: {run_folder}")
    os.mkdir(run_folder)
    return run_folder


def _fetchValues(
    text: object,
    store_ids: list,
    store_id_dropdowns: dict,
    store_id_textboxes: dict,
    store_labeling_config: dict,
    isosbestic_control: bool = False,
) -> str:
    if not store_id_dropdowns or not len(store_ids) > 0:
        return "####Alert !! \n No store_ids selected."

    comboBoxValues, textBoxValues = [], []
    dropdown_keys = list(store_id_dropdowns.keys())

    # Get dropdown values
    for key in dropdown_keys:
        comboBoxValues.append(store_id_dropdowns[key].value)

    # Get textbox values (all store_ids always have a textbox)
    for key in dropdown_keys:
        textbox_value = store_id_textboxes[key].value or ""
        textBoxValues.append(textbox_value)

        # Validation: Check for whitespace
        if len(textbox_value.split()) > 1:
            return "####Alert !! \n Whitespace is not allowed in the text box entry."

        # Validation: Check for empty required fields
        dropdown_value = store_id_dropdowns[key].value
        if not textbox_value and dropdown_value in ["control", "signal", "event TTLs"]:
            return "####Alert !! \n One of the text box entry is empty."

    if len(comboBoxValues) != len(textBoxValues):
        return "####Alert !! \n Number of entries in combo box and text box should be same."

    store_labels = []
    signal_regions = []
    control_regions = []
    for i in range(len(comboBoxValues)):
        if comboBoxValues[i] == "control" or comboBoxValues[i] == "signal":
            if "_" in textBoxValues[i]:
                return "####Alert !! \n Please do not use underscore in region name."
            store_labels.append("{}_{}".format(comboBoxValues[i], textBoxValues[i]))
            if comboBoxValues[i] == "signal":
                signal_regions.append(textBoxValues[i])
            else:
                control_regions.append(textBoxValues[i])
        elif comboBoxValues[i] == "event TTLs":
            store_labels.append(textBoxValues[i])
        else:
            store_labels.append(comboBoxValues[i])

    # Validation: reject duplicate store_labels entries
    seen = set()
    duplicates = []
    for name in store_labels:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        return (
            "####Alert !! \n Duplicate name(s) in store_labels: {}. "
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

    store_labeling_config["store_ids"] = text.value
    store_labeling_config["store_labels"] = store_labels
    return "#### No alerts !!"


def _save(store_labeling_config: dict, select_location: str, npm_params: dict[str, object] | None = None) -> str:
    store_ids_array = np.asarray(store_labeling_config["store_ids"])
    store_labels_array = np.asarray(store_labeling_config["store_labels"])

    empty_indices = np.where(store_labels_array == "")[0].tolist()
    if empty_indices:
        detail = (
            f"Empty string in the list store_labels at index {empty_indices[0]} "
            f"(store_id {store_ids_array[empty_indices[0]]!r})."
            if len(empty_indices) == 1
            else (
                f"Empty strings in the list store_labels at {len(empty_indices)} indices: "
                f"{empty_indices} (store_ids {[str(store_ids_array[i]) for i in empty_indices]})."
            )
        )
        alert_message = f"#### Alert !! \n {detail} Provide a store label for each store_id."
        logger.error(detail)
        return alert_message

    if store_ids_array.shape[0] != store_labels_array.shape[0]:
        detail = (
            f"Length of list store_ids ({store_ids_array.shape[0]}) and store_labels "
            f"({store_labels_array.shape[0]}) "
            "is not equal; each store_id must be paired with exactly one store label."
        )
        alert_message = f"#### Alert !! \n {detail}"
        logger.error(detail)
        return alert_message

    if not os.path.exists(os.path.join(Path.home(), ".storesList.json")):
        store_id_to_store_labels = dict()

        for i in range(store_ids_array.shape[0]):
            if store_ids_array[i] in store_id_to_store_labels:
                store_id_to_store_labels[store_ids_array[i]].append(store_labels_array[i])
                store_id_to_store_labels[store_ids_array[i]] = list(set(store_id_to_store_labels[store_ids_array[i]]))
            else:
                store_id_to_store_labels[store_ids_array[i]] = [store_labels_array[i]]

        with open(os.path.join(Path.home(), ".storesList.json"), "w") as cache_file:
            json.dump(store_id_to_store_labels, cache_file, indent=4)
    else:
        with open(os.path.join(Path.home(), ".storesList.json")) as cache_file:
            store_id_to_store_labels = json.load(cache_file)

        for i in range(store_ids_array.shape[0]):
            if store_ids_array[i] in store_id_to_store_labels:
                store_id_to_store_labels[store_ids_array[i]].append(store_labels_array[i])
                store_id_to_store_labels[store_ids_array[i]] = list(set(store_id_to_store_labels[store_ids_array[i]]))
            else:
                store_id_to_store_labels[store_ids_array[i]] = [store_labels_array[i]]

        with open(os.path.join(Path.home(), ".storesList.json"), "w") as cache_file:
            json.dump(store_id_to_store_labels, cache_file, indent=4)

    store_array = np.asarray([store_ids_array, store_labels_array])
    logger.info(store_array)
    if os.path.exists(select_location):
        # Overwrite mode: clear all derived data from the previous run before saving the new store_array.
        shutil.rmtree(select_location)
        logger.info(f"Cleared output directory for overwrite: {select_location}")
    os.mkdir(select_location)

    np.savetxt(os.path.join(select_location, "storesList.csv"), store_array, delimiter=",", fmt="%s")
    if npm_params is not None:
        write_npm_params(run_folder=select_location, npm_params=npm_params)
    logger.info(f"Storeslist file saved at {select_location}")
    logger.info("Storeslist : \n" + str(store_array))
    return "#### No alerts !!"


def build_store_labeling_template(
    events: list[str],
    flags: list[str],
    folder_path: str,
    isosbestic_control: bool = False,
    channel_previews: dict[str, dict[str, np.ndarray]] | None = None,
    npm_params: dict[str, object] | None = None,
) -> pn.template.BootstrapTemplate:
    """Build and return the Label Stores GUI Panel template without serving it.

    Parameters
    ----------
    events : list of str
        store_id strings discovered from the data acquisition files.
    flags : list of str
        Feature flags (e.g. ``"data_np_v2"``) that control which instructions widget is shown.
    folder_path : str
        Absolute path to the session directory.
    isosbestic_control : bool, optional
        Whether isosbestic-control naming applies. Default is False.
    channel_previews : dict, optional
        NPM chev/chod/chpr channel traces (``{name: {"x", "y"}}``) used to render
        the NPM channel-preview plots. Required when the flags indicate NPM data.
    npm_params : dict, optional
        NPM decomposition parameters persisted next to ``storesList.csv`` on save
        so Step 2 can reproduce the decomposition. Set when flags indicate NPM data.

    Returns
    -------
    pn.template.BootstrapTemplate
        Fully configured Panel template ready to be served.
    """
    allnames = events

    template = pn.template.BootstrapTemplate(title="Label Stores GUI - {}".format(os.path.basename(folder_path)))

    if "data_np_v2" in flags or "data_np" in flags or "event_np" in flags:
        store_labeling_instructions = StoreLabelingInstructionsNPM(
            folder_path=folder_path, channel_previews=channel_previews or {}
        )
    else:
        store_labeling_instructions = StoreLabelingInstructions(folder_path=folder_path)
    store_labeling_selector = StoreLabelingSelector(allnames=allnames)

    # ------------------------------------------------------------------------------------------------------------------
    # onclick closure functions
    # on clicking overwrite_button, following function is executed
    def overwrite_button_actions(event: object) -> None:
        if event.new == "over_write_file":
            options = discover_run_folders(folder_path)
            store_labeling_selector.set_select_location_options(options=options)
        else:
            run_name = store_labeling_selector.get_run_name()
            options = [show_dir(folder_path, run_name=run_name or None)]
            store_labeling_selector.set_select_location_options(options=options)

    def run_name_input_changed(event: object) -> None:
        if store_labeling_selector.get_overwrite_mode() != "create_new_file":
            return
        run_name = event.new or None
        try:
            options = [show_dir(folder_path, run_name=run_name)]
        except ValueError as exc:
            store_labeling_selector.set_alert_message(f"####Alert !! \n {exc}")
            return
        store_labeling_selector.set_select_location_options(options=options)
        store_labeling_selector.set_alert_message("#### No alerts !!")

    def fetchValues(event: object) -> None:
        store_labeling_config = dict()
        alert_message = _fetchValues(
            text=store_labeling_selector.text,
            store_ids=store_labeling_selector.store_ids,
            store_id_dropdowns=store_labeling_selector.store_id_dropdowns,
            store_id_textboxes=store_labeling_selector.store_id_textboxes,
            store_labeling_config=store_labeling_config,
            isosbestic_control=isosbestic_control,
        )
        store_labeling_selector.set_alert_message(alert_message)
        store_labeling_selector.set_literal_input_2(store_labeling_config=store_labeling_config)

    # on clicking 'Select Stores' button, following function is executed
    def update_values(event: object) -> None:
        take_widgets = store_labeling_selector.get_take_widgets()
        expanded_store_ids = []
        for i in range(len(take_widgets[1])):
            for j in range(take_widgets[1][i]):
                expanded_store_ids.append(take_widgets[0][i])
        if len(expanded_store_ids) > 0:
            store_ids = store_labeling_selector.get_cross_selector() + expanded_store_ids
        else:
            store_ids = store_labeling_selector.get_cross_selector()
        store_labeling_selector.store_ids = store_ids
        store_labeling_selector.set_change_widgets(store_ids)

        store_id_to_store_labels = dict()
        if os.path.exists(os.path.join(Path.home(), ".storesList.json")):
            with open(os.path.join(Path.home(), ".storesList.json")) as f:
                store_id_to_store_labels = json.load(f)

        store_labeling_selector.configure_store_ids(store_id_to_store_labels=store_id_to_store_labels)

    # on clicking save button, following function is executed
    def save_button(event: object = None) -> None:
        store_labeling_config = store_labeling_selector.get_literal_input_2()
        select_location = store_labeling_selector.get_select_location()
        alert_message = _save(
            store_labeling_config=store_labeling_config, select_location=select_location, npm_params=npm_params
        )
        store_labeling_selector.set_alert_message(alert_message)
        store_labeling_selector.set_path(os.path.join(select_location, "storesList.csv"))

    # ------------------------------------------------------------------------------------------------------------------

    # Connect button callbacks
    button_name_to_onclick_fn = {
        "update_options": update_values,
        "save": save_button,
        "overwrite_button": overwrite_button_actions,
        "show_config_button": fetchValues,
    }
    store_labeling_selector.attach_callbacks(button_name_to_onclick_fn)
    store_labeling_selector.attach_run_name_watcher(run_name_input_changed)

    template.main.append(pn.Row(store_labeling_instructions.widget, store_labeling_selector.widget))

    return template


def build_store_labeling_page(
    inputParameters: dict[str, object], events: list[str], flags: list[str], folder_path: str
) -> None:
    """Write storesList.csv for one session, headlessly or via the Panel GUI.

    In headless mode (``store_id_to_store_label`` key present in ``inputParameters``)
    the mapping is written directly.  Otherwise a Panel GUI is launched in a
    browser so the user can assign store labels interactively.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; may contain ``store_id_to_store_label`` for
        headless operation.
    events : list of str
        store_id strings discovered from the acquisition files.
    flags : list of str
        Feature flags (e.g. ``"data_np_v2"``) passed to the GUI template.
    folder_path : str
        Absolute path to the session directory.
    """
    logger.debug("Saving stores list file.")

    # NPM decomposition is parameterized by interactive Step-1 choices; persist them
    # next to storesList.csv so Step 2 can reproduce the same in-memory decomposition.
    is_npm = "data_np_v2" in flags or "data_np" in flags or "event_np" in flags
    npm_params = {key: inputParameters.get(key) for key in NPM_PARAM_KEYS} if is_npm else None

    # Headless path: if store_id_to_store_label provided, write storesList.csv without building the Panel UI
    store_id_to_store_label = inputParameters.get("store_id_to_store_label")
    if isinstance(store_id_to_store_label, dict) and len(store_id_to_store_label) > 0:
        run_name = inputParameters.get("run_name") or None
        run_name_policy = inputParameters.get("run_name_policy", "create")
        run_folder = make_dir(folder_path, run_name=run_name, run_name_policy=run_name_policy)
        store_array = np.asarray(
            [list(store_id_to_store_label.keys()), list(store_id_to_store_label.values())], dtype=str
        )
        np.savetxt(os.path.join(run_folder, "storesList.csv"), store_array, delimiter=",", fmt="%s")
        if npm_params is not None:
            write_npm_params(run_folder=run_folder, npm_params=npm_params)
        logger.info(f"Storeslist file saved at {run_folder}")
        logger.info("Storeslist : \n" + str(store_array))
        return

    channel_previews = _compute_npm_channel_previews(inputParameters, folder_path) if is_npm else None

    template = build_store_labeling_template(
        events,
        flags,
        folder_path,
        isosbestic_control=bool(inputParameters.get("isosbestic_control")),
        channel_previews=channel_previews,
        npm_params=npm_params,
    )

    # creating widgets, adding them to template and showing a GUI on a new browser window
    number = scanPortsAndFind(start_port=5000, end_port=5200)
    template.show(port=number)


def _compute_npm_channel_previews(
    inputParameters: dict[str, object], folder_path: str
) -> dict[str, dict[str, np.ndarray]]:
    """Decompose the NPM session in memory and return chev/chod/chpr preview traces.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; supplies ``noChannels`` and the NPM
        configuration populated by :func:`read_header`.
    folder_path : str
        Absolute path to the NPM session directory.

    Returns
    -------
    dict
        Maps each chev/chod/chpr channel name to ``{"x": timestamps, "y": data}``.
    """
    extractor = NpmRecordingExtractor(
        folder_path=folder_path,
        num_ch=inputParameters["noChannels"],
        npm_timestamp_column_names=inputParameters.get("npm_timestamp_column_names"),
        npm_time_units=inputParameters.get("npm_time_units"),
        npm_split_events=inputParameters.get("npm_split_events"),
    )
    streams = extractor.decompose()
    previews = {}
    for name, stream in streams.items():
        if "data" in stream and ("chev" in name or "chod" in name or "chpr" in name):
            x = stream["timestamps"]
            y = stream["data"]
            # chod/chpr borrow chev's timestamps, which can be one sample shorter
            # than their own data (ragged interleaving); align lengths for plotting.
            n = min(len(x), len(y))
            previews[name] = {"x": x[:n], "y": y[:n]}
    return previews


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


def orchestrate_store_labeling_page(inputParameters: dict[str, object]) -> None:
    """Run the step-1 store_ids configuration for every selected session folder.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``session_folders``, ``abspath``,
        ``isosbestic_control``, and ``noChannels``.
    """
    inputParameters = inputParameters
    session_folders = inputParameters["session_folders"]
    isosbestic_control = inputParameters["isosbestic_control"]
    num_ch = inputParameters["noChannels"]
    headless = is_headless()

    logger.info(session_folders)

    try:
        for i in session_folders:
            folder_path = os.path.join(inputParameters["abspath"], i)
            events, flags = read_header(inputParameters, num_ch, folder_path, headless)
            build_store_labeling_page(inputParameters, events, flags, folder_path)
        logger.info("#" * 400)
    except Exception as e:
        logger.error(str(e))
        raise e
