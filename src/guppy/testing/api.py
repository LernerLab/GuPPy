"""
Python API for GuPPy pipeline steps.

Save Input Parameters (automatic provenance)
- Writes GuPPyParamtersUsed.json into each selected data folder.
- In the GUI this snapshot is now written automatically by each numbered step;
  ``save_parameters_snapshot`` exposes the same write directly for tests/scripts.

This module is intentionally minimal and non-invasive.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from guppy.analysis.standard_io import write_tonic_to_hdf5
from guppy.analysis.tonic import compute_tonic_means
from guppy.frontend.tonic_epochs import load_site_traces
from guppy.orchestration.export_nwb import orchestrate_export_nwb
from guppy.orchestration.group_analysis import run_group_analysis_step
from guppy.orchestration.group_labeling import build_group_labeling_page
from guppy.orchestration.home import build_homepage
from guppy.orchestration.import_custom_events import orchestrate_custom_events_page
from guppy.orchestration.metadata import build_metadata_templates
from guppy.orchestration.preprocess import extractTsAndSignal, removeArtifactsFromSignal
from guppy.orchestration.psth import psthForEachStore
from guppy.orchestration.read_raw_data import orchestrate_read_raw_data
from guppy.orchestration.save_parameters import save_parameters
from guppy.orchestration.store_labeling import (
    build_store_labeling_template,
    read_header,
)
from guppy.orchestration.transients import executeFindFreqAndAmp
from guppy.orchestration.visualize import visualizeResults
from guppy.utils.utils import resolve_run_folders, run_folder_for_run


def _validate_sessions_under_base_dir(*, abs_sessions: list[str], base_dir: str) -> None:
    """Validate that every session directory exists and lives somewhere under ``base_dir``.

    Sessions need not be siblings: ``base_dir`` only has to contain them, so a run can
    mix sessions kept in different sub-directories of a shared data root.

    Parameters
    ----------
    abs_sessions : list of str
        Absolute paths to the selected session directories.
    base_dir : str
        Absolute path to the root the FileSelector is initialized with.

    Raises
    ------
    ValueError
        If a session is missing, is not a directory, or lies outside ``base_dir``.
    """
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        if os.path.commonpath([base_dir, session]) != base_dir:
            raise ValueError(
                f"All selected_folders must live under base_dir. "
                f"Got session {session!r}, which is outside {base_dir!r}"
            )


def _normalize_selected_runs(
    selected_runs: dict[str, list[str]],
    abs_sessions: list[str],
    *,
    parameter_name: str = "selected_runs",
) -> dict[str, list[str]]:
    """Validate and absolute-ify session keys in a selected_runs mapping.

    Every session in ``abs_sessions`` must appear as a key with a non-empty
    list of run-name suffixes.
    """
    if not isinstance(selected_runs, dict):
        raise ValueError(
            f"{parameter_name} must be a dict[session_path, list[run_name]]; " f"got {type(selected_runs).__name__}."
        )
    normalized: dict[str, list[str]] = {}
    abs_sessions_set = set(abs_sessions)
    for session_key, run_names in selected_runs.items():
        absolute = os.path.abspath(session_key)
        if absolute not in abs_sessions_set:
            raise ValueError(
                f"{parameter_name} key {session_key!r} is not in selected_folders; "
                f"expected one of {sorted(abs_sessions_set)!r}."
            )
        if (
            not isinstance(run_names, list)
            or not run_names
            or not all(isinstance(run_name, str) and run_name for run_name in run_names)
        ):
            raise ValueError(
                f"{parameter_name}[{session_key!r}] must be a non-empty list of non-empty strings; "
                f"got {run_names!r}."
            )
        normalized[absolute] = list(run_names)
    missing = sorted(abs_sessions_set - normalized.keys())
    if missing:
        raise ValueError(
            f"{parameter_name} is missing entries for sessions {missing!r}; "
            "every selected session must specify at least one run name."
        )
    return normalized


def save_parameters_snapshot(*, base_dir: str, selected_folders: Iterable[str]) -> None:
    """
    Write ``GuPPyParamtersUsed.json`` into each selected folder (provenance snapshot).

    In the GUI this snapshot is now written automatically by each consuming step
    (steps 1–4); this helper exposes the same ``save_parameters`` write directly
    for tests and scripted provenance. It builds the form headlessly (rooting the
    file selectors at ``base_dir``), sets the FileSelector to
    ``selected_folders``, and calls ``save_parameters`` with the current
    parameters.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to analyze. All must reside
        somewhere under ``base_dir``.

    Raises
    ------
    RuntimeError
        If the template does not expose the required testing hooks
        (``_hooks['getInputParameters']`` and ``_widgets['files_1']``).
    """
    # Build the template headlessly
    template = build_homepage(start_path=base_dir)

    # Sanity checks: ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("build_homepage did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("build_homepage did not expose 'files_1' widget")

    # Select folders and write the parameter snapshot, mirroring the per-step auto-write.
    template._widgets["files_1"].value = list(selected_folders)
    save_parameters(inputParameters=template._hooks["getInputParameters"]())


def import_custom_events(
    *, base_dir: str, selected_folders: Iterable[str], custom_events_map: dict[str, dict[str, list[float]]]
) -> None:
    """Write custom event CSVs into sessions via the actual Panel-backed logic, headlessly.

    Mirrors the optional "Import Custom Events" GUI step: builds the form
    headlessly (rooting the file selectors at ``base_dir``), sets the
    FileSelector to ``selected_folders``, injects ``custom_events_map``, and calls
    ``orchestrate_custom_events_page``. Each event is written as a
    GuPPy-compatible ``<name>.csv`` into its session folder. This is an unnumbered,
    optional step, so it is not part of the numbered ``step1``–``step5`` sequence.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to write events into.
    custom_events_map : dict[str, dict[str, list[float]]]
        Mapping from session-folder name to ``{event_name: [timestamps]}``. Events
        are written only into sessions present as keys.

    Raises
    ------
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    template = build_homepage(start_path=base_dir)

    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("build_homepage did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("build_homepage did not expose 'files_1' widget")

    template._widgets["files_1"].value = list(selected_folders)
    input_params = template._hooks["getInputParameters"]()
    input_params["custom_events_map"] = custom_events_map
    orchestrate_custom_events_page(input_params)


def _parse_store_label(*, store_label: str) -> tuple[str, str]:
    """Split a store label into the Label Stores page's row type and name.

    Parameters
    ----------
    store_label : str
        Store label such as ``"signal_DMS"``, ``"control_DMS"``,
        ``"covariate_akinesia"``, or an event-TTL name.

    Returns
    -------
    tuple of (str, str)
        The row type (one of ``"signal"``, ``"control"``, ``"behavioral covariate"``,
        ``"event TTLs"``) and the name entered in that row's textbox. Event-TTL
        labels are returned verbatim as the name.
    """
    if store_label.startswith("signal_"):
        return "signal", store_label[len("signal_") :]
    if store_label.startswith("control_"):
        return "control", store_label[len("control_") :]
    if store_label.startswith("covariate_"):
        return "behavioral covariate", store_label[len("covariate_") :]
    return "event TTLs", store_label


def _raise_on_alert(*, selector: object) -> None:
    """Surface the Label Stores page's alert as a ValueError.

    Parameters
    ----------
    selector : StoreLabelingSelector
        The page's selector component; its alert pane holds ``"#### No alerts !!"``
        when the last action succeeded.

    Raises
    ------
    ValueError
        With the alert text, when the page reported a problem.
    """
    if selector.alert.object != "#### No alerts !!":
        raise ValueError(selector.alert.object)


def _drive_npm_configuration_form(
    *,
    template: object,
    npm_timestamp_column_name: str | None,
    npm_time_unit: str | None,
    npm_split_events: list[bool] | None,
) -> None:
    """Fill in and confirm the Label Stores page's NPM configuration form.

    Parameters
    ----------
    template : pn.template.BootstrapTemplate
        Template built by ``build_store_labeling_template`` for an NPM session.
    npm_timestamp_column_name : str or None
        Timestamp column to select; requires the session to offer more than one.
    npm_time_unit : str or None
        Time unit to select; ``None`` keeps the form's default.
    npm_split_events : list of bool or None
        Per-file split-events answers; ``None`` keeps the form's defaults.

    Raises
    ------
    ValueError
        When a supplied value has no form widget to receive it.
    """
    instructions = template._widgets["instructions"]
    multiple_event_ttls = instructions.multiple_event_ttls
    if npm_split_events is not None:
        if len(npm_split_events) != len(multiple_event_ttls):
            raise ValueError(
                f"npm_split_events has {len(npm_split_events)} entries but the session has "
                f"{len(multiple_event_ttls)} NPM files; provide one boolean per file."
            )
        for file_index, split in enumerate(npm_split_events):
            checkbox = instructions.split_event_checkboxes.get(file_index)
            if checkbox is not None:
                checkbox.value = bool(split)
            elif split:
                raise ValueError(
                    f"npm_split_events[{file_index}] is True but NPM file {file_index} has only one "
                    "event TTL, so there is nothing to split."
                )
    if npm_timestamp_column_name is not None:
        if instructions.timestamp_column_select is None:
            raise ValueError(
                f"npm_timestamp_column_name={npm_timestamp_column_name!r} was supplied but the "
                "session's NPM files offer only one timestamp column."
            )
        instructions.timestamp_column_select.value = npm_timestamp_column_name
    if npm_time_unit is not None:
        instructions.time_unit_select.value = npm_time_unit
    template._hooks["confirm_npm_configuration"]()


def _drive_store_labeling_page(
    *,
    template: object,
    folder_path: str,
    store_id_to_store_label: dict[str, str],
    run_name: str | None,
    run_name_policy: str,
) -> None:
    """Drive one session's Label Stores page to save storesList.csv.

    Sets the page's widgets to encode ``store_id_to_store_label``, clicks through
    the same callbacks the GUI uses, and raises the page's alert as a
    ``ValueError`` whenever an action fails.

    Parameters
    ----------
    template : pn.template.BootstrapTemplate
        Template built by ``build_store_labeling_template`` (NPM form already
        confirmed, when present).
    folder_path : str
        Absolute path to the session directory.
    store_id_to_store_label : dict of {str: str}
        Mapping from store_ids to store labels; insertion order becomes the
        storesList.csv column order.
    run_name : str or None
        Explicit run-name suffix, or ``None`` for the auto-incremented integer.
    run_name_policy : {"create", "overwrite"}
        Collision behavior for an explicit ``run_name``.
    """
    selector = template._widgets["selector"]

    available_store_ids = list(selector.cross_selector.options)
    missing = [store_id for store_id in store_id_to_store_label if store_id not in available_store_ids]
    if missing:
        raise ValueError(
            f"store_id_to_store_label contains store_ids not discovered in {folder_path!r}: {missing}. "
            f"Available store_ids: {available_store_ids}."
        )
    selector.cross_selector.value = list(store_id_to_store_label)
    selector.update_options.clicks += 1

    # Two passes: control rows reference their paired signal row by widget key, and those
    # options exist only after every signal row carries its type and name.
    signal_key_by_name: dict[str, str] = {}
    control_rows: list[tuple[str, str, str]] = []
    for i, (store_id, store_label) in enumerate(store_id_to_store_label.items()):
        widget_key = f"{store_id}_{i}"
        row_type, name = _parse_store_label(store_label=store_label)
        selector.store_id_dropdowns[widget_key].value = row_type
        if row_type == "control":
            control_rows.append((widget_key, store_label, name))
            continue
        selector.store_id_textboxes[widget_key].value = name
        if row_type == "signal":
            signal_key_by_name[name] = widget_key
    for widget_key, store_label, name in control_rows:
        signal_key = signal_key_by_name.get(name)
        if signal_key is None:
            raise ValueError(
                f"store label {store_label!r} has no matching 'signal_{name}' in store_id_to_store_label; "
                "a control channel inherits its name from its paired signal."
            )
        selector.store_id_control_refs[widget_key].value = signal_key

    selector.show_config_button.clicks += 1
    _raise_on_alert(selector=selector)

    target_run_folder = run_folder_for_run(folder_path, run_name) if run_name is not None else None
    if run_name_policy == "overwrite" and target_run_folder is not None and os.path.isdir(target_run_folder):
        selector.overwrite_button.clicked = "over_write_file"
        selector.select_location.value = target_run_folder
    else:
        # "create" policy, or overwriting a run folder that does not exist yet.
        selector.overwrite_button.clicked = "create_new_file"
        if run_name:
            selector.run_name.value = run_name
            _raise_on_alert(selector=selector)
        selector.select_location.value = selector.select_location.options[0]

    template._hooks["save_button"]()
    _raise_on_alert(selector=selector)


def step1(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    store_id_to_store_label: dict[str, str],
    isosbestic_control: bool = True,
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    dandi_uri_map: dict[str, str] | None = None,
    run_name: str | None = None,
    run_name_policy: Literal["create", "overwrite"] = "create",
) -> None:
    """
    Run pipeline Step 1 (Label Stores) by driving the real Panel pages headlessly.

    Builds the homepage template rooted at ``base_dir``, sets the FileSelector to
    ``selected_folders``, retrieves the full input parameters via
    ``getInputParameters()``, then for each session builds the Label Stores page
    and drives its widgets and save callback to encode ``store_id_to_store_label``
    — the exact code path the GUI's save button exercises.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    store_id_to_store_label : dict[str, str]
        Mapping from raw store_ids (e.g., "Dv1A") to store labels
        (e.g., "control_DMS"). Insertion order is preserved.
    isosbestic_control : bool
        Whether isosbestic-control naming applies; when True every signal in the
        mapping must have a paired control.
    npm_timestamp_column_name : str | None
        Timestamp column to use in NPM files that offer more than one. None to use the first.
    npm_time_unit : str | None
        Unit of the NPM session's timestamps (e.g., 'seconds', 'milliseconds'), applied to every
        file in the folder. None defaults to seconds.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty mapping, invalid directories, a session
        outside base_dir, or a mapping the page cannot express), or when the page
        reports an alert while being driven.
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    # Validate base_dir
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Validate selected_folders
    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    # Validate store_id_to_store_label
    if not isinstance(store_id_to_store_label, dict) or not store_id_to_store_label:
        raise ValueError("store_id_to_store_label must be a non-empty dict[str, str]")
    for store_id, store_label in store_id_to_store_label.items():
        if not isinstance(store_id, str) or not store_id.strip():
            raise ValueError(
                f"Invalid store_id key: {store_id!r}. Keys must be non-empty strings (the store id "
                "from the acquisition file)."
            )
        if not isinstance(store_label, str) or not store_label.strip():
            raise ValueError(
                f"Invalid store_label for store_id {store_id!r}: {store_label!r}. Values must be non-empty "
                "strings (the store label such as 'control_DMS' or 'signal_NAc')."
            )

    if run_name_policy not in ("create", "overwrite"):
        raise ValueError(f"run_name_policy must be 'create' or 'overwrite'; got {run_name_policy!r}.")

    # Headless build: construct the template rooted at base_dir
    homepage = build_homepage(start_path=base_dir)

    # Ensure hooks/widgets exposed
    if not hasattr(homepage, "_hooks") or "getInputParameters" not in homepage._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(homepage, "_widgets") or "files_1" not in homepage._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    homepage._widgets["files_1"].value = abs_sessions
    input_params = homepage._hooks["getInputParameters"]()

    input_params["isosbestic_control"] = isosbestic_control

    # Inject DANDI mode and URI map for streaming
    if dandi_uri_map is not None:
        input_params["mode"] = "dandi"
        input_params["dandi_uri_map"] = dandi_uri_map
    else:
        input_params["mode"] = "local"

    # Drive each session's Label Stores page exactly as the GUI does.
    num_ch = input_params["noChannels"]
    for session in abs_sessions:
        events, flags, npm_interactive = read_header(input_params, num_ch, session)
        template = build_store_labeling_template(
            events,
            flags,
            session,
            isosbestic_control=isosbestic_control,
            inputParameters=input_params,
            npm_interactive=npm_interactive,
        )
        if npm_interactive is not None:
            _drive_npm_configuration_form(
                template=template,
                npm_timestamp_column_name=npm_timestamp_column_name,
                npm_time_unit=npm_time_unit,
                npm_split_events=npm_split_events,
            )
        elif not (npm_timestamp_column_name is None and npm_time_unit is None and npm_split_events is None):
            raise ValueError(
                f"NPM parameters were supplied but session {session!r} contains no NPM data, "
                "so there is no NPM configuration form to receive them."
            )
        _drive_store_labeling_page(
            template=template,
            folder_path=session,
            store_id_to_store_label=store_id_to_store_label,
            run_name=run_name,
            run_name_policy=run_name_policy,
        )


def step2(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    number_of_cores: int = 1,
    dandi_uri_map: dict[str, str] | None = None,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 2 (Read Raw Data) via the actual Panel-backed logic, headlessly.

    This builds the template headlessly (rooting the file selectors at
    ``base_dir``), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.readTevTsq.readRawData(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_name : str | None
        Timestamp column to use in NPM files that offer more than one. None to use the first.
    npm_time_unit : str | None
        Unit of the NPM session's timestamps (e.g., 'seconds', 'milliseconds'), applied to every
        file in the folder. None defaults to seconds.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.
    number_of_cores : int
        Number of worker processes to use for parallel data reading. Defaults to ``1``
        (single-process) to avoid multiprocessing conflicts in test environments.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    # Validate base_dir
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Validate selected_folders
    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    # Headless build: construct the template rooted at base_dir
    template = build_homepage(start_path=base_dir)

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Override parallelism — default 1 keeps tests single-process
    input_params["numberOfCores"] = number_of_cores

    # Per-session output-directory subset filter — every session must have at least one run name.
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Inject DANDI mode and URI map for streaming
    if dandi_uri_map is not None:
        input_params["mode"] = "dandi"
        input_params["dandi_uri_map"] = dandi_uri_map
    else:
        input_params["mode"] = "local"

    # Call the underlying Step 2 worker directly, as the GUI does
    orchestrate_read_raw_data(input_params)


def _build_preprocess_input_parameters(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None,
    npm_time_unit: str | None,
    npm_split_events: list[bool] | None,
    combine_data: bool,
    zscore_method: str,
    baseline_window_start: int,
    baseline_window_end: int,
    isosbestic_control: bool,
    control_fit_method: Literal["IRWLS", "OLS"],
    control_fit_window_mode: Literal["full trace", "baseline epoch"],
    control_fit_window_start: int,
    control_fit_window_end: int,
    photobleaching_detrend: bool,
    time_for_lights_turn_on: float,
    selected_runs: dict[str, list[str]],
) -> tuple[dict[str, object], list[str], dict[str, list[str]]]:
    """
    Build the input-parameter dict the preprocessing workers consume, headlessly.

    Validates the folder arguments, builds the Panel template rooted at ``base_dir``,
    retrieves ``getInputParameters()``, and overwrites the keys the caller specified.

    Returns
    -------
    input_params : dict
        The populated input parameters.
    abs_sessions : list of str
        Absolute paths to the selected session directories.
    normalized_selected_runs : dict
        The per-session run-name filter.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    # Validate base_dir
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Validate selected_folders
    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    # Headless build: construct the template rooted at base_dir
    template = build_homepage(start_path=base_dir)

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Inject combine_data
    input_params["combine_data"] = combine_data

    # Inject the warm-up trim (bypasses the integer-only widget, so fractions are allowed)
    input_params["timeForLightsTurnOn"] = time_for_lights_turn_on

    # Inject z-score parameters
    input_params["zscore_method"] = zscore_method
    input_params["baselineWindowStart"] = baseline_window_start
    input_params["baselineWindowEnd"] = baseline_window_end

    # Inject isosbestic_control
    input_params["isosbestic_control"] = isosbestic_control

    # Inject control fitting method
    input_params["control_fit_method"] = control_fit_method

    # Inject control fit window parameters
    input_params["controlFitWindowMode"] = control_fit_window_mode
    input_params["controlFitWindowStart"] = control_fit_window_start
    input_params["controlFitWindowEnd"] = control_fit_window_end

    # Inject photobleaching detrending
    input_params["photobleaching_detrend"] = photobleaching_detrend

    # Per-session output-directory subset filter — every session must have at least one run name.
    normalized_selected_runs = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["selected_runs"] = normalized_selected_runs

    return input_params, abs_sessions, normalized_selected_runs


def step3(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    zscore_method: str = "standard z-score",
    baseline_window_start: int = 0,
    baseline_window_end: int = 0,
    isosbestic_control: bool = True,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
    control_fit_window_mode: Literal["full trace", "baseline epoch"] = "full trace",
    control_fit_window_start: int = 0,
    control_fit_window_end: int = 0,
    photobleaching_detrend: bool = False,
    time_for_lights_turn_on: float = 1.0,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 3 (Extract timestamps and signal) via the Panel-backed logic, headlessly.

    This builds the template headlessly (rooting the file selectors at
    ``base_dir``), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.preprocess.extractTsAndSignal(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_name : str | None
        Timestamp column to use in NPM files that offer more than one. None to use the first.
    npm_time_unit : str | None
        Unit of the NPM session's timestamps (e.g., 'seconds', 'milliseconds'), applied to every
        file in the folder. None defaults to seconds.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.
    combine_data : bool
        Whether to enable data combining logic in Step 3.
    zscore_method : str
        Z-score computation method. One of ``'standard z-score'``, ``'baseline z-score'``,
        or ``'modified z-score'``. Defaults to ``'standard z-score'``.
    baseline_window_start : int
        Start of the baseline window in seconds. Only used when ``zscore_method`` is
        ``'baseline z-score'``. Defaults to 0.
    baseline_window_end : int
        End of the baseline window in seconds. Only used when ``zscore_method`` is
        ``'baseline z-score'``. Defaults to 0.
    isosbestic_control : bool
        Whether a separate isosbestic control channel is present. When ``False``, GuPPy
        synthesizes a control channel from the signal. Defaults to ``True``.
    control_fit_method : str
        Regression method for fitting the control channel to the signal. One of
        ``'IRWLS'`` (robust, down-weights outliers) or ``'OLS'`` (ordinary least
        squares). Defaults to ``'IRWLS'``.
    control_fit_window_mode : str
        Control-fit mode. ``'full trace'`` (default) re-fits within each artifact-removal
        chunk. ``'baseline epoch'`` estimates fit coefficients once from the fit window
        (isosbestic control only) and applies them across the whole trace.
    control_fit_window_start : int
        Fit-window start in seconds. Only used when ``control_fit_window_mode`` is
        ``'baseline epoch'``. Defaults to 0.
    control_fit_window_end : int
        Fit-window end in seconds. Only used when ``control_fit_window_mode`` is
        ``'baseline epoch'``. Defaults to 0.
    photobleaching_detrend : bool
        When True, fit an exponential trend to the dF/F after the control channel is
        subtracted and remove it. Defaults to False. Requires ``isosbestic_control=True``.
    time_for_lights_turn_on : float
        Seconds of warm-up discarded from the start of the recording. Defaults to 1.0.
        Accepts fractional values; the GUI widget is integer-only.
    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    input_params, _, _ = _build_preprocess_input_parameters(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=combine_data,
        zscore_method=zscore_method,
        baseline_window_start=baseline_window_start,
        baseline_window_end=baseline_window_end,
        isosbestic_control=isosbestic_control,
        control_fit_method=control_fit_method,
        control_fit_window_mode=control_fit_window_mode,
        control_fit_window_start=control_fit_window_start,
        control_fit_window_end=control_fit_window_end,
        photobleaching_detrend=photobleaching_detrend,
        time_for_lights_turn_on=time_for_lights_turn_on,
        selected_runs=selected_runs,
    )

    # Call the underlying Step 3 worker directly, as the GUI does
    extractTsAndSignal(input_params)


def tonic_analysis(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    tonic_epochs: dict[str, pd.DataFrame],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run the optional Tonic Analysis step headlessly.

    Writes the per-recording-site epoch windows into every selected output directory and
    averages the Step-3 traces over them, exactly as the interactive page does when the
    user clicks Save. Requires Step 3 to have run.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    tonic_epochs : dict[str, pd.DataFrame]
        Mapping of recording-site name to an epoch-window DataFrame (columns ``label``,
        ``start``, ``end``) written as ``tonic_epochs_<recording_site>.csv``.
    selected_runs : dict[str, list[str]]
        Per-session output-directory subset filter.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir),
        or if an epoch window does not overlap its recording site's timespan.
    """
    input_params, abs_sessions, _ = _build_preprocess_input_parameters(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=combine_data,
        zscore_method="standard z-score",
        baseline_window_start=0,
        baseline_window_end=0,
        isosbestic_control=True,
        control_fit_method="IRWLS",
        control_fit_window_mode="full trace",
        control_fit_window_start=0,
        control_fit_window_end=0,
        photobleaching_detrend=False,
        time_for_lights_turn_on=1.0,
        selected_runs=selected_runs,
    )

    for run_folder in resolve_run_folders(abs_sessions, input_params):
        site_traces = load_site_traces(run_folder)
        for recording_site, epochs in tonic_epochs.items():
            epochs.to_csv(os.path.join(run_folder, f"tonic_epochs_{recording_site}.csv"), index=False)
            trace = site_traces[recording_site]
            write_tonic_to_hdf5(
                run_folder,
                compute_tonic_means(trace["y_zscore"], trace["y_dff"], trace["x"], epochs),
                recording_site,
            )


def select_artifact_windows(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    artifact_coords: dict[str, np.ndarray],
    artifact_removal_method: str = "replace with NaN",
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    zscore_method: str = "standard z-score",
    baseline_window_start: int = 0,
    baseline_window_end: int = 0,
    isosbestic_control: bool = True,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
    control_fit_window_mode: Literal["full trace", "baseline epoch"] = "full trace",
    control_fit_window_start: int = 0,
    control_fit_window_end: int = 0,
    photobleaching_detrend: bool = False,
    time_for_lights_turn_on: float = 1.0,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run the optional Select Artifact Windows step headlessly.

    Writes the keep-window coordinates into every selected output directory and records
    the removal method into each directory's ``GuPPyParamtersUsed.json``, exactly as the
    interactive page does when the user clicks Save.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    artifact_coords : dict[str, np.ndarray]
        Mapping of recording-site name to the keep-window coordinates array (shape
        ``(2M, 2)``, time in column 0) written as
        ``coordsForPreProcessing_<recording_site>.npy``.
    artifact_removal_method : str
        Removal method recorded for these runs; ``'concatenate'`` or ``'replace with NaN'``.
    selected_runs : dict[str, list[str]]
        Per-session output-directory subset filter.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    """
    input_params, abs_sessions, normalized_selected_runs = _build_preprocess_input_parameters(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=combine_data,
        zscore_method=zscore_method,
        baseline_window_start=baseline_window_start,
        baseline_window_end=baseline_window_end,
        isosbestic_control=isosbestic_control,
        control_fit_method=control_fit_method,
        control_fit_window_mode=control_fit_window_mode,
        control_fit_window_start=control_fit_window_start,
        control_fit_window_end=control_fit_window_end,
        photobleaching_detrend=photobleaching_detrend,
        time_for_lights_turn_on=time_for_lights_turn_on,
        selected_runs=selected_runs,
    )

    for run_folder in resolve_run_folders(abs_sessions, input_params):
        for recording_site, coords in artifact_coords.items():
            np.save(os.path.join(run_folder, f"coordsForPreProcessing_{recording_site}.npy"), coords)

    save_parameters(inputParameters=input_params, artifacts_removal_method=artifact_removal_method)


def remove_artifacts(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    zscore_method: str = "standard z-score",
    baseline_window_start: int = 0,
    baseline_window_end: int = 0,
    isosbestic_control: bool = True,
    control_fit_method: Literal["IRWLS", "OLS"] = "IRWLS",
    control_fit_window_mode: Literal["full trace", "baseline epoch"] = "full trace",
    control_fit_window_start: int = 0,
    control_fit_window_end: int = 0,
    photobleaching_detrend: bool = False,
    time_for_lights_turn_on: float = 1.0,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run the optional Remove Artifacts step headlessly.

    Re-runs timestamp correction and z-score against the windows saved by
    :func:`select_artifact_windows`, then excises the artifacts. The removal method is
    read from each run's recorded provenance.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    selected_runs : dict[str, list[str]]
        Per-session output-directory subset filter.

    Raises
    ------
    ValueError
        If validation fails, or if any selected run has no saved artifact windows.
    """
    input_params, _, _ = _build_preprocess_input_parameters(
        base_dir=base_dir,
        selected_folders=selected_folders,
        npm_timestamp_column_name=npm_timestamp_column_name,
        npm_time_unit=npm_time_unit,
        npm_split_events=npm_split_events,
        combine_data=combine_data,
        zscore_method=zscore_method,
        baseline_window_start=baseline_window_start,
        baseline_window_end=baseline_window_end,
        isosbestic_control=isosbestic_control,
        control_fit_method=control_fit_method,
        control_fit_window_mode=control_fit_window_mode,
        control_fit_window_start=control_fit_window_start,
        control_fit_window_end=control_fit_window_end,
        photobleaching_detrend=photobleaching_detrend,
        time_for_lights_turn_on=time_for_lights_turn_on,
        selected_runs=selected_runs,
    )

    removeArtifactsFromSignal(input_params)


def step4(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    compute_corr: bool = False,
    use_transients_as_events: bool = False,
    select_for_compute_psth: str = "z_score",
    select_for_transients: str = "z_score",
    number_of_cores: int = 1,
    bin_psth_trials: int = 0,
    use_time_or_trials: str = "Time (min)",
    time_for_lights_turn_on: float = 1.0,
    auc_units: str = "samples",
    compute_binned_metrics: bool = False,
    binned_metrics_width: int = 120,
    compute_psth_significance: bool = False,
    psth_comparisons: Iterable[tuple[str, str]] = (),
    psth_significance_alpha: float = 0.05,
    psth_bootstrap_resamples: int = 1000,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 4 (PSTH Computation) via the Panel-backed logic, headlessly.

    This builds the template headlessly (rooting the file selectors at
    ``base_dir``), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.computePsth.psthForEachStore(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_name : str | None
        Timestamp column to use in NPM files that offer more than one. None to use the first.
    npm_time_unit : str | None
        Unit of the NPM session's timestamps (e.g., 'seconds', 'milliseconds'), applied to every
        file in the folder. None defaults to seconds.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.
    combine_data : bool
        Whether to enable combined-session processing mode in Step 4. Defaults to False.
    compute_corr : bool
        Whether to compute cross-correlation between signals. Defaults to False.
    use_transients_as_events : bool
        Whether to use each recording site's detected transients as its event timestamps.
        Defaults to False.
    select_for_compute_psth : str
        Signal type to use for PSTH computation. One of ``'z_score'``, ``'dff'``, or
        ``'Both'``. Defaults to ``'z_score'``.
    select_for_transients : str
        Signal type to use for transient detection. One of ``'z_score'``, ``'dff'``, or
        ``'Both'``. Defaults to ``'z_score'``.
    number_of_cores : int
        Number of worker processes for PSTH and transient computations. Defaults to ``1``
        (single-process) to avoid multiprocessing conflicts in test environments.
    bin_psth_trials : int
        Number of time minutes or trials to bin together for PSTH computation. ``0`` disables
        binning (the default). When positive, ``use_time_or_trials`` controls the interpretation.
    use_time_or_trials : str
        Whether ``bin_psth_trials`` is interpreted as a time window in minutes (``'Time (min)'``)
        or a number of trials (``'# of trials'``). Only meaningful when ``bin_psth_trials > 0``.
        Defaults to ``'Time (min)'``.
    time_for_lights_turn_on : float
        Seconds of warm-up discarded from the start of the recording. Must match the value
        passed to :func:`step3`. Defaults to 1.0.
    auc_units : str
        Integration spacing for the peak/AUC areas. ``'samples'`` (the default) integrates
        with one-sample spacing; ``'seconds'`` integrates against the PSTH time axis, giving
        areas in z-score (or ΔF/F) × seconds.
    compute_binned_metrics : bool
        Whether to divide the session into fixed-width time bins and write the per-bin mean
        z-score, mean ΔF/F and transient counts. Defaults to False.
    binned_metrics_width : int
        Width of those bins in seconds. Only meaningful when ``compute_binned_metrics`` is
        ``True``. Defaults to 120.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    # Validate base_dir
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Validate selected_folders
    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    # Headless build: construct the template rooted at base_dir
    template = build_homepage(start_path=base_dir)

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Inject combine_data
    input_params["combine_data"] = combine_data

    # Inject the warm-up trim; must match the value used in step 3 so the PSTH anchor
    # lines up with the trimmed signal.
    input_params["timeForLightsTurnOn"] = time_for_lights_turn_on

    # Inject cross-correlation flag
    input_params["computeCorr"] = compute_corr

    # Inject the spontaneous-activity flag
    input_params["useTransientsAsEvents"] = use_transients_as_events

    # Per-session output-directory subset filter
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Inject signal-type selection parameters
    input_params["selectForComputePsth"] = select_for_compute_psth
    input_params["selectForTransientsComputation"] = select_for_transients

    # Override parallelism — default 1 keeps tests single-process
    input_params["numberOfCores"] = number_of_cores

    # Inject PSTH binning parameters
    input_params["bin_psth_trials"] = bin_psth_trials
    input_params["use_time_or_trials"] = use_time_or_trials

    # Inject the peak/AUC integration spacing
    input_params["auc_units"] = auc_units

    # Inject the whole-session binned metrics parameters
    input_params["computeBinnedMetrics"] = compute_binned_metrics
    input_params["binnedMetricsWidth"] = binned_metrics_width

    # Inject the PSTH significance parameters
    input_params["computePsthSignificance"] = compute_psth_significance
    input_params["psthComparisonsA"] = [pair[0] for pair in psth_comparisons]
    input_params["psthComparisonsB"] = [pair[1] for pair in psth_comparisons]
    input_params["psthSignificanceAlpha"] = psth_significance_alpha
    input_params["psthBootstrapResamples"] = psth_bootstrap_resamples

    # Call the underlying Step 4 workers directly, in the order the GUI runs them:
    # transients first, so their timestamps are available as events for the PSTH.
    executeFindFreqAndAmp(input_params)
    psthForEachStore(input_params)


def label_groups(
    *,
    member_run_folders: Iterable[str],
    destination_directory: str,
    group_name: str,
) -> None:
    """Define a group headlessly, as the Label Groups page does.

    Writes only ``group_members.json`` into ``<destination_directory>/<group_name>_group/``.
    The group holds no averaged results until :func:`group_analysis` runs against it.

    Parameters
    ----------
    member_run_folders : iterable of str
        Output (run) directories to record as the group's members. Each must already
        hold Step-4 results.
    destination_directory : str
        Directory the group output directory is written into.
    group_name : str
        Name of the group. Becomes the ``<group_name>_group`` directory name.

    Raises
    ------
    ValueError
        When the page rejects the group name or member selection (raised from
        the page's alert).
    """
    destination = os.path.abspath(destination_directory)
    page = build_group_labeling_page(inputParameters={"abspath": destination, "selected_group_folders": []})
    page.group_name.value = group_name
    page.destination_selector.value = [destination]
    page.members_selector.value = [os.path.abspath(run_folder) for run_folder in member_run_folders]
    page.save.clicks += 1
    # The page reports validation failures on its alert instead of raising.
    if page.alert.visible:
        raise ValueError(page.alert.object)


def group_analysis(
    *,
    base_dir: str,
    selected_group_folders: Iterable[str],
    select_for_compute_psth: str = "z_score",
    select_for_transients: str = "z_score",
    use_transients_as_events: bool = False,
    compute_corr: bool = False,
    compute_psth_significance: bool = False,
    psth_comparisons: Iterable[tuple[str, str]] = (),
    psth_significance_alpha: float = 0.05,
    psth_bootstrap_resamples: int = 1000,
) -> None:
    """Run the Group Analysis step headlessly against already-defined groups.

    Averages each group's recorded member runs into its own directory. Define the groups
    first with :func:`label_groups`.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector.
    selected_group_folders : iterable of str
        Group output directories to average into.
    select_for_compute_psth : str
        Which PSTH metric to average: ``"z_score"``, ``"dff"`` or ``"Both"``.
    select_for_transients : str
        Which metric's transient results to combine.
    use_transients_as_events : bool
        Whether transient trains stand in for external event TTLs.
    compute_corr : bool
        Whether cross-correlation outputs are combined.
    """
    template = build_homepage(start_path=base_dir)

    absolute_groups = [os.path.abspath(folder) for folder in selected_group_folders]
    template._widgets["group_folders_selector"].value = absolute_groups
    input_params = template._hooks["getInputParameters"]()

    input_params["selected_group_folders"] = absolute_groups
    input_params["selectForComputePsth"] = select_for_compute_psth
    input_params["selectForTransientsComputation"] = select_for_transients
    input_params["useTransientsAsEvents"] = use_transients_as_events
    input_params["computeCorr"] = compute_corr
    input_params["computePsthSignificance"] = compute_psth_significance
    input_params["psthComparisonsA"] = [pair[0] for pair in psth_comparisons]
    input_params["psthComparisonsB"] = [pair[1] for pair in psth_comparisons]
    input_params["psthSignificanceAlpha"] = psth_significance_alpha
    input_params["psthBootstrapResamples"] = psth_bootstrap_resamples

    run_group_analysis_step(input_params)


def step5(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    visualize_zscore_or_dff: str = "z_score",
    use_transients_as_events: bool = False,
    select_for_transients: str = "z_score",
    selected_group_folders: list[str] | None = None,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 5 (Visualize Results) via the Panel-backed logic, headlessly.

    This builds the template headlessly (rooting the file selectors at
    ``base_dir``), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls
    ``visualizeResults(input_params)``. No GUI is spawned.

    Callers that need to suppress the web server (e.g. tests) should patch
    ``VisualizationDashboard.show`` before calling this function.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_name : str | None
        Timestamp column to use in NPM files that offer more than one. None to use the first.
    npm_time_unit : str | None
        Unit of the NPM session's timestamps (e.g., 'seconds', 'milliseconds'), applied to every
        file in the folder. None defaults to seconds.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files. None if not applicable.
    visualize_zscore_or_dff : str
        Signal type to visualize. One of ``'z_score'`` or ``'dff'``. Defaults to ``'z_score'``.
    use_transients_as_events : bool
        Whether step 4 used each recording site's detected transients as its event
        timestamps; must match the value step 4 ran with. Defaults to False.
    select_for_transients : str
        Metric the transient detector ran on in step 4; must match the value step 4 ran
        with. One of ``'z_score'``, ``'dff'`` or ``'Both'``. Defaults to ``'z_score'``.
    selected_group_folders : list[str] | None
        Group output directories to visualize alongside the selected session runs.
        Injected as ``selected_group_folders``. Defaults to ``None`` (treated as empty list).

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or a session outside base_dir).
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    # Validate base_dir
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Validate selected_folders
    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    # Headless build: construct the template rooted at base_dir
    template = build_homepage(start_path=base_dir)

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Inject visualization signal-type selection
    input_params["visualize_zscore_or_dff"] = visualize_zscore_or_dff

    # Inject the spontaneous-activity flag, which decides whether the transient event
    # PSTHs step 4 computed are offered in the dashboard
    input_params["useTransientsAsEvents"] = use_transients_as_events
    input_params["selectForTransientsComputation"] = select_for_transients

    # Groups to visualize alongside the selected session runs
    input_params["selected_group_folders"] = (
        [os.path.abspath(folder) for folder in selected_group_folders] if selected_group_folders else []
    )

    # Per-session output-directory subset filter
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Call the underlying Step 5 worker directly, as the GUI does
    visualizeResults(input_params)


def _build_headless_input_parameters(
    *, base_dir: str, selected_folders: Iterable[str]
) -> tuple[dict[str, object], list[str]]:
    """Validate the folder selection and return the input parameters a headless step runs with.

    Mirrors the production call chain: build the homepage rooted at ``base_dir``,
    select the sessions, then read the parameters back off the form.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.

    Returns
    -------
    input_params : dict
        The full pipeline input parameters, ready for per-step injection.
    abs_sessions : list of str
        The selected session paths, made absolute.

    Raises
    ------
    ValueError
        If ``base_dir`` or any selected folder is missing, or a session's parent is not ``base_dir``.
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("base_dir must be a non-empty string")
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir does not exist or is not a directory: {base_dir}")

    sessions = list(selected_folders or [])
    if not sessions:
        raise ValueError("selected_folders must be a non-empty iterable of session directories")
    abs_sessions = [os.path.abspath(session) for session in sessions]
    _validate_sessions_under_base_dir(abs_sessions=abs_sessions, base_dir=base_dir)

    template = build_homepage(start_path=base_dir)

    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    template._widgets["files_1"].value = abs_sessions
    return template._hooks["getInputParameters"](), abs_sessions


def step6(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    combine_data: bool = False,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 6 (Input Metadata) via the Panel-backed logic, headlessly.

    Builds one metadata page per selected session that needs one, without serving any of them.
    Sessions GuPPy processed out of an NWB file are skipped entirely, as in the GUI.

    Because nothing is served, this writes no ``nwb_metadata.yaml``; it exercises the page
    construction and the session-source resolution. Tests that need a saved metadata file should
    build one with :func:`guppy.utils.nwb_metadata.build_metadata_dict` and
    :func:`guppy.utils.nwb_metadata.dump_yaml`.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    combine_data : bool
        The pipeline's ``combine_data`` setting. Step 6 refuses ``True``. Defaults to ``False``.
    selected_runs : dict of {str: list of str}
        Per-session output-directory subset filter, one entry per selected session.

    Raises
    ------
    ValueError
        If validation fails, if ``combine_data`` is ``True``, or if a session's source
        cannot be resolved.
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    input_params, abs_sessions = _build_headless_input_parameters(base_dir=base_dir, selected_folders=selected_folders)
    input_params["combine_data"] = combine_data
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Build the pages the GUI would serve, without serving them
    build_metadata_templates(inputParameters=input_params)


def step7(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    combine_data: bool = False,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 7 (Export to NWB) via the Panel-backed logic, headlessly.

    Writes one ``.nwb`` file per selected ``(session, run)`` into that run's output directory,
    picking up the session's ``nwb_metadata.yaml`` overlay when one is present.

    One failed session is skipped without aborting the rest of the batch, matching the GUI. Since
    no progress channel is bound headlessly, those failures are reported only through the log --
    callers that need to assert on them should bind a ``StepProgress`` first.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside somewhere under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    combine_data : bool
        The pipeline's ``combine_data`` setting. Step 7 refuses ``True``. Defaults to ``False``.
    selected_runs : dict of {str: list of str}
        Per-session output-directory subset filter, one entry per selected session.

    Raises
    ------
    ValueError
        If validation fails, if ``combine_data`` is ``True``, or if a selected run was
        processed with the unsupported ``concatenate`` artifact-removal method.
    RuntimeError
        If the template does not expose the required testing hooks/widgets.
    """
    input_params, abs_sessions = _build_headless_input_parameters(base_dir=base_dir, selected_folders=selected_folders)
    input_params["combine_data"] = combine_data
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Call the underlying Step 7 worker directly, as the GUI does
    orchestrate_export_nwb(input_params)
