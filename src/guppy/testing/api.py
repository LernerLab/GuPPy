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
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from guppy.analysis.standard_io import write_tonic_to_hdf5
from guppy.analysis.tonic import compute_tonic_means
from guppy.frontend.tonic_epochs import load_site_traces
from guppy.orchestration.export_nwb import orchestrate_export_nwb
from guppy.orchestration.home import build_homepage
from guppy.orchestration.import_custom_events import orchestrate_custom_events_page
from guppy.orchestration.metadata import orchestrate_metadata_page
from guppy.orchestration.preprocess import extractTsAndSignal, removeArtifactsFromSignal
from guppy.orchestration.psth import psthForEachStore
from guppy.orchestration.read_raw_data import orchestrate_read_raw_data
from guppy.orchestration.save_parameters import save_parameters
from guppy.orchestration.store_labeling import orchestrate_store_labeling_page
from guppy.orchestration.transients import executeFindFreqAndAmp
from guppy.orchestration.visualize import visualizeResults
from guppy.utils.utils import resolve_run_folders


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


def _normalize_group_selected_runs(
    group_selected_runs: dict[str, list[str]] | None,
    abs_group_folders: list[str],
) -> dict[str, list[str]]:
    """Validate group_selected_runs, allowing empty/None only when no group folders are selected."""
    if not abs_group_folders:
        if group_selected_runs:
            raise ValueError(
                f"group_selected_runs was provided but no group_folders were selected; got {group_selected_runs!r}."
            )
        return {}
    if group_selected_runs is None:
        raise ValueError(
            "group_selected_runs is required when group_folders is non-empty; "
            "every group session must specify at least one run name."
        )
    return _normalize_selected_runs(group_selected_runs, abs_group_folders, parameter_name="group_selected_runs")


def save_parameters_snapshot(*, base_dir: str, selected_folders: Iterable[str]) -> None:
    """
    Write ``GuPPyParamtersUsed.json`` into each selected folder (provenance snapshot).

    In the GUI this snapshot is now written automatically by each consuming step
    (steps 1–4); this helper exposes the same ``save_parameters`` write directly
    for tests and scripted provenance. It builds the form headlessly (using
    ``GUPPY_BASE_DIR`` to bypass the folder dialog), sets the FileSelector to
    ``selected_folders``, and calls ``save_parameters`` with the current
    parameters.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to analyze. All must share the
        same parent directory.

    Raises
    ------
    RuntimeError
        If the template does not expose the required testing hooks
        (``_hooks['getInputParameters']`` and ``_widgets['files_1']``).
    """
    os.environ["GUPPY_BASE_DIR"] = base_dir

    # Build the template headlessly
    template = build_homepage()

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
    headlessly (using ``GUPPY_BASE_DIR`` to bypass the folder dialog), sets the
    FileSelector to ``selected_folders``, injects ``custom_events_map``, and calls
    ``orchestrate_custom_events_page``. Each event is written as a
    GuPPy-compatible ``<name>.csv`` into its session folder. This is an unnumbered,
    optional step, so it is not part of the numbered ``step1``–``step5`` sequence.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside under this path.
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
    os.environ["GUPPY_BASE_DIR"] = base_dir

    template = build_homepage()

    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("build_homepage did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("build_homepage did not expose 'files_1' widget")

    template._widgets["files_1"].value = list(selected_folders)
    input_params = template._hooks["getInputParameters"]()
    input_params["custom_events_map"] = custom_events_map
    orchestrate_custom_events_page(input_params)


def step1(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    store_id_to_store_label: dict[str, str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    dandi_uri_map: dict[str, str] | None = None,
    run_name: str | None = None,
    run_name_policy: Literal["create", "overwrite"] = "create",
) -> None:
    """
    Run pipeline Step 1 (Label Stores) via the actual Panel-backed logic.

    This builds the Step 1 template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, injects the provided
    ``store_id_to_store_label``, and calls ``execute(inputParameters)`` from
    ``guppy.saveStoresList``. The execute() function is minimally augmented to
    support a headless branch when ``store_id_to_store_label`` is present, while leaving
    Panel behavior unchanged.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    store_id_to_store_label : dict[str, str]
        Mapping from raw store_ids (e.g., "Dv1A") to store labels
        (e.g., "control_DMS"). Insertion order is preserved.
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
        If validation fails (e.g., empty mapping, invalid directories, or parent
        mismatch).
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

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

    # Headless build: set base_dir and construct the template
    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject store_ids mapping for headless execution
    input_params["store_id_to_store_label"] = dict(store_id_to_store_label)

    # Inject run-name configuration (None falls back to legacy auto-incremented integer suffix)
    input_params["run_name"] = run_name
    input_params["run_name_policy"] = run_name_policy

    # Add npm parameters
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Inject DANDI mode and URI map for streaming
    if dandi_uri_map is not None:
        input_params["mode"] = "dandi"
        input_params["dandi_uri_map"] = dandi_uri_map
    else:
        input_params["mode"] = "local"

    # Call the underlying Step 1 executor (now headless-aware)
    orchestrate_store_labeling_page(input_params)


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

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.readTevTsq.readRawData(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

    # Headless build: set base_dir and construct the template
    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

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

    Validates the folder arguments, builds the Panel template with
    ``GUPPY_BASE_DIR`` set, retrieves ``getInputParameters()``, and overwrites the
    keys the caller specified.

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
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

    # Headless build: set base_dir and construct the template
    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

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

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.preprocess.extractTsAndSignal(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
        must reside directly under this path.
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
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch),
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
        must reside directly under this path.
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
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
        must reside directly under this path.
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
    average_for_group: bool = False,
    group_folders: list[str] | None = None,
    select_for_compute_psth: str = "z_score",
    select_for_transients: str = "z_score",
    number_of_cores: int = 1,
    bin_psth_trials: int = 0,
    use_time_or_trials: str = "Time (min)",
    time_for_lights_turn_on: float = 1.0,
    auc_units: str = "samples",
    selected_runs: dict[str, list[str]],
    group_selected_runs: dict[str, list[str]] | None = None,
) -> None:
    """
    Run pipeline Step 4 (PSTH Computation) via the Panel-backed logic, headlessly.

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.computePsth.psthForEachStore(input_params)`` that the
    UI invokes on its background worker thread. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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
    average_for_group : bool
        Whether to run group-level averaging across sessions instead of per-session PSTH
        computation. When ``True``, individual PSTH files must already exist in each session's
        output directory, and results are written to ``<base_dir>/average/``. Defaults to False.
    group_folders : list[str] | None
        Absolute paths to the session directories to include in group averaging. Only used
        when ``average_for_group`` is ``True``. Injected as ``group_session_folders`` in
        ``input_params``. Defaults to ``None`` (treated as empty list).
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

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

    # Headless build: set base_dir and construct the template
    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

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

    # Inject group analysis parameters
    input_params["averageForGroup"] = average_for_group
    abs_group_folders = [os.path.abspath(folder) for folder in group_folders] if group_folders else []
    input_params["group_session_folders"] = abs_group_folders

    # Per-session output-directory subset filter for individual + group analysis
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["group_selected_runs"] = _normalize_group_selected_runs(group_selected_runs, abs_group_folders)

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

    # Call the underlying Step 4 worker directly, as the GUI does
    psthForEachStore(input_params)

    # Also compute frequency/amplitude and transients occurrences (normally triggered by CLI main)
    executeFindFreqAndAmp(input_params)


def step5(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_name: str | None = None,
    npm_time_unit: str | None = None,
    npm_split_events: list[bool] | None = None,
    visualize_zscore_or_dff: str = "z_score",
    visualize_average_results: bool = False,
    group_folders: list[str] | None = None,
    selected_runs: dict[str, list[str]],
    group_selected_runs: dict[str, list[str]] | None = None,
) -> None:
    """
    Run pipeline Step 5 (Visualize Results) via the Panel-backed logic, headlessly.

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls
    ``visualizeResults(input_params)``. No GUI is spawned.

    Callers that need to suppress the web server (e.g. tests) should patch
    ``VisualizationDashboard.show`` before calling this function.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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
    visualize_average_results : bool
        When ``True``, visualize the group-averaged results from the ``average/``
        directory instead of the individual sessions. Injected as
        ``visualizeAverageResults``. Defaults to ``False``.
    group_folders : list[str] | None
        Session directories whose group average is being visualized; required when
        ``visualize_average_results`` is ``True``. Injected as ``group_session_folders``.

    Raises
    ------
    ValueError
        If validation fails (e.g., empty iterable, invalid directories, or parent mismatch).
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

    # Headless build: set base_dir and construct the template
    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

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

    # Inject group-average visualization selection
    input_params["visualizeAverageResults"] = visualize_average_results
    abs_group_folders = [os.path.abspath(folder) for folder in group_folders] if group_folders else []
    input_params["group_session_folders"] = abs_group_folders

    # Per-session output-directory subset filter for individual + group visualization
    input_params["selected_runs"] = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["group_selected_runs"] = _normalize_group_selected_runs(group_selected_runs, abs_group_folders)

    # Call the underlying Step 5 worker directly, as the GUI does
    visualizeResults(input_params)


def _build_headless_input_parameters(
    *, base_dir: str, selected_folders: Iterable[str]
) -> tuple[dict[str, object], list[str]]:
    """Validate the folder selection and return the input parameters a headless step runs with.

    Mirrors the production call chain: set ``GUPPY_BASE_DIR`` so the folder dialog is bypassed,
    build the homepage, select the sessions, then read the parameters back off the form.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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
    for session in abs_sessions:
        if not os.path.isdir(session):
            raise ValueError(f"Session path does not exist or is not a directory: {session}")
        parent = os.path.dirname(session)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {session!r}, expected {base_dir!r}"
            )

    os.environ["GUPPY_BASE_DIR"] = base_dir
    template = build_homepage()

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

    Builds one metadata page per selected session that needs one, without serving any of them:
    ``orchestrate_metadata_page`` skips ``template.show`` whenever ``GUPPY_BASE_DIR`` is set.
    Sessions GuPPy processed out of an NWB file are skipped entirely, as in the GUI.

    Because nothing is served, this writes no ``nwb_metadata.yaml``; it exercises the page
    construction and the session-source resolution. Tests that need a saved metadata file should
    build one with :func:`guppy.utils.nwb_metadata.build_metadata_dict` and
    :func:`guppy.utils.nwb_metadata.dump_yaml`.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
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

    # Call the underlying Step 6 worker directly, as the GUI does
    orchestrate_metadata_page(input_params)


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
        must reside directly under this path.
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
