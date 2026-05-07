"""
Python API for GuPPy pipeline steps.

Step 1: Save Input Parameters
- Writes GuPPyParamtersUsed.json into each selected data folder.
- Mirrors the Panel UI's Step 1 behavior without invoking any UI by default.

This module is intentionally minimal and non-invasive.
"""

from __future__ import annotations

import os
from typing import Iterable, Literal

import numpy as np

from guppy.orchestration.home import build_homepage
from guppy.orchestration.preprocess import extractTsAndSignal
from guppy.orchestration.psth import psthForEachStorename
from guppy.orchestration.read_raw_data import orchestrate_read_raw_data
from guppy.orchestration.storenames import orchestrate_storenames_page
from guppy.orchestration.transients import executeFindFreqAndAmp
from guppy.orchestration.visualize import visualizeResults
from guppy.utils.utils import select_output_dirs


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
        if not isinstance(run_names, list) or not run_names or not all(isinstance(r, str) and r for r in run_names):
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


def step1(*, base_dir: str, selected_folders: Iterable[str]) -> None:
    """
    Run pipeline Step 1 (Save Input Parameters) via the Panel logic.

    This calls the exact ``onclickProcess`` function defined in
    ``savingInputParameters()``, in headless mode. The ``GUPPY_BASE_DIR``
    environment variable is used to bypass the Tk folder selection dialog.
    The function programmatically sets the FileSelector value to
    ``selected_folders`` and triggers the underlying callback that writes
    ``GuPPyParamtersUsed.json`` into each selected folder.

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
        If the ``savingInputParameters`` template does not expose the required
        testing hooks (``_hooks['onclickProcess']`` and ``_widgets['files_1']``).
    """
    os.environ["GUPPY_BASE_DIR"] = base_dir

    # Build the template headlessly
    template = build_homepage()

    # Sanity checks: ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "onclickProcess" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'onclickProcess' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and trigger actual step-1 logic
    template._widgets["files_1"].value = list(selected_folders)
    template._hooks["onclickProcess"]()


def step2(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    storenames_map: dict[str, str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
    npm_split_events: list[bool] | None = None,
    dandi_uri_map: dict[str, str] | None = None,
    run_name: str | None = None,
    run_name_policy: Literal["create", "overwrite"] = "create",
) -> None:
    """
    Run pipeline Step 2 (Save Storenames) via the actual Panel-backed logic.

    This builds the Step 2 template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, injects the provided
    ``storenames_map``, and calls ``execute(inputParameters)`` from
    ``guppy.saveStoresList``. The execute() function is minimally augmented to
    support a headless branch when ``storenames_map`` is present, while leaving
    Panel behavior unchanged.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    storenames_map : dict[str, str]
        Mapping from raw storenames (e.g., "Dv1A") to semantic names
        (e.g., "control_DMS"). Insertion order is preserved.
    npm_timestamp_column_names : list[str | None] | None
        List of timestamp column names for NPM files, one per CSV file. None if not applicable.
    npm_time_units : list[str] | None
        List of time units for NPM files, one per CSV file (e.g., 'seconds', 'milliseconds'). None if not applicable.
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
    abs_sessions = [os.path.abspath(s) for s in sessions]
    for s in abs_sessions:
        if not os.path.isdir(s):
            raise ValueError(f"Session path does not exist or is not a directory: {s}")
        parent = os.path.dirname(s)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {s!r}, expected {base_dir!r}"
            )

    # Validate storenames_map
    if not isinstance(storenames_map, dict) or not storenames_map:
        raise ValueError("storenames_map must be a non-empty dict[str, str]")
    for k, v in storenames_map.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(
                f"Invalid storename key: {k!r}. Keys must be non-empty strings (the raw store name "
                "from the acquisition file)."
            )
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"Invalid semantic name for key {k!r}: {v!r}. Values must be non-empty strings "
                "(the semantic label such as 'control_DMS' or 'signal_NAc')."
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

    # Inject storenames mapping for headless execution
    input_params["storenames_map"] = dict(storenames_map)

    # Inject run-name configuration (None falls back to legacy auto-incremented integer suffix)
    input_params["runName"] = run_name
    input_params["runNamePolicy"] = run_name_policy

    # Add npm parameters
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Inject DANDI mode and URI map for streaming
    if dandi_uri_map is not None:
        input_params["mode"] = "dandi"
        input_params["dandi_uri_map"] = dandi_uri_map
    else:
        input_params["mode"] = "local"

    # Call the underlying Step 2 executor (now headless-aware)
    orchestrate_storenames_page(input_params)


def step3(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
    npm_split_events: list[bool] | None = None,
    number_of_cores: int = 1,
    dandi_uri_map: dict[str, str] | None = None,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 3 (Read Raw Data) via the actual Panel-backed logic, headlessly.

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.readTevTsq.readRawData(input_params)`` that the
    UI normally launches via subprocess. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_names : list[str | None] | None
        List of timestamp column names for NPM files, one per CSV file. None if not applicable.
    npm_time_units : list[str] | None
        List of time units for NPM files, one per CSV file (e.g., 'seconds', 'milliseconds'). None if not applicable.
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
    abs_sessions = [os.path.abspath(s) for s in sessions]
    for s in abs_sessions:
        if not os.path.isdir(s):
            raise ValueError(f"Session path does not exist or is not a directory: {s}")
        parent = os.path.dirname(s)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {s!r}, expected {base_dir!r}"
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
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Override parallelism — default 1 keeps tests single-process
    input_params["numberOfCores"] = number_of_cores

    # Per-session output-directory subset filter — every session must have at least one run name.
    input_params["selectedOutputs"] = _normalize_selected_runs(selected_runs, abs_sessions)

    # Inject DANDI mode and URI map for streaming
    if dandi_uri_map is not None:
        input_params["mode"] = "dandi"
        input_params["dandi_uri_map"] = dandi_uri_map
    else:
        input_params["mode"] = "local"

    # Call the underlying Step 3 worker directly (no subprocess)
    orchestrate_read_raw_data(input_params)


def step4(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
    npm_split_events: list[bool] | None = None,
    combine_data: bool = False,
    remove_artifacts: bool = False,
    artifact_removal_method: str | None = None,
    artifact_coords: dict[str, np.ndarray] | None = None,
    zscore_method: str = "standard z-score",
    baseline_window_start: int = 0,
    baseline_window_end: int = 0,
    isosbestic_control: bool = True,
    selected_runs: dict[str, list[str]],
) -> None:
    """
    Run pipeline Step 4 (Extract timestamps and signal) via the Panel-backed logic, headlessly.

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.preprocess.extractTsAndSignal(input_params)`` that the
    UI normally launches via subprocess. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_names : list[str | None] | None
        List of timestamp column names for NPM files, one per CSV file. None if not applicable.
    npm_time_units : list[str] | None
        List of time units for NPM files, one per CSV file (e.g., 'seconds', 'milliseconds'). None if not applicable.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.
    combine_data : bool
        Whether to enable data combining logic in Step 4.
    remove_artifacts : bool
        Whether to run artifact removal.
    artifact_removal_method : str | None
        Artifact removal method to use ('concatenate' or 'replace with NaN').
        Only applied when ``remove_artifacts`` is True.
    artifact_coords : dict[str, np.ndarray] | None
        Mapping of pair name to coordinates array (shape ``(N_clicks, 2)``, x/time in
        column 0) to write as ``coordsForPreProcessing_<pair_name>.npy`` into every
        ``_output_*`` directory before artifact removal runs. Bypasses the interactive
        artifact-selection UI. Ignored when ``remove_artifacts`` is False.
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
    abs_sessions = [os.path.abspath(s) for s in sessions]
    for s in abs_sessions:
        if not os.path.isdir(s):
            raise ValueError(f"Session path does not exist or is not a directory: {s}")
        parent = os.path.dirname(s)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {s!r}, expected {base_dir!r}"
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
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Inject combine_data
    input_params["combine_data"] = combine_data

    # Inject artifact removal parameters
    input_params["removeArtifacts"] = remove_artifacts
    if artifact_removal_method is not None:
        input_params["artifactsRemovalMethod"] = artifact_removal_method

    # Inject z-score parameters
    input_params["zscore_method"] = zscore_method
    input_params["baselineWindowStart"] = baseline_window_start
    input_params["baselineWindowEnd"] = baseline_window_end

    # Inject isosbestic_control
    input_params["isosbestic_control"] = isosbestic_control

    # Per-session output-directory subset filter — every session must have at least one run name.
    normalized_selected_runs = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["selectedOutputs"] = normalized_selected_runs

    # Write artifact coordinates into each output directory so that the artifact
    # removal worker can find them without the interactive selection UI.
    if remove_artifacts and artifact_coords:
        for session in abs_sessions:
            for output_dir in select_output_dirs(session, normalized_selected_runs[session]):
                for pair_name, coords in artifact_coords.items():
                    np.save(os.path.join(output_dir, f"coordsForPreProcessing_{pair_name}.npy"), coords)

    # Call the underlying Step 4 worker directly (no subprocess)
    extractTsAndSignal(input_params)


def step5(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
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
    selected_runs: dict[str, list[str]],
    group_selected_runs: dict[str, list[str]] | None = None,
) -> None:
    """
    Run pipeline Step 5 (PSTH Computation) via the Panel-backed logic, headlessly.

    This builds the template headlessly (using ``GUPPY_BASE_DIR`` to bypass
    the folder dialog), sets the FileSelector to ``selected_folders``, retrieves
    the full input parameters via ``getInputParameters()``, and calls the
    underlying worker ``guppy.computePsth.psthForEachStorename(input_params)`` that the
    UI normally launches via subprocess. No GUI is spawned.

    Parameters
    ----------
    base_dir : str
        Root directory used to initialize the FileSelector. All ``selected_folders``
        must reside directly under this path.
    selected_folders : Iterable[str]
        Absolute paths to the session directories to process.
    npm_timestamp_column_names : list[str | None] | None
        List of timestamp column names for NPM files, one per CSV file. None if not applicable.
    npm_time_units : list[str] | None
        List of time units for NPM files, one per CSV file (e.g., 'seconds', 'milliseconds'). None if not applicable.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files, one per CSV file. None if not applicable.
    combine_data : bool
        Whether to enable combined-session processing mode in Step 5. Defaults to False.
    compute_corr : bool
        Whether to compute cross-correlation between signals. Defaults to False.
    average_for_group : bool
        Whether to run group-level averaging across sessions instead of per-session PSTH
        computation. When ``True``, individual PSTH files must already exist in each session's
        output directory, and results are written to ``<base_dir>/average/``. Defaults to False.
    group_folders : list[str] | None
        Absolute paths to the session directories to include in group averaging. Only used
        when ``average_for_group`` is ``True``. Injected as ``folderNamesForAvg`` in
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
    abs_sessions = [os.path.abspath(s) for s in sessions]
    for s in abs_sessions:
        if not os.path.isdir(s):
            raise ValueError(f"Session path does not exist or is not a directory: {s}")
        parent = os.path.dirname(s)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {s!r}, expected {base_dir!r}"
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
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Inject combine_data
    input_params["combine_data"] = combine_data

    # Inject cross-correlation flag
    input_params["computeCorr"] = compute_corr

    # Inject group analysis parameters
    input_params["averageForGroup"] = average_for_group
    abs_group_folders = [os.path.abspath(f) for f in group_folders] if group_folders else []
    input_params["folderNamesForAvg"] = abs_group_folders

    # Per-session output-directory subset filter for individual + group analysis
    input_params["selectedOutputs"] = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["groupSelectedOutputs"] = _normalize_group_selected_runs(group_selected_runs, abs_group_folders)

    # Inject signal-type selection parameters
    input_params["selectForComputePsth"] = select_for_compute_psth
    input_params["selectForTransientsComputation"] = select_for_transients

    # Override parallelism — default 1 keeps tests single-process
    input_params["numberOfCores"] = number_of_cores

    # Inject PSTH binning parameters
    input_params["bin_psth_trials"] = bin_psth_trials
    input_params["use_time_or_trials"] = use_time_or_trials

    # Call the underlying Step 5 worker directly (no subprocess)
    psthForEachStorename(input_params)

    # Also compute frequency/amplitude and transients occurrences (normally triggered by CLI main)
    executeFindFreqAndAmp(input_params)


def step6(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
    npm_split_events: list[bool] | None = None,
    visualize_zscore_or_dff: str = "z_score",
    selected_runs: dict[str, list[str]],
    group_selected_runs: dict[str, list[str]] | None = None,
) -> None:
    """
    Run pipeline Step 6 (Visualize Results) via the Panel-backed logic, headlessly.

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
    npm_timestamp_column_names : list[str | None] | None
        List of timestamp column names for NPM files, one per CSV file. None if not applicable.
    npm_time_units : list[str] | None
        List of time units for NPM files, one per CSV file. None if not applicable.
    npm_split_events : list[bool] | None
        List of booleans indicating whether to split events for NPM files. None if not applicable.
    visualize_zscore_or_dff : str
        Signal type to visualize. One of ``'z_score'`` or ``'dff'``. Defaults to ``'z_score'``.

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
    abs_sessions = [os.path.abspath(s) for s in sessions]
    for s in abs_sessions:
        if not os.path.isdir(s):
            raise ValueError(f"Session path does not exist or is not a directory: {s}")
        parent = os.path.dirname(s)
        if parent != base_dir:
            raise ValueError(
                f"All selected_folders must share the same parent equal to base_dir. "
                f"Got parent {parent!r} for session {s!r}, expected {base_dir!r}"
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
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Inject visualization signal-type selection
    input_params["visualize_zscore_or_dff"] = visualize_zscore_or_dff

    # Per-session output-directory subset filter for individual + group visualization
    input_params["selectedOutputs"] = _normalize_selected_runs(selected_runs, abs_sessions)
    input_params["groupSelectedOutputs"] = _normalize_group_selected_runs(
        group_selected_runs,
        [os.path.abspath(f) for f in (input_params.get("folderNamesForAvg") or [])],
    )

    # Call the underlying Step 6 worker directly (no subprocess)
    visualizeResults(input_params)
