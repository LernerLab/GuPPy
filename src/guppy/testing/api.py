"""
Python API for GuPPy pipeline steps.

Step 1: Save Input Parameters
- Writes GuPPyParamtersUsed.json into each selected data folder.
- Mirrors the Panel UI's Step 1 behavior without invoking any UI by default.

This module is intentionally minimal and non-invasive.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

import numpy as np

from guppy.orchestration.home import build_homepage
from guppy.orchestration.preprocess import extractTsAndSignal
from guppy.orchestration.psth import psthForEachStorename
from guppy.orchestration.read_raw_data import orchestrate_read_raw_data
from guppy.orchestration.storenames import orchestrate_storenames_page
from guppy.orchestration.transients import executeFindFreqAndAmp


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
            raise ValueError(f"Invalid storename key: {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Invalid semantic name for key {k!r}: {v!r}")

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

    # Add npm parameters
    input_params["npm_timestamp_column_names"] = npm_timestamp_column_names
    input_params["npm_time_units"] = npm_time_units
    input_params["npm_split_events"] = npm_split_events

    # Call the underlying Step 2 executor (now headless-aware)
    orchestrate_storenames_page(input_params)


def step3(
    *,
    base_dir: str,
    selected_folders: Iterable[str],
    npm_timestamp_column_names: list[str | None] | None = None,
    npm_time_units: list[str] | None = None,
    npm_split_events: list[bool] | None = None,
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

    # Write artifact coordinates into each output directory so that the artifact
    # removal worker can find them without the interactive selection UI.
    if remove_artifacts and artifact_coords:
        for session in abs_sessions:
            for output_dir in glob.glob(os.path.join(session, "*_output_*")):
                if os.path.isdir(output_dir):
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
    compute_corr: bool = False,
    average_for_group: bool = False,
    group_folders: list[str] | None = None,
    select_for_compute_psth: str = "z_score",
    select_for_transients: str = "z_score",
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

    # Inject cross-correlation flag
    input_params["computeCorr"] = compute_corr

    # Inject group analysis parameters
    input_params["averageForGroup"] = average_for_group
    input_params["folderNamesForAvg"] = [os.path.abspath(f) for f in group_folders] if group_folders else []

    # Inject signal-type selection parameters
    input_params["selectForComputePsth"] = select_for_compute_psth
    input_params["selectForTransientsComputation"] = select_for_transients

    # Call the underlying Step 5 worker directly (no subprocess)
    psthForEachStorename(input_params)

    # Also compute frequency/amplitude and transients occurrences (normally triggered by CLI main)
    executeFindFreqAndAmp(input_params)
