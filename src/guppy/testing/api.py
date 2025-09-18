"""
Python API for GuPPy pipeline steps.

Step 1: Save Input Parameters
- Writes GuPPyParamtersUsed.json into each selected data folder.
- Mirrors the Panel UI's Step 1 behavior without invoking any UI by default.

This module is intentionally minimal and non-invasive.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Iterable, List

from guppy.savingInputParameters import savingInputParameters
from guppy.saveStoresList import execute
from guppy.readTevTsq import readRawData
from guppy.preprocess import extractTsAndSignal
from guppy.computePsth import psthForEachStorename
from guppy.findTransientsFreqAndAmp import executeFindFreqAndAmp






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
    template = savingInputParameters()

    # Sanity checks: ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "onclickProcess" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'onclickProcess' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and trigger actual step-1 logic
    template._widgets["files_1"].value = list(selected_folders)
    template._hooks["onclickProcess"]()


def step2(*, base_dir: str, selected_folders: Iterable[str], storenames_map: dict[str, str], npm_timestamp_column_name: str | None = None, npm_time_unit: str = "seconds", npm_split_events: bool = True) -> None:
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
    template = savingInputParameters()

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
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Call the underlying Step 2 executor (now headless-aware)
    execute(input_params)


def step3(*, base_dir: str, selected_folders: Iterable[str], npm_timestamp_column_name: str | None = None, npm_time_unit: str = "seconds", npm_split_events: bool = True) -> None:
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
    template = savingInputParameters()

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters (match Step 2 style)
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Call the underlying Step 3 worker directly (no subprocess)
    readRawData(input_params)


def step4(*, base_dir: str, selected_folders: Iterable[str], npm_timestamp_column_name: str | None = None, npm_time_unit: str = "seconds", npm_split_events: bool = True) -> None:
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
    template = savingInputParameters()

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters (match Step 2 style)
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Call the underlying Step 4 worker directly (no subprocess)
    extractTsAndSignal(input_params)


def step5(*, base_dir: str, selected_folders: Iterable[str], npm_timestamp_column_name: str | None = None, npm_time_unit: str = "seconds", npm_split_events: bool = True) -> None:
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
    template = savingInputParameters()

    # Ensure hooks/widgets exposed
    if not hasattr(template, "_hooks") or "getInputParameters" not in template._hooks:
        raise RuntimeError("savingInputParameters did not expose 'getInputParameters' hook")
    if not hasattr(template, "_widgets") or "files_1" not in template._widgets:
        raise RuntimeError("savingInputParameters did not expose 'files_1' widget")

    # Select folders and fetch input parameters
    template._widgets["files_1"].value = abs_sessions
    input_params = template._hooks["getInputParameters"]()

    # Inject explicit NPM parameters (match Step 2 style)
    input_params["npm_timestamp_column_name"] = npm_timestamp_column_name
    input_params["npm_time_unit"] = npm_time_unit
    input_params["npm_split_events"] = npm_split_events

    # Call the underlying Step 5 worker directly (no subprocess)
    psthForEachStorename(input_params)

    # Also compute frequency/amplitude and transients occurrences (normally triggered by CLI main)
    executeFindFreqAndAmp(input_params)
