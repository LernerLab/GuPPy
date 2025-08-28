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
from typing import Iterable, List

from guppy.savingInputParameters import savingInputParameters






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
