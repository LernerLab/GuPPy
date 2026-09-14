"""Settings GuPPy remembers between launches.

The input root folder and the output root folder belong to a project rather than
to one analysis: they hold still while the sessions and run names chosen inside
them change from run to run. Remembering the pair means a returning user does not
re-pick them, which is what lets the Root Folder Selection card stay out of the
way.
"""

import json
import logging
import os
from pathlib import Path

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

# Overrides where the settings live. Set it to keep a separate configuration for a sandbox
# or a second profile; the test suite sets it so a run never rewrites the developer's own.
SETTINGS_PATH_VARIABLE = "GUPPY_SETTINGS_PATH"

INPUT_ROOT_FOLDER_KEY = "input_root_folder"
OUTPUT_ROOT_FOLDER_KEY = "output_root_folder"


def settings_path() -> Path:
    """Return the path of the settings file.

    Returns
    -------
    Path
        The path named by ``GUPPY_SETTINGS_PATH``, or ``settings.json`` in the
        platform's per-user configuration directory, beside the log directory
        :mod:`guppy.logging_config` uses.
    """
    override = os.environ.get(SETTINGS_PATH_VARIABLE)
    if override:
        return Path(override)
    return Path(user_config_dir("guppy", "LernerLab")) / "settings.json"


def load_settings() -> dict[str, str]:
    """Return the remembered settings.

    Returns
    -------
    dict of {str: str}
        The stored settings, or an empty dict before anything has been stored.
    """
    path = settings_path()
    if not path.exists():
        return {}
    try:
        with path.open() as settings_file:
            stored = json.load(settings_file)
    except (json.JSONDecodeError, OSError):
        # The file is editable by hand and lives outside the install, so a damaged one
        # must not stop GuPPy from starting; the defaults it holds are a convenience.
        logger.warning("Ignoring unreadable settings file %s", path)
        return {}
    return {key: value for key, value in stored.items() if isinstance(value, str)}


def remember_root_folders(*, input_root_folder: str | None, output_root_folder: str | None) -> None:
    """Store the root folders to offer as defaults at the next launch.

    Parameters
    ----------
    input_root_folder : str or None
        Folder the session folders are selected inside.
    output_root_folder : str or None
        Folder the mirrored output tree is written into. ``None`` under the
        inside-the-session layout, which has no output root folder to remember.
    """
    stored = load_settings()
    if input_root_folder:
        stored[INPUT_ROOT_FOLDER_KEY] = str(input_root_folder)
    if output_root_folder:
        stored[OUTPUT_ROOT_FOLDER_KEY] = str(output_root_folder)

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as settings_file:
        json.dump(stored, settings_file, indent=2)


def remembered_root_folders() -> tuple[str | None, str | None]:
    """Return the remembered root folders that still exist on disk.

    A folder that has since been moved or deleted is reported as ``None``, so a
    stale setting leaves the form empty rather than pointing at nothing.

    Returns
    -------
    input_root_folder, output_root_folder : str or None
        The remembered folders.
    """
    stored = load_settings()

    def existing(key: str) -> str | None:
        value = stored.get(key)
        return value if value and Path(value).is_dir() else None

    return existing(INPUT_ROOT_FOLDER_KEY), existing(OUTPUT_ROOT_FOLDER_KEY)
