import logging

import numpy as np

from ..utils.utils import select_run_folders

logger = logging.getLogger(__name__)


def gather_group_run_folders(inputParameters: dict[str, object], group_session_folders: list[str]) -> np.ndarray:
    """Validate the group-averaging folder selection and gather its selected output directories.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters. The optional ``"group_selected_runs"`` entry maps
        each session folder to the specific output run(s) chosen for averaging.
    group_session_folders : list of str
        Session folder paths selected for group averaging.

    Returns
    -------
    np.ndarray
        Flat array of the selected output directories across all group folders.

    Raises
    ------
    ValueError
        When ``group_session_folders`` is empty.
    """
    if len(group_session_folders) == 0:
        message = (
            "No folders selected for group averaging (group_session_folders is empty in inputParameters). "
            "Select folders in the 'Group Folders for Averaging' picker before running the average step."
        )
        logger.error(message)
        raise ValueError(message)
    group_selected_runs = inputParameters.get("group_selected_runs") or {}
    run_folders = []
    for i in range(len(group_session_folders)):
        filepath = group_session_folders[i]
        run_folders.append(select_run_folders(filepath, group_selected_runs.get(filepath)))
    return np.concatenate(run_folders)
