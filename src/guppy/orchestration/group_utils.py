import logging

import numpy as np

from ..utils.utils import select_output_dirs

logger = logging.getLogger(__name__)


def gather_group_stores_list_paths(inputParameters: dict[str, object], folderNamesForAvg: list[str]) -> np.ndarray:
    """Validate the group-averaging folder selection and gather its selected output directories.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters. The optional ``"groupSelectedOutputs"`` entry maps
        each session folder to the specific output run(s) chosen for averaging.
    folderNamesForAvg : list of str
        Session folder paths selected for group averaging.

    Returns
    -------
    np.ndarray
        Flat array of the selected output directories across all group folders.

    Raises
    ------
    ValueError
        When ``folderNamesForAvg`` is empty.
    """
    if len(folderNamesForAvg) == 0:
        message = (
            "No folders selected for group averaging (folderNamesForAvg is empty in inputParameters). "
            "Select folders in the 'Group Folders for Averaging' picker before running the average step."
        )
        logger.error(message)
        raise ValueError(message)
    group_selected_outputs = inputParameters.get("groupSelectedOutputs") or {}
    storesListPath = []
    for i in range(len(folderNamesForAvg)):
        filepath = folderNamesForAvg[i]
        storesListPath.append(select_output_dirs(filepath, group_selected_outputs.get(filepath)))
    return np.concatenate(storesListPath)
