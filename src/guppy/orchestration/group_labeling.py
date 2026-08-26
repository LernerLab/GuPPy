"""Label Groups step: define a group of output runs, without computing anything.

Clicking the sidebar button opens a Panel page in a new browser window where a group is
named, given a destination, and given its member runs. Saving writes only
``group_members.json`` into ``<destination>/<group_name>_group/`` — the group directory is
a definition at that point, exactly as a run folder holds ``storesList.csv`` before Step 2
fills it. The Group Analysis step then averages into whichever group directories are
selected on the homepage.
"""

import logging

from ..frontend.frontend_utils import scanPortsAndFind
from ..frontend.group_labeling import GroupLabelingPage, save_group_definition
from ..utils.utils import group_folder_for_group, is_headless, validate_group_name
from ..utils.validation import validate_group_member_run_folders

logger = logging.getLogger(__name__)


def orchestrate_group_labeling_page(inputParameters: dict[str, object]) -> None:
    """Open the Label Groups page, or write the definition directly when headless.

    Headless callers supply ``group_name``, ``group_destination_directory`` and
    ``group_member_run_folders``; the GUI collects the same three on the page.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``selected_group_folders`` to populate the
        page's edit list, and the headless keys above when present.

    Raises
    ------
    ValueError
        When the headless group name or member selection is invalid.
    """
    group_name = inputParameters.get("group_name")
    destination_directory = inputParameters.get("group_destination_directory")

    if is_headless() and group_name and destination_directory:
        validate_group_name(group_name)
        member_run_folders = list(inputParameters.get("group_member_run_folders") or [])
        validate_group_member_run_folders(member_run_folders=member_run_folders)
        group_folder = group_folder_for_group(destination_directory=destination_directory, group_name=group_name)
        save_group_definition(group_folder=group_folder, member_run_folders=member_run_folders)
        return

    page = GroupLabelingPage(
        start_path=inputParameters.get("abspath") or "",
        selected_group_folders=list(inputParameters.get("selected_group_folders") or []),
    )
    template = page.build_template()
    template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
