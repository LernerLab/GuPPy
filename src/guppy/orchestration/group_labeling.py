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
from ..frontend.group_labeling import GroupLabelingPage

logger = logging.getLogger(__name__)


def build_group_labeling_page(*, inputParameters: dict[str, object]) -> GroupLabelingPage:
    """Construct the Label Groups page from the pipeline input parameters.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``abspath`` as the page's starting
        directory and ``selected_group_folders`` to populate its edit list.

    Returns
    -------
    GroupLabelingPage
        The constructed page, not yet served.
    """
    return GroupLabelingPage(
        start_path=inputParameters.get("abspath") or "",
        selected_group_folders=list(inputParameters.get("selected_group_folders") or []),
    )


def orchestrate_group_labeling_page(inputParameters: dict[str, object]) -> None:
    """Open the Label Groups page in a new browser window.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; see :func:`build_group_labeling_page`.
    """
    page = build_group_labeling_page(inputParameters=inputParameters)
    template = page.build_template()
    template.show(port=scanPortsAndFind(start_port=5000, end_port=5200))
