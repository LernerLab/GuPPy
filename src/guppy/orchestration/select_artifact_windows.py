"""Select Artifact Windows step: an interactive page shown on the homepage.

This optional step runs no compute. Clicking its sidebar button shows a page composed from the
Step-3 outputs already on disk. Saving on that page writes
``coordsForPreProcessing_<recording_site>.npy`` and records the removal method, which the Remove
Artifacts step then consumes.
"""

from ..frontend.artifact_windows_page import build_artifact_window_page
from ..utils import page_links
from ..utils.utils import resolve_run_folders
from ..utils.validation import validate_preprocessing_outputs_present


def orchestrate_select_artifact_windows(inputParameters: dict[str, object]) -> None:
    """Offer the Select Artifact Windows page for the selected sessions.

    Parameters
    ----------
    inputParameters : dict
        Full pipeline input parameters; uses ``session_folders``, ``selected_runs``,
        and ``combine_data``.

    Raises
    ------
    ValueError
        If any selected run has no Step-3 preprocessing outputs.
    """
    run_folders = resolve_run_folders(inputParameters["session_folders"], inputParameters)
    validate_preprocessing_outputs_present(run_folders=run_folders)
    page_links.offer_page(name="Select Artifact Windows", content=build_artifact_window_page(run_folders=run_folders))
