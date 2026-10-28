"""Tonic Analysis step: an interactive page shown on the homepage.

Clicking its sidebar button shows a page composed from the Step-3 outputs already on disk.
Averaging those traces over the windows is a read-only reduction, so saving on that page both
writes the definitions (``tonic_epochs_<recording_site>.csv``) and computes the per-epoch means
(``tonic_<recording_site>.h5``) — no further pipeline step is needed.
"""

from ..frontend.tonic_epochs import build_tonic_epoch_page
from ..utils import page_links
from ..utils.utils import resolve_run_folders
from ..utils.validation import validate_preprocessing_outputs_present


def orchestrate_tonic_analysis(inputParameters: dict[str, object]) -> None:
    """Offer the Tonic Analysis page for the selected sessions.

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
    validate_preprocessing_outputs_present(run_folders=run_folders, action="running tonic analysis")
    page_links.offer_page(name="Tonic Analysis", content=build_tonic_epoch_page(run_folders=run_folders))
