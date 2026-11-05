"""Step-4 (transient-analysis) result view.

When the PSTH/transients run finishes, ``home.py`` enables the step's Show results button, whose
click calls :func:`open_transients_view` to show the detected transient peaks.
"""

from ..frontend.transient_peaks import build_peaks_view_page
from ..utils import page_links
from ..utils.utils import resolve_run_folders


def open_transients_view(session_folders: list, inputParameters: dict) -> None:
    """Offer the transient peaks page for the selected runs."""
    run_folders = resolve_run_folders(session_folders, inputParameters)
    page_links.offer_page(
        name="Transient peaks",
        content=build_peaks_view_page(run_folders, inputParameters["selectForTransientsComputation"]),
    )
