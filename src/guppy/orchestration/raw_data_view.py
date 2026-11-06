"""Step-2 (Read Raw Data) result view.

When the Read Raw Data job finishes, ``home.py`` opens this view and points the step's View
results button at it.
"""

from ..frontend.raw_data_view import build_raw_data_view_page
from ..utils import page_links
from ..utils.utils import select_run_folders


def open_raw_data_view(session_folders: list, inputParameters: dict) -> None:
    """Offer the raw data page for every selected run."""
    # Step 2 reads every run on its own, even when Combine Data later joins them.
    run_folders = [
        run_folder
        for session_folder in session_folders
        for run_folder in select_run_folders(session_folder, inputParameters=inputParameters)
    ]
    page_links.offer_page(name="Raw Data", content=build_raw_data_view_page(run_folders=run_folders))
