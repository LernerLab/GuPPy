"""Step-3 (preprocessing) result view.

When the preprocessing compute job finishes, ``home.py`` enables the step's Show results button,
whose click calls :func:`open_preprocess_view` to compose the page from the on-disk outputs.
"""

from ..frontend.artifact_removal import build_preprocess_view_page
from ..utils import page_links
from ..utils.utils import resolve_run_folders


def open_preprocess_view(session_folders: list, inputParameters: dict) -> None:
    """Offer the preprocessing results page for the selected runs."""
    run_folders = resolve_run_folders(session_folders, inputParameters)
    page_links.offer_page(name="Preprocessing", content=build_preprocess_view_page(run_folders=run_folders))
