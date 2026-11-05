"""Remove Artifacts result view.

When the Remove Artifacts compute job finishes, ``home.py`` enables the step's Show results
button, whose click calls :func:`open_artifact_view` to compose the page from the on-disk outputs.
"""

from ..frontend.artifact_removal import build_artifact_review_page
from ..utils import page_links
from ..utils.utils import resolve_run_folders


def open_artifact_view(session_folders: list, inputParameters: dict) -> None:
    """Offer the artifact removal results page for the selected runs."""
    run_folders = resolve_run_folders(session_folders, inputParameters)
    page_links.offer_page(name="Artifact Removal", content=build_artifact_review_page(run_folders=run_folders))
