"""Remove Artifacts result view, served by the persistent main app.

The Remove Artifacts compute job runs on a background thread of the main app process and
writes its outputs to disk. When it finishes, ``home.py`` calls
:func:`open_artifact_view`, which opens a browser tab on the main app's
``/artifact-view`` route (served by :data:`build_artifact_view`). See
``orchestration/step_view.py`` for the shared token/registry/serving plumbing.
"""

from .step_view import StepView, resolve_run_folders
from ..frontend.artifact_removal import build_artifact_review_page


def _build_page(session_folders: list, inputParameters: dict) -> "object":
    run_folders = resolve_run_folders(session_folders, inputParameters)
    return build_artifact_review_page(run_folders=run_folders)


_view = StepView(route="artifact-view", title="GuPPy — Artifact Removal", build_page=_build_page)

# Route factory for main.py's route map, and completion hook for home.py.
build_artifact_view = _view.route_factory
open_artifact_view = _view.open
