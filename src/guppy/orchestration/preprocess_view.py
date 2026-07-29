"""Step-3 (preprocessing) result view, served by the persistent main app.

The preprocessing compute job runs on a background thread of the main app process and writes
its outputs to disk. When it finishes, ``home.py`` calls :func:`open_preprocess_view`, which opens a browser tab on the
main app's ``/preprocess-view`` route (served by :data:`build_preprocess_view`). The page
is composed from the on-disk outputs. See ``orchestration/step_view.py`` for the shared
token/registry/serving plumbing.
"""

from .step_view import StepView, resolve_run_folders
from ..frontend.artifact_removal import build_preprocess_view_page


def _build_page(session_folders: list, inputParameters: dict) -> "object":
    run_folders = resolve_run_folders(session_folders, inputParameters)
    return build_preprocess_view_page(
        run_folders, inputParameters["removeArtifacts"], inputParameters["plot_zScore_dff"]
    )


_view = StepView(route="preprocess-view", title="GuPPy — Preprocessing", build_page=_build_page)

# Route factory for main.py's route map, and completion hook for home.py.
build_preprocess_view = _view.route_factory
open_preprocess_view = _view.open
