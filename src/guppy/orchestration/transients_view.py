"""Step-4 (transient-analysis) result view, served by the persistent main app.

The transient-analysis compute job runs as a subprocess and writes its outputs to disk.
When the PSTH/transients run finishes, ``home.py`` calls :func:`open_transients_view`,
which opens a browser tab on the main app's ``/transients-view`` route (served by
:data:`build_transients_view`) showing the detected transient peaks. See
``orchestration/step_view.py`` for the shared token/registry/serving plumbing.
"""

from .step_view import StepView, resolve_run_folders
from ..frontend.transient_peaks import build_peaks_view_page


def _build_page(session_folders: list, inputParameters: dict) -> "object":
    run_folders = resolve_run_folders(session_folders, inputParameters)
    return build_peaks_view_page(run_folders, inputParameters["selectForTransientsComputation"])


_view = StepView(route="transients-view", title="GuPPy — Transient peaks", build_page=_build_page)

# Route factory for main.py's route map, and completion hook for home.py.
build_transients_view = _view.route_factory
open_transients_view = _view.open
