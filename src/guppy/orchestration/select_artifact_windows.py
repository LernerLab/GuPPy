"""Select Artifact Windows step: an interactive page served by the persistent main app.

This optional step runs no compute. Clicking its sidebar button opens a browser tab on the
main app's ``/select-artifact-windows`` route (served by
:data:`build_select_artifact_windows_view`), composed from the Step-3 outputs already on
disk. Saving on that page writes ``coordsForPreProcessing_<recording_site>.npy`` and
records the removal method, which the Remove Artifacts step then consumes. See
``orchestration/step_view.py`` for the shared token/registry/serving plumbing.
"""

import logging

from .step_view import StepView, resolve_run_folders
from ..frontend.artifact_windows_page import build_artifact_window_page
from ..utils.validation import validate_preprocessing_outputs_present

logger = logging.getLogger(__name__)


def _build_page(session_folders: list, inputParameters: dict) -> "object":
    run_folders = resolve_run_folders(session_folders, inputParameters)
    validate_preprocessing_outputs_present(run_folders=run_folders)
    return build_artifact_window_page(run_folders=run_folders)


_view = StepView(route="select-artifact-windows", title="GuPPy — Select Artifact Windows", build_page=_build_page)

# Route factory for main.py's route map, and open hook for home.py.
build_select_artifact_windows_view = _view.route_factory
open_select_artifact_windows_view = _view.open


def orchestrate_select_artifact_windows(inputParameters: dict[str, object]) -> None:
    """Open the Select Artifact Windows page for the selected sessions.

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
    session_folders = inputParameters["session_folders"]
    validate_preprocessing_outputs_present(run_folders=resolve_run_folders(session_folders, inputParameters))
    open_select_artifact_windows_view(session_folders, inputParameters)
